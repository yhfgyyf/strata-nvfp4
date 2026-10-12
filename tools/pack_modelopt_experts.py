"""Lossless layout repack of NVIDIA ModelOpt NVFP4 routed experts.

No GGUF weights are generated. E2M1 codes and E4M3 group scales remain bit-exact.
ModelOpt weight_scale_2 is already the dequantization multiplier stored in the
Strata NVFP4 blob tail; it must not be inverted like compressed-tensors' scale.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from modelopt_source import Source, np, torch

N, FF, EXPERTS, LAYERS = 2560, 640, 512, 48
ROLE_BYTES = N * FF // 64 * 36
BLOB_BYTES = 3 * ROLE_BYTES + 16


def repack(source, stem):
    packed = source.get(stem + ".weight").numpy()
    scale_tensor = source.get(stem + ".weight_scale")
    assert scale_tensor.dtype == torch.float8_e4m3fn
    scales = scale_tensor.view(torch.uint8).numpy()
    global_scale = source.get(stem + ".weight_scale_2").float().item()
    if not math.isfinite(global_scale) or global_scale <= 0 or np.any(scales >= 127):
        raise ValueError(f"Invalid NVFP4 scales: {stem}")
    rows, half_cols = packed.shape
    assert scales.shape == (rows, half_cols // 8)
    codes = np.empty((rows, half_cols * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    groups = codes.reshape(rows, -1, 4, 16)
    blocks = np.empty((rows, half_cols // 32, 36), dtype=np.uint8)
    blocks[:, :, :4] = scales.reshape(rows, -1, 4)
    blocks[:, :, 4:] = (groups[..., :8] | (groups[..., 8:] << 4)).reshape(rows, -1, 32)
    return blocks, np.float32(global_scale)


def verify_repack(source, stem, blocks, multiplier):
    packed = source.get(stem + ".weight").numpy()
    reference_codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    reference_codes[:, 0::2] = packed & 15
    reference_codes[:, 1::2] = packed >> 4
    codes = blocks[:, :, 4:].reshape(blocks.shape[0], -1, 4, 8)
    restored = np.concatenate((codes & 15, codes >> 4), axis=-1).reshape(reference_codes.shape)
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    np.testing.assert_array_equal(restored, reference_codes)
    raw_scales = source.get(stem + ".weight_scale").view(torch.uint8).numpy()
    np.testing.assert_array_equal(blocks[:, :, :4].reshape(raw_scales.shape), raw_scales)
    scales = source.get(stem + ".weight_scale").float().repeat_interleave(16, 1)
    reference = torch.from_numpy(lut[reference_codes])
    expected = reference * scales * source.get(stem + ".weight_scale_2").float()
    got = torch.from_numpy(lut[restored]) * scales * float(multiplier)
    relative = torch.linalg.vector_norm(got - expected) / torch.linalg.vector_norm(expected)
    assert float(relative) == 0.0, stem
    return {"source": stem, "codes_exact": True, "group_scales_exact": True,
            "modelopt_scale_relative_rms": float(relative),
            "packed_sha256": hashlib.sha256(blocks.tobytes()).hexdigest()}


def experts(source, out, sample_only):
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for layer, expert in ((0, 0), (1, 511), (23, 127), (47, 511)):
        for role in ("gate", "up", "down"):
            stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
            blocks, inverse = repack(source, stem)
            checks.append(verify_repack(source, stem, blocks, inverse))
    report = {"source_index_sha256": source.fingerprint, "checks": checks,
              "quantization": "NVFP4 retained; code and scale layout only", "passed": True}
    (out / "experts-samples.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NVFP4_SAMPLES_PASS", len(checks), flush=True)
    sample = out / "sample-expert.bin"
    with sample.open("wb") as handle:
        tail = np.zeros(4, dtype="<f4")
        for index, role in enumerate(("gate", "up", "down")):
            stem = f"model.language_model.layers.0.mlp.experts.0.{role}_proj"
            blocks, tail[index] = repack(source, stem)
            blocks.tofile(handle)
        tail.tofile(handle)
    if sample_only:
        return
    final = out / "experts.bin"
    part = out / "experts.bin.part"
    ledger = out / "experts-repack-state.json"
    if final.exists():
        raise FileExistsError(final)
    state = {"source_index_sha256": source.fingerprint, "layers_completed": 0, "offset": 0}
    if ledger.exists():
        state = json.loads(ledger.read_text())
        assert state["source_index_sha256"] == source.fingerprint
        assert part.exists() and part.stat().st_size >= state["offset"]
    elif part.exists():
        raise ValueError("Partial expert pack has no resume ledger")
    else:
        part.touch(exist_ok=False)
        ledger.write_text(json.dumps(state) + "\n")
    import shutil
    expected_size = LAYERS * EXPERTS * BLOB_BYTES
    assert shutil.disk_usage(out).free > expected_size - state["offset"] + (3 << 30)
    start = time.monotonic()
    with part.open("r+b") as handle:
        handle.truncate(state["offset"])
        handle.seek(state["offset"])
        for layer in range(state["layers_completed"], LAYERS):
            for expert in range(EXPERTS):
                tail = np.zeros(4, dtype="<f4")
                for index, role in enumerate(("gate", "up", "down")):
                    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj"
                    blocks, tail[index] = repack(source, stem)
                    assert blocks.nbytes == ROLE_BYTES
                    blocks.tofile(handle)
                tail.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
            state.update(layers_completed=layer + 1, offset=handle.tell())
            temporary = ledger.with_suffix(".tmp")
            temporary.write_text(json.dumps(state) + "\n")
            temporary.replace(ledger)
            print("NVFP4_LAYER", layer + 1, f"{handle.tell() / 1e9:.3f}/{expected_size / 1e9:.3f} GB",
                  f"{time.monotonic() - start:.1f}s", flush=True)
    assert part.stat().st_size == expected_size
    part.replace(final)
    lines = [f"# strata native experts v3: raw NVFP4 blobs (n_expert {EXPERTS}, total {expected_size})"]
    lines += [f"{layer} 40 40 {layer * EXPERTS * BLOB_BYTES} {BLOB_BYTES}" for layer in range(LAYERS)]
    (out / "native_experts.txt").write_text("\n".join(lines) + "\n")
    report.update(bytes=expected_size, seconds=time.monotonic() - start, file=str(final))
    (out / "experts-repack-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NVFP4_EXPERTS_READY", final, expected_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    experts(Source(args.model), args.out, args.sample_only)


if __name__ == "__main__":
    main()
