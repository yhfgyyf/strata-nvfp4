"""Prepare Strata's dense pack and metadata indices without a GGUF weights file.

The dense compatibility policy is explicit: Q8_0 large projections, BF16 token
embedding/small projections, F32 norms, F16 PLE convolution. PLE remains the
original FP8 safetensors bytes with its per-tensor scale and shard order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "third_party/llama.cpp"
sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py"), str(ROOT / "tools")]

import gguf
from conversion.qwen4exp import Qwen4ExpTextModel
from iq_pack import form_of
from modelopt_source import Source, np, torch

Q = gguf.GGMLQuantizationType


def publish_index(writer, path):
    # Reuse the existing typed metadata serializer. STRI v1 is a metadata-only
    # index with an external payload; it is deliberately not a GGUF weights file.
    temporary = path.with_suffix(".tmp")
    writer.write_header_to_file(temporary)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()
    header = temporary.read_bytes()
    header += b"\0" * ((-len(header)) % 32)
    temporary.write_bytes(b"STRI" + struct.pack("<I", 1) + header[8:])
    temporary.replace(path)


def prepare(source, out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    dense = out / "dense.bin"
    if dense.exists():
        raise FileExistsError(dense)
    model = Qwen4ExpTextModel(source.path, gguf.LlamaFileType.MOSTLY_Q8_0, out / "model.stidx",
                             hparams=json.loads((source.path / "config.json").read_text()))
    writer = model.gguf_writer
    rows, records, offset = [], [], 0
    part = out / "dense.bin.part"
    if part.exists():
        raise FileExistsError(part)
    with part.open("wb") as handle:
        for name, header in sorted(source.headers.items()):
            if "visual." in name or name.startswith("mtp.") or ".mlp.experts." in name:
                continue
            if ".ple_embedding." in name or name.endswith(("_scale", "_scale_inv")):
                continue
            normalized = name.replace("language_model.", "")
            bid = re.search(r"\.layers\.(\d+)\.", normalized)
            for target, tensor in model.modify_tensors(source.values(name), normalized, int(bid[1]) if bid else None):
                form = form_of(target)
                kind = {"BF16": Q.BF16, "F32": Q.F32, "F16": Q.F16}.get(form, Q.Q8_0)
                if tensor.ndim <= 1:
                    kind = Q.F32
                if target == "token_embd.weight":
                    kind = Q.BF16
                values = tensor.contiguous().numpy()
                if kind == Q.F32:
                    blob = values.astype("<f4")
                elif kind == Q.F16:
                    blob = values.astype("<f2")
                else:
                    blob = gguf.quants.quantize(values, kind)
                assert handle.tell() == offset
                blob.tofile(handle)
                pad = (-blob.nbytes) % 32
                handle.write(b"\0" * pad)
                writer.add_tensor_info(target, tensor.shape, np.dtype(np.float32), blob.nbytes, raw_dtype=kind)
                ne0, ne1 = tensor.shape[-1], tensor.shape[-2] if tensor.ndim > 1 else 0
                if kind == Q.Q8_0:
                    row = [target, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7
                else:
                    index_kind = {Q.BF16: "4", Q.F16: "5", Q.F32: "2"}[kind]
                    row = [target, "0", index_kind, str(offset), str(blob.nbytes), "0", str(blob.nbytes),
                           str(ne0), str(ne1), "0", "0", "1"] + ["0"] * 7
                rows.append(row)
                records.append({"name": target, "source": name, "type": kind.name,
                                "offset": offset, "bytes": blob.nbytes,
                                "sha256": hashlib.sha256(blob.tobytes()).hexdigest()})
                offset += blob.nbytes + pad
                if len(records) % 64 == 0:
                    print("RAW_DENSE_PROGRESS", len(records), offset, flush=True)
    part.replace(dense)
    model._ple_row_dim = 160
    model.hparams["layer_types"] = ["full_attention" if x == "qwen_sparse_attention" else x
                                   for x in model.hparams["layer_types"]]
    model.prepare_metadata(vocab_only=False)
    writer.add_string("strata.raw.data_file", "dense.bin")
    writer.add_uint64("strata.raw.data_offset", 0)
    writer.add_string("strata.raw.source_index_sha256", source.fingerprint)
    publish_index(writer, out / "model.stidx")
    pool = 0
    for row in rows:
        row[5] = str(pool)
        pool += (int(row[6]) + 63) // 64 * 64
    text = f"# strata raw dense pack v3\n# align 64 pool {pool} tensors {len(rows)}\n"
    text += "".join(" ".join(row) + "\n" for row in rows)
    (out / "index.txt").write_text(text)

    ple_report = prepare_ple(source, out)
    report = {"passed": True, "source_index_sha256": source.fingerprint, "dense_bytes": offset,
              "dense_policy": "Q8_0 large projections; BF16 embedding/small; F32 norms; F16 PLE convolution",
              "tensors": records, "ple": ple_report}
    (out / "dense-prepare-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RAW_DENSE_READY", len(records), offset, "PLE reads original FP8", flush=True)


def prepare_ple(source, out):
    out.mkdir(parents=True, exist_ok=True)
    # Map FP8 PLE in place; preserve numerical-to-physical shard order.
    names = [f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{k}.weight" for k in range(128)]
    files = {source.index[name] for name in names}
    assert len(files) == 1
    filename = files.pop()
    physical = sorted(range(128), key=lambda k: source.headers[names[k]]["data_offsets"][0])
    row_map = [physical.index(k) for k in range(128)]
    first = source.headers[names[physical[0]]]["data_offsets"][0]
    for ordinal, k in enumerate(physical):
        h = source.headers[names[k]]
        assert h["dtype"] == "F8_E4M3" and h["shape"] == [2500012, 160]
        assert h["data_offsets"] == [first + ordinal * 2500012 * 160, first + (ordinal + 1) * 2500012 * 160]
    data_offset = source.bases[filename] + first
    ple_bytes = 320001536 * 160
    assert (source.path / filename).stat().st_size >= data_offset + ple_bytes
    scale_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    scale = float(source.get(scale_name).float().item())
    assert np.isfinite(scale) and scale > 0
    ple = gguf.GGUFWriter(None, "strata-ple")
    ple.add_string("strata.raw.data_file", str(source.path / filename))
    ple.add_uint64("strata.raw.data_offset", data_offset)
    ple.add_uint64("strata.ple.rows_per_shard", 2500012)
    ple.add_array("strata.ple.shard_map", row_map)
    ple.add_string("strata.ple.format", "f8_e4m3")
    ple.add_float32("strata.ple.scale", scale)
    ple.add_tensor_info("per_layer_token_embd.weight", (320001536, 160), np.dtype(np.float32),
                        ple_bytes, raw_dtype=Q.I8)
    # The checkpoint also stores MTP after PLE. Describe its actual first tensor
    # so the reader correctly validates an embedded table rather than requiring
    # the entire remaining safetensors file to consist solely of PLE.
    tail_name = "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    tail = source.headers[tail_name]
    assert source.index[tail_name] == filename and tail['dtype'] == 'BF16'
    assert tail['data_offsets'][0] == first + ple_bytes
    ple.add_tensor_info(tail_name, tail['shape'], np.dtype(np.float32),
                        tail['data_offsets'][1] - tail['data_offsets'][0], raw_dtype=Q.BF16)
    publish_index(ple, out / "ple.stidx")
    return {"source": str(source.path / filename), "dtype": "F8_E4M3", "modified": False,
            "scale": scale, "data_offset": data_offset, "bytes": ple_bytes,
            "physical_shard_map": row_map, "embedded_tail_tensor": tail_name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dense-q8", action="store_true",
                        help="explicitly choose Q8_0 for the dense compatibility projections")
    mode.add_argument("--ple-only", action="store_true", help="only index the original FP8 PLE shard")
    args = parser.parse_args()
    source = Source(args.model)
    if args.ple_only:
        print(json.dumps(prepare_ple(source, args.out)), flush=True)
    else:
        prepare(source, args.out)
