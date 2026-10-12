"""Install the NVIDIA NVFP4 model and the W4A16 R256 runtime on Linux."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from download_model import download, verify
from vendor import fetch

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT, **kwargs)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "runtime")
    ap.add_argument("--skip-download", action="store_true", help="verify an existing complete checkpoint")
    ap.add_argument("--reuse-pack", type=Path, help="use an already prepared NVFP4 pack")
    ap.add_argument("--reuse-mtp", type=Path, help="use an already prepared MTP runtime")
    ap.add_argument("--reuse-vision", type=Path, help="directory with mmproj-bf16.gguf and tokenizer-only.gguf")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--parallel", type=int, choices=(1, 2), default=2)
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--config", type=Path, default=ROOT / "strata.json")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--check", action="store_true", help="read-only hardware and dependency check")
    return ap


def check(args):
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise RuntimeError("This installer requires Linux and Python 3.12+")
    for command in ("cmake", "ninja", "c++", "nvidia-smi"):
        if not shutil.which(command):
            raise RuntimeError(f"Missing {command}; see docs/INSTALL.md")
    if not (args.cuda_home / "bin/nvcc").is_file():
        raise RuntimeError(f"CUDA compiler missing at {args.cuda_home}; use --cuda-home")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], text=True)
    print(gpu.strip())
    rows = [line.split(",") for line in gpu.strip().splitlines()]
    if len(rows) != 1 or rows[0][2].strip() != "8.9" or float(rows[0][1]) < 46000:
        raise RuntimeError("The bundled 256K/2-slot configuration targets one SM89 GPU with 48GB VRAM")
    mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")}
    if mem["MemTotal"] < 60 * 1024 * 1024:
        raise RuntimeError("64GB system RAM required for this configuration")
    if args.jobs < 1 or not 512 <= args.context <= 262144 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid jobs, context or port")
    return gpu


def stage(script, arguments):
    run([sys.executable, "-u", ROOT / "tools" / script, *arguments],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})


def report_ok(path, fingerprint):
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    if not report.get("passed") or report.get("source_index_sha256") != fingerprint:
        raise ValueError("Incomplete or different-model data: " + str(path))
    return True


def prepare(args):
    from modelopt_source import Source
    from prepare_assets import tokenizer, vision
    source = Source(args.model_dir)
    pack = args.reuse_pack or args.data_dir / "packs/nvidia-nvfp4-raw"
    mtp = args.reuse_mtp or args.data_dir / "mtp"
    image = args.reuse_vision or args.data_dir / "vision"
    pack.mkdir(parents=True, exist_ok=True)
    if not args.reuse_pack:
        if not report_ok(pack / "experts-repack-report.json", source.fingerprint):
            stage("pack_modelopt_experts.py", ["--model", args.model_dir, "--out", pack])
        if not report_ok(pack / "dense-prepare-report.json", source.fingerprint):
            dense = pack / ".dense-stage"
            if dense.exists():
                raise RuntimeError(f"Interrupted dense export remains at {dense}; move it aside before retrying")
            stage("pack_modelopt_dense.py", ["--model", args.model_dir, "--out", dense, "--dense-q8"])
            for p in dense.iterdir():
                if (pack / p.name).exists():
                    raise FileExistsError(pack / p.name)
            for p in dense.iterdir():
                p.rename(pack / p.name)
            dense.rmdir()
        if not (pack / "expert-profile.bin").exists():
            shutil.copyfile(ROOT / "data/expert-profile.bin", pack / "expert-profile.bin")
        tokenizer(pack)
    for name in ("experts.bin", "native_experts.txt", "dense.bin", "index.txt", "model.stidx", "ple.stidx", "expert-profile.bin", "tokenizer/tokenizer.json"):
        if not (pack / name).is_file():
            raise FileNotFoundError(pack / name)
    if not args.reuse_mtp and not report_ok(mtp / "prepare-report.json", source.fingerprint):
        staging = mtp.with_name(mtp.name + ".stage")
        if staging.exists():
            raise RuntimeError(f"Interrupted MTP export remains at {staging}; move it aside before retrying")
        stage("pack_modelopt_mtp.py", ["--model", args.model_dir, "--out", staging, "--workers", args.jobs])
        if mtp.exists():
            raise FileExistsError(mtp)
        staging.rename(mtp)
    for name in ("experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin"):
        if not (mtp / name).is_file():
            raise FileNotFoundError(mtp / name)
    if not args.no_vision:
        if not args.reuse_vision and not report_ok(image / "prepare-report.json", source.fingerprint):
            vision(args.model_dir, pack, image)
        for name in ("mmproj-bf16.gguf", "tokenizer-only.gguf"):
            if not (image / name).is_file():
                raise FileNotFoundError(image / name)
    return pack, mtp, image


def config(args, pack, mtp, vision):
    env = dict(STRATA_NVFP4_W4A16="1", STRATA_PREFILL_NVFP4="fp16", STRATA_PREFILL_RING="256",
               STRATA_PREFILL_CPU_SHARE="0", STRATA_PCIE_BALANCE="0", STRATA_AUX_CPUS="auto",
               STRATA_WATCHDOG_S="180", STRATA_PLE_MMAP_RANDOM="1", STRATA_NVFP4_FUSED_MOE="2",
               STRATA_PREFILL_DEQUANT_AHEAD="1", STRATA_NVFP4_DEQUANT_FAST="1",
               STRATA_PREFILL_COMPACT_GUH="0", STRATA_PREFILL_NVFP4_FP8="0")
    native = ["--pack", str(pack), "--native", str(pack / "model.stidx"), "--ple-gguf", str(pack / "ple.stidx"),
              "--ple-io", "mmap", "--expert-profile", str(pack / "expert-profile.bin"), "--expert-cache", "auto",
              "--resident-experts", "--prefill", "8192", "--spec", "4", "--max-context", str(args.context),
              "--kv", "int8", "--pcie-frac", "0", "--vram-reserve-mib", "1536", "--mtp", str(mtp),
              "--batch-mtp", "--spec-min-p", "0.7", "--prompt-cache", "6", "--prompt-cache-tail",
              "--mtp-draft-vocab", str(mtp / "draft_vocab.bin")]
    result = dict(exe=str(ROOT / "build/strata"), args=native, env=env, cwd=str(ROOT),
                  tokenizer=str(pack / "tokenizer"), model_name="Qwen3.8-flash-next", log=str(ROOT / "logs/engine.log"),
                  host="127.0.0.1", port=args.port, gpu=0, parallel=args.parallel, open_browser=False,
                  lib_dirs=[str(args.cuda_home / "lib64")], tool_call_recovery=True, reasoning_loop_recovery="recover")
    if not args.no_vision:
        native.append("--vision")
        result["vision"] = dict(exe=str(ROOT / "build-vision/bin/strata-vision"), mmproj=str(vision / "mmproj-bf16.gguf"),
                                model=str(vision / "tokenizer-only.gguf"), gpu=True, max_tokens=1024)
    return result


def main():
    args = parser().parse_args()
    for key in ("model_dir", "data_dir", "cuda_home", "config", "reuse_pack", "reuse_mtp", "reuse_vision"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    check(args)
    if args.check:
        return
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    data_needed = 80 * 1024**3 if not args.reuse_pack else 0
    model_needed = 0
    if not args.skip_download:
        manifest = json.loads((ROOT / "data/checkpoint-manifest.json").read_text())
        model_needed = sum(max(0, p["size"] - (args.model_dir / p["path"]).stat().st_size)
                           if (args.model_dir / p["path"]).is_file() else p["size"] for p in manifest["files"])
    if args.model_dir.stat().st_dev == args.data_dir.stat().st_dev:
        data_needed += model_needed
        model_needed = 0
    if (shutil.disk_usage(args.data_dir).free < data_needed or
            shutil.disk_usage(args.model_dir).free < model_needed):
        raise RuntimeError("Insufficient disk space; see docs/INSTALL.md")
    llama = fetch()
    verify(args.model_dir) if args.skip_download else download(args.source, args.model_dir)
    env = {**os.environ, "CUDACXX": str(args.cuda_home / "bin/nvcc"), "PATH": str(args.cuda_home / "bin") + ":" + os.environ["PATH"]}
    run(["cmake", "-S", ROOT, "-B", ROOT / "build", "-G", "Ninja", "-DSTRATA_ENABLE_CUDA=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=89", "-DSTRATA_BUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release", f"-DSTRATA_GGML_DIR={llama}"], env=env)
    run(["cmake", "--build", ROOT / "build", "--target", "strata", "-j", args.jobs])
    if not args.no_vision:
        run(["cmake", "-S", ROOT / "tools/vision", "-B", ROOT / "build-vision", "-G", "Ninja",
             f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DSTRATA_PORTABLE=OFF"], env=env)
        run(["cmake", "--build", ROOT / "build-vision", "--target", "strata-vision", "-j", args.jobs])
    pack, mtp, image = prepare(args)
    desired = json.dumps(config(args, pack, mtp, image), indent=2) + "\n"
    if args.config.exists() and args.config.read_text() != desired:
        raise FileExistsError(f"Configuration already exists: {args.config}; use --config for a new file")
    (ROOT / "logs").mkdir(exist_ok=True)
    if not args.config.exists():
        with args.config.open("x") as f:
            f.write(desired)
    print("Installed. Start with ./start.sh", flush=True)
    if args.start:
        os.execv(sys.executable, [sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(args.config)])


if __name__ == "__main__":
    main()
