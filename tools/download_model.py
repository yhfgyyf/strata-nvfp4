"""Download either mirror and verify it against the pinned NVIDIA checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODELSCOPE = "nv-community/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
ROOT = Path(__file__).resolve().parents[1]


def verify(path):
    manifest = json.loads((ROOT / "data/checkpoint-manifest.json").read_text())
    for record in manifest["files"]:
        if record["path"] == ".gitattributes":
            continue
        file = path / record["path"]
        if not file.is_file() or file.stat().st_size != record["size"]:
            raise ValueError(f"Missing or incomplete checkpoint file: {file}")
        if record.get("lfs"):
            with file.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            expected = record["lfs"]["sha256"]
        else:
            content = file.read_bytes()
            actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            expected = record["git_blob_id"]
        if actual != expected:
            raise ValueError(f"Checkpoint hash mismatch: {record['path']}")
        print("Verified", record["path"], flush=True)
    return path


def download(source, path):
    path = path.resolve()
    if source == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=path)
    else:
        from modelscope import snapshot_download
        snapshot_download(MODELSCOPE, revision="master", local_dir=str(path))
    return verify(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    print(verify(a.model_dir.resolve()) if a.verify_only else download(a.source, a.model_dir))
