"""Fetch the same llama.cpp sources used by the engine and data tools."""
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
ZIP_SHA256 = "cbe23c594282ead2937abb3f008e51fcec4609d9256652fff42c7cc1c21ea47b"


def fetch():
    target = ROOT / "third_party/llama.cpp"
    marker = target / ".strata-pin"
    if target.exists():
        if marker.is_file() and marker.read_text().strip() == COMMIT:
            return target
        raise RuntimeError(f"Unverified dependency directory: {target}; move it aside before retrying")
    with tempfile.TemporaryDirectory(dir=ROOT / "third_party") as scratch:
        scratch = Path(scratch)
        archive = ROOT / "third_party" / f"llama.cpp-{COMMIT}.zip"
        if not archive.exists():
            temporary = scratch / "source.zip"
            print("Downloading llama.cpp", COMMIT, flush=True)
            urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.zip", temporary)
            temporary.replace(archive)
        with archive.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != ZIP_SHA256:
                raise ValueError("llama.cpp archive hash mismatch")
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Unsafe dependency archive entry")
            z.extractall(scratch)
        source = scratch / f"llama.cpp-{COMMIT}"
        (source / ".strata-pin").write_text(COMMIT + "\n")
        if target.exists():
            raise FileExistsError(target)
        shutil.move(source, target)
    return target


if __name__ == "__main__":
    print(fetch())
