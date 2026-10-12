"""Read-only, per-tensor mapping of this compressed-tensors checkpoint."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.index_bytes = (self.path / "model.safetensors.index.json").read_bytes()
        self.fingerprint = hashlib.sha256(self.index_bytes).hexdigest()
        self.index = json.loads(self.index_bytes)["weight_map"]
        self.headers, self.bases = {}, {}
        for filename in sorted(set(self.index.values())):
            file = self.path / filename
            with file.open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            self.bases[filename] = 8 + size
            for name, value in header.items():
                if name == "__metadata__":
                    continue
                start, end = value["data_offsets"]
                if start < 0 or end < start or 8 + size + end > file.stat().st_size:
                    raise ValueError(f"Invalid source tensor: {name}")
                self.headers[name] = value
        if set(self.index) != set(self.headers):
            raise ValueError("Source index and shard headers disagree")

    def get(self, name):
        filename = self.index[name]
        header = self.headers[name]
        start, end = header["data_offsets"]
        data = np.memmap(self.path / filename, mode="r", dtype=np.uint8,
                         offset=self.bases[filename] + start, shape=(end - start,))
        kind = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64,
                "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}[header["dtype"]]
        return torch.from_numpy(data).view(kind).reshape(header["shape"])

    def values(self, name):
        value = self.get(name).float()
        if self.headers[name]["dtype"] == "F8_E4M3":
            value = value * self.get(name + "_scale").float()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite source values: {name}")
        return value
