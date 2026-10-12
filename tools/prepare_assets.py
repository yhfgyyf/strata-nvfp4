"""Export the tokenizer, draft vocabulary and image encoder data."""
import argparse
import json
import re
import struct
from pathlib import Path

import regex
from modelopt_source import Source, STRATA, np, torch
import sys

LLAMA = STRATA / "third_party/llama.cpp"
sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
import gguf
from conversion.qwen4exp import Qwen4ExpTextModel, Qwen4ExpVisionModel
from pack_modelopt_mtp import vocabulary
from strata_tokenizer import Tokenizer, UNICODE_TO_BYTE, extract


def tokenizer(pack):
    header = (pack / "model.stidx").read_bytes()
    if header[:8] != b"STRI" + struct.pack("<I", 1):
        raise ValueError("Expected a raw model.stidx")
    temporary = pack / "tokenizer-metadata.header"
    with temporary.open("xb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + header[8:])
    try:
        extract(temporary, pack)
        tok = Tokenizer.from_gguf(temporary)
        for text in ("Hello, world!", "中文和 English", "1234567890", "a\n\nb", "😀"):
            if tok.decode(tok.encode(text)) != text:
                raise ValueError("Tokenizer round-trip failed")
    finally:
        temporary.unlink()


def draft_vocab(model, out):
    forbidden = regex.compile(r"[\p{Cyrillic}\p{Hiragana}\p{Katakana}\p{Hangul}]")
    selected = []
    for i, encoded in sorted(vocabulary(model / "tokenizer.json").items()):
        try:
            text = bytes(UNICODE_TO_BYTE[c] for c in encoded).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            # Byte fragments are needed to compose words; special tokens retain their IDs.
            text = encoded if encoded.startswith("<") else ""
        if not forbidden.search(text) and not encoded.startswith("[PAD"):
            selected.append(i)
    np.asarray(selected, dtype="<u4").tofile(out)
    print("Draft vocabulary", len(selected), flush=True)


def vision(model_dir, pack, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mmproj-bf16.gguf"
    if path.exists():
        raise FileExistsError(path)
    source = Source(model_dir)
    model = Qwen4ExpVisionModel(model_dir, gguf.LlamaFileType.MOSTLY_BF16, path,
                               hparams=json.loads((model_dir / "config.json").read_text()))
    writer = model.gguf_writer
    count = 0
    for name in sorted(n for n in source.headers if "visual." in n):
        filtered = model.filter_tensors((name, lambda n=name: source.get(n)))
        if filtered is None:
            raise ValueError("Vision tensor was rejected: " + name)
        normalized = filtered[0]
        bid = re.search(r"\.blocks\.(\d+)\.", normalized)
        block = int(bid[1]) if bid else None
        for target, value in model.modify_tensors(source.values(name), normalized, block):
            kind = model.tensor_force_quant(normalized, target, block, value.ndim)
            if isinstance(kind, bool):
                full = value.ndim == 1 or not target.endswith(".weight") or model.match_model_tensor_name(
                    target, gguf.MODEL_TENSOR.V_ENC_EMBD_POS, block)
                kind = gguf.GGMLQuantizationType.F32 if full else gguf.GGMLQuantizationType.BF16
            writer.add_tensor(target, gguf.quants.quantize(value.contiguous().numpy(), kind), raw_dtype=kind)
            count += 1
    if count != 334:
        raise ValueError(f"Expected 334 image tensors, got {count}")
    model.prepare_metadata(vocab_only=False)
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    # A vocab-only GGUF has no tensor records or language-model payload.
    text = Qwen4ExpTextModel(model_dir, gguf.LlamaFileType.MOSTLY_Q8_0, out / "tokenizer-only.gguf",
                            hparams=json.loads((model_dir / "config.json").read_text()))
    text._ple_row_dim = 160
    text.hparams["layer_types"] = ["full_attention" if x == "qwen_sparse_attention" else x
                                  for x in text.hparams["layer_types"]]
    text.write_vocab()
    (out / "prepare-report.json").write_text(json.dumps(dict(passed=True, tensors=count,
        source_index_sha256=source.fingerprint, language_model_weights=False)) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--vision-out", type=Path)
    ap.add_argument("--draft-out", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    tokenizer(a.pack)
    if a.draft_out:
        draft_vocab(a.model, a.draft_out)
    if a.vision_out:
        vision(a.model, a.pack, a.vision_out)
