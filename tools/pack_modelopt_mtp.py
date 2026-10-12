"""Build the existing Q2_0/Q8_0 MTP runtime from NVIDIA's FP8/BF16 MTP.

The main experts and PLE are untouched. FP8 MTP routed weights are decoded on
the CPU using their 128x128 multipliers, then use Strata's existing quantizers.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

from modelopt_source import Source, np, torch, STRATA
sys.path.insert(0, str(STRATA / 'tools'))
from mtp_pack import q2_0, dequant
from mtp_rt import BLOB, FF, H, NE, Q8, blob_of, q8_0

SOURCE = None


def initialize(model):
    global SOURCE
    torch.set_num_threads(1)
    SOURCE = Source(model)


def expert(e):
    stem = f'mtp.layers.0.mlp.experts.{e}.'
    gate = SOURCE.values(stem + 'gate_proj.weight').numpy()
    up = SOURCE.values(stem + 'up_proj.weight').numpy()
    dn = SOURCE.values(stem + 'down_proj.weight').numpy()
    gu = np.concatenate((gate, up), axis=0)
    assert gu.shape == (2 * FF, H) and dn.shape == (H, FF)
    guq = q2_0(gu).reshape(2 * FF, -1)
    dnq = q2_0(dn).reshape(H, -1)
    errors = {}
    if e in (0, 1, 2, 7, 127, 255, 383, 511):
        for name, raw, packed in (('gate_up', gu, guq), ('down', dn, dnq)):
            restored = dequant('q2_0', packed.reshape(-1), raw.size).reshape(raw.shape)
            assert np.isfinite(restored).all()
            errors[name] = float(np.sqrt(np.mean((restored - raw) ** 2)) /
                                 (np.sqrt(np.mean(raw ** 2)) + 1e-12))
    return e, blob_of(guq, dnq), errors


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def vocabulary(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data.get('model'), str):
        # Strata's exported tokenizer keeps its vocabulary in a separate file.
        return {int(i): s for s, i in json.loads((Path(path).parent / 'vocab.json').read_text()).items()}
    result = {int(i): s for s, i in data['model']['vocab'].items()}
    result.update({int(t['id']): t['content'] for t in data.get('added_tokens', [])})
    return result


def copy_draft_vocab(model, out, draft):
    ids = np.fromfile(draft, dtype='<u4')
    tokens = vocabulary(Path(model) / 'tokenizer.json')
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Draft vocabulary is empty or contains duplicate IDs')
    if any(int(i) not in tokens for i in ids):
        raise ValueError('Draft vocabulary contains IDs absent from this checkpoint')
    ids.tofile(out / 'draft_vocab.bin')
    return dict(tokens=len(ids), source_sha256=sha(draft), sha256=sha(out / 'draft_vocab.bin'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--draft-vocab', type=Path, default=STRATA / 'data/draft_vocab.bin')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(1)
    source = Source(args.model)
    report = dict(source_index_sha256=source.fingerprint, expert_source='FP8 with 128x128 weight_scale_inv multipliers',
                  expert_runtime='existing Strata Q2_0 per-block MSE quantizer',
                  dense_policy='Q8_0 projections; BF16 small weights; F32 Gemma norms',
                  calibration_dataset_used=False, gguf_weights_created=False, expert_sample_errors=[], dense_tensors=[])
    part = args.out / 'experts.bin.part'
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.model,)) as pool, part.open('xb') as f:
        for e, blob, errors in pool.map(expert, range(NE), chunksize=1):
            assert f.tell() == e * BLOB and len(blob) == BLOB
            f.write(blob)
            if errors:
                report['expert_sample_errors'].append(dict(expert=e, **errors))
            if (e + 1) % 32 == 0:
                print('MTP_EXPERT_PROGRESS', e + 1, round(time.monotonic() - started, 1), flush=True)
    assert part.stat().st_size == NE * BLOB
    part.replace(args.out / 'experts.bin')
    names = sorted(n for n in source.headers if n.startswith('mtp.') and '.experts.' not in n)
    assert len(names) == 29, (len(names), names)
    lines = []
    part = args.out / 'dense.bin.part'
    with part.open('xb') as f:
        for name in names:
            assert source.headers[name]['dtype'] == 'BF16', name
            short = name.removeprefix('mtp.').removeprefix('layers.0.')
            raw = source.get(name)
            if raw.ndim == 1:
                assert 'norm' in short, 'Non-norm 1D MTP tensor: ' + name
                data = (raw.float().numpy() + np.float32(1)).astype('<f4').tobytes()
                kind, rows, cols = 'f32', 1, raw.numel()
            else:
                rows, cols = raw.shape
                if short in Q8:
                    data, kind = q8_0(raw.float().numpy()), 'q8_0'
                else:
                    data, kind = raw.view(torch.uint16).numpy().tobytes(), 'bf16'
            f.write(b'\0' * (-f.tell() % 256))
            offset = f.tell()
            f.write(data)
            lines.append(f'{short} {kind} {rows} {cols} {offset} {len(data)}')
            report['dense_tensors'].append(dict(name=short, kind=kind, rows=rows, cols=cols, bytes=len(data)))
    part.replace(args.out / 'dense.bin')
    (args.out / 'dense.txt').write_text('\n'.join(lines) + '\n')
    report['draft_vocab'] = copy_draft_vocab(args.model, args.out, args.draft_vocab)
    report['runtime_files'] = {name: dict(bytes=(args.out / name).stat().st_size, sha256=sha(args.out / name))
                               for name in ('experts.bin', 'dense.bin', 'dense.txt', 'draft_vocab.bin')}
    report.update(passed=True, seconds=time.monotonic() - started)
    (args.out / 'prepare-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('MTP_READY', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
