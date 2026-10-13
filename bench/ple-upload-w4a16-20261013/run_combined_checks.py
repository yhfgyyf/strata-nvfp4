"""Numerical, API and two-slot capacity checks for the three PLE controls together."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path('/home/tc/Strata-nvfp4')
ROOT = SOURCE / 'bench/ple-upload-w4a16-20261013'
PREVIOUS = SOURCE / 'bench/prefill-residency-p0-p3-20261012/nvidia-next-20261012'
sys.path.insert(0, str(PREVIOUS))
import run_cases as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--run', default='combined-checks-1')
    args = parser.parse_args()
    assert args.build.resolve().is_relative_to(ROOT) and (args.build / 'build.json').is_file()
    assert Path(args.run).name == args.run
    runner.ROOT = ROOT
    runner.EXE = SOURCE / 'engine/strata-nvfp4-w4a16-r256'
    runner.PORT = 30113
    os.umask(0o077)
    out = ROOT / args.run
    out.mkdir(exist_ok=False)
    paths = (runner.CONFIG, runner.SCRIPT, runner.EXE, runner.PROFILE, *runner.METADATA)
    hashes = {str(p): runner.sha(p) for p in paths}
    base = runner.CONFIG.read_bytes()

    def guard(check_stop=True):
        for path, digest in hashes.items(): assert runner.sha(path) == digest, 'Protected file changed: ' + path
        assert not runner.miners(), 'External miner appeared'
        if check_stop: assert not (ROOT / 'stop-request').exists(), 'Operator stopped tests'

    primary = None
    lifecycle = False
    try:
        deadline = time.monotonic() + 300
        while runner.api('v1/status', port=30001)['activity']['in_flight']:
            assert time.monotonic() < deadline, 'Production remains busy'
            time.sleep(2)
        guard()
        cfg = json.loads(base)
        cfg['_file'] = str(runner.CONFIG)
        owner = runner.server_owner(cfg)
        assert all(os.getpgid(pid) == int(owner.name) for pid in runner.gpu_pids())
        lifecycle = True
        with (out / 'lifecycle.private.log').open('ab') as log:
            subprocess.run(['bash', str(runner.SCRIPT), 'stop_strata'], stdout=log, stderr=log, check=True)
        assert not runner.gpu_pids()
        cfg = json.loads(base)
        cfg['env'].update(STRATA_PLE_EARLY_UPLOAD='1', STRATA_PLE_RAW_UPLOAD='1', STRATA_PLE_RAW_CACHE_MIB='64')
        import capacity_checks
        original_numerical = capacity_checks.numerical
        def numerical(config, folder, label, check_guard):
            original_numerical(config, folder, label, check_guard)
            comparisons = []
            for size in (8192, 32768, 131072):
                candidate = folder / (label + '-numerical-%d.private.bin' % size)
                baseline = ROOT / 'step1-v3-aba-1' / ('W4A16-ready-A1-numerical-%d.private.bin' % size)
                equal = candidate.read_bytes() == baseline.read_bytes()
                comparisons.append(dict(input_tokens=size, baseline_sha256=runner.sha(baseline),
                                        candidate_sha256=runner.sha(candidate), bit_equal=equal))
            runner.save(folder / 'numerical-comparison.public.json', dict(rows=comparisons))
            assert all(row['bit_equal'] for row in comparisons), 'Combined first-token logits differ'
        capacity_checks.numerical = numerical
        spec = dict(label='W4A16-combined', mode='fp16', reserve=1536, compact=0, ring=256,
                    build=str(args.build), numerical=True, capacity_checks=True, ledger=True)
        runner.phase(spec, json.dumps(cfg).encode(), out, guard, 1, [8192])
        runner.save(out / 'completed.public.json', dict(complete=True,
            scope='Combined numerical/feature/capacity and ledger checks; not a throughput campaign'))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if lifecycle:
            try:
                guard(check_stop=False)
                cfg = json.loads(base)
                cfg['_file'] = str(runner.CONFIG)
                owner = runner.server_owner(cfg, required=False)
                if not owner:
                    assert not runner.gpu_pids(), 'GPU process remains; refuse restart'
                    with (out / 'lifecycle.private.log').open('ab') as log:
                        subprocess.run(['bash', str(runner.SCRIPT), 'start_strata'], stdout=log, stderr=log, check=True)
                runner.loaded(port=30001)
                assert runner.sha(runner.native(runner.EXE) / 'exe') == hashes[str(runner.EXE)]
                guard(check_stop=False)
                runner.save(out / 'restored.public.json', dict(loaded=True, files=hashes))
            except BaseException as exc:
                if primary is not None and hasattr(primary, 'add_note'): primary.add_note('Restore failure: ' + repr(exc))
                try: print('RESTORE_ERROR', repr(exc), file=sys.stderr, flush=True)
                except BaseException: pass
                if primary is None: raise


if __name__ == '__main__': main()
