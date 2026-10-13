"""W4A16-only isolated trials using the established server measurement and teardown helpers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path('/home/tc/Strata-nvfp4')
HERE = SOURCE / 'bench/ple-upload-w4a16-20261013'
PREVIOUS = SOURCE / 'bench/prefill-residency-p0-p3-20261012/nvidia-next-20261012'
sys.path.insert(0, str(PREVIOUS))
import run_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    parser.add_argument('--baseline-build', help='Retained baseline build for a production-versus-candidate campaign')
    parser.add_argument('--run', required=True)
    parser.add_argument('--candidate-early', type=int, choices=(0, 1), default=1)
    parser.add_argument('--baseline-early', type=int, choices=(0, 1), default=0)
    parser.add_argument('--candidate-raw', type=int, choices=(0, 1), default=0)
    parser.add_argument('--baseline-raw', type=int, choices=(0, 1), default=0)
    parser.add_argument('--candidate-cache-mib', type=int, default=0)
    parser.add_argument('--baseline-cache-mib', type=int, default=0)
    parser.add_argument('--label', default='W4A16-early')
    parser.add_argument('--capacity-checks', action='store_true')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--arms', default='A1,B,A2')
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    assert args.cases > 0
    assert Path(args.label).name == args.label
    assert 0 <= args.baseline_cache_mib <= 512 and 0 <= args.candidate_cache_mib <= 512
    arms = args.arms.split(',')
    assert arms and all(arm in ('A1', 'B', 'A2') for arm in arms)
    build = Path(args.build).resolve()
    assert build.is_relative_to(HERE) and (build / 'build.json').is_file()
    baseline_build = Path(args.baseline_build).resolve() if args.baseline_build else build
    assert baseline_build.is_relative_to(SOURCE / 'bench') and (baseline_build / 'build.json').is_file()
    run_cases.ROOT = HERE
    run_cases.EXE = SOURCE / 'engine/strata-nvfp4-w4a16-r256'
    run_cases.PORT = 30113
    os.umask(0o077)
    out = HERE / args.run
    out.mkdir(exist_ok=False)
    paths = (run_cases.CONFIG, run_cases.SCRIPT, run_cases.EXE, run_cases.PROFILE, *run_cases.METADATA)
    protected = {str(p): run_cases.sha(p) for p in paths}
    assert protected[str(run_cases.CONFIG)] == 'f7b15c8a26987323404e2cada27db61e694f345bc3f0aa7632ed427be9c65115'
    assert not run_cases.gpu_pids(), 'Stop the owned production service before an isolated campaign'
    base = run_cases.CONFIG.read_bytes()
    for index, path in enumerate(paths):
        dest = out / ('backup-%d-' % index + path.name)
        shutil.copy2(path, dest)
        assert run_cases.sha(dest) == protected[str(path)]
    helpers = {str(p): run_cases.sha(p) for p in (PREVIOUS / 'run_cases.py', PREVIOUS / 'capacity_checks.py',
               Path(__file__), run_cases.INPUTS / 'inputs.public.json')}
    run_cases.save(out / 'protected.public.json', dict(files=protected, helpers=helpers))

    def guard(check_stop=True):
        for path, digest in protected.items():
            assert run_cases.sha(path) == digest, 'Protected target changed: ' + path
        assert not run_cases.miners(), 'External miner appeared; preserve it and stop the campaign'
        if check_stop: assert not (HERE / 'stop-request').exists(), 'Operator stopped tests'

    original_measure = run_cases.measure
    def measure(*items):
        row = original_measure(*items)
        process = run_cases.native(items[3]['exe'])
        status = (process / 'status').read_text().splitlines()
        row['host_memory_kib'] = {line.split(':')[0]: int(line.split()[1]) for line in status
                                if line.startswith(('VmRSS:', 'VmHWM:'))}
        mem = Path('/proc/meminfo').read_text().splitlines()
        row['host_memory_kib']['MemAvailable'] = int(next(x for x in mem if x.startswith('MemAvailable:')).split()[1])
        if '-cold-' in row['label']:
            assert row['reused'] == 0 and row['full_output'], 'Incomplete or prefix-reused speed sample retained'
        run_cases.save(items[4] / (row['label'] + '.public.json'), row)
        return row
    run_cases.measure = measure
    import capacity_checks
    original_numerical = capacity_checks.numerical
    def numerical(cfg, outdir, label, guard):
        original_numerical(cfg, outdir, label, guard)
        if label.endswith(('-B', '-A2')):
            rows = []
            for path in sorted(outdir.glob(label + '-numerical-*.private.bin')):
                baseline = outdir / path.name.replace(label, args.label + '-A1', 1)
                equal = path.read_bytes() == baseline.read_bytes()
                rows.append(dict(file=path.name, baseline_sha256=run_cases.sha(baseline),
                                 candidate_sha256=run_cases.sha(path), bit_equal=equal))
            run_cases.save(outdir / (label + '-numerical-comparison.public.json'), dict(rows=rows))
            assert rows and all(row['bit_equal'] for row in rows), 'First-token logits changed; do not begin speed trials'
    capacity_checks.numerical = numerical
    primary = None
    try:
        for arm in arms:
            guard()
            cfg = json.loads(base)
            cfg['env']['STRATA_PLE_EARLY_UPLOAD'] = str(args.candidate_early if arm == 'B' else args.baseline_early)
            cfg['env']['STRATA_PLE_RAW_UPLOAD'] = str(args.candidate_raw if arm == 'B' else args.baseline_raw)
            cfg['env']['STRATA_PLE_RAW_CACHE_MIB'] = str(args.candidate_cache_mib if arm == 'B' else args.baseline_cache_mib)
            spec = dict(label=args.label + '-' + arm, mode='fp16', reserve=1536, compact=0, ring=256,
                        build=str(build if arm == 'B' else baseline_build), numerical=True,
                        capacity_checks=args.capacity_checks and arm == 'B')
            run_cases.phase(spec, json.dumps(cfg).encode(), out, guard, args.cases, [8192, 32768, 131072])
        guard()
        run_cases.save(out / 'completed.public.json', dict(complete=True, arms=arms, cases=args.cases,
                       scope='W4A16 only; no profiler; INT8 KV retained; arithmetic mean of3 matched trials'))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            guard(check_stop=False)
            assert not run_cases.gpu_pids(), 'Refuse production restart over a remaining/unowned GPU process'
            with (out / 'restore.private.log').open('ab') as log:
                subprocess.run(['bash', str(run_cases.SCRIPT), 'start_strata'], stdout=log, stderr=log, check=True)
            run_cases.loaded(port=30001)
            process = run_cases.native(run_cases.EXE)
            assert run_cases.sha(process / 'exe') == protected[str(run_cases.EXE)]
            guard(check_stop=False)
            run_cases.save(out / 'restored.public.json', dict(loaded=True, native_pid=int(process.name), files=protected))
            print('PRODUCTION_RESTORED', flush=True)
        except BaseException as exc:
            if primary is not None and hasattr(primary, 'add_note'):
                primary.add_note('Production restoration failure: ' + repr(exc))
            try: print('RESTORE_ERROR', repr(exc), file=sys.stderr, flush=True)
            except BaseException: pass
            if primary is None: raise


if __name__ == '__main__':
    main()
