"""One 8K/512 Nsight request per early-upload arm; isolated configs, guarded production recovery."""
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
import profile_ring_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, default=ROOT / 'step1-r1/build')
    parser.add_argument('--run', default='step1-trace-2')
    parser.add_argument('--label', default='W4A16-early-trace')
    parser.add_argument('--cuda-fixture', type=Path)
    args = parser.parse_args()
    assert args.build.resolve().is_relative_to(ROOT) and (args.build / 'build.json').is_file()
    assert Path(args.run).name == args.run and Path(args.label).name == args.label
    runner.ROOT = ROOT
    runner.EXE = SOURCE / 'engine/strata-nvfp4-w4a16-r256'
    runner.PORT = 30113
    os.umask(0o077)
    out = ROOT / args.run
    out.mkdir(exist_ok=False)
    paths = (runner.CONFIG, runner.SCRIPT, runner.EXE, runner.PROFILE, *runner.METADATA)
    hashes = {str(p): runner.sha(p) for p in paths}
    base = runner.CONFIG.read_bytes()
    def guard():
        for path, digest in hashes.items(): assert runner.sha(path) == digest, 'Protected file changed: ' + path
        assert not runner.miners(), 'External miner appeared'
    primary = None
    lifecycle = False
    try:
        # Ownership and config rechecked after the potentially long idle wait.
        deadline = time.monotonic() + 300
        while runner.api('v1/status', port=30001)['activity']['in_flight']:
            assert time.monotonic() < deadline, 'Production remains busy; do not interrupt it'
            time.sleep(2)
        guard()
        production = json.loads(base)
        production['_file'] = str(runner.CONFIG)
        owner = runner.server_owner(production)
        assert all(os.getpgid(pid) == int(owner.name) for pid in runner.gpu_pids())
        lifecycle = True  # enter recovery before the first stop
        with (out / 'lifecycle.private.log').open('ab') as log:
            subprocess.run(['bash', str(runner.SCRIPT), 'stop_strata'], stdout=log, stderr=log, check=True)
        assert not runner.gpu_pids()
        if args.cuda_fixture:
            assert args.cuda_fixture.resolve().is_relative_to(ROOT) and args.cuda_fixture.is_file()
            with (out / 'cuda-fixture.private.log').open('xb') as log:
                result = subprocess.run([str(args.cuda_fixture)], stdout=subprocess.PIPE, stderr=log, check=True, text=True)
            fixture = json.loads(result.stdout)
            assert fixture['passed']
            fixture['binary_sha256'] = runner.sha(args.cuda_fixture)
            runner.save(out / 'cuda-fixture.public.json', fixture)
        for arm, early in [('A', '0'), ('B', '1')]:
            cfg = json.loads(base)
            cfg['env']['STRATA_PLE_EARLY_UPLOAD'] = early
            cfg['env']['STRATA_PLE_RAW_UPLOAD'] = '0'
            cfg['env']['STRATA_PLE_RAW_CACHE_MIB'] = '0'
            spec = dict(label=args.label + '-' + arm, mode='fp16', reserve=1536, compact=0, ring=256,
                        build=str(args.build))
            profile_ring_graph.phase(spec, json.dumps(cfg).encode(), out, guard, 1, [8192])
        runner.save(out / 'completed.public.json', dict(complete=True, instrumented=True,
                    scope='One cold8K/512 request per arm; not unprofiled throughput'))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if lifecycle:
            try:
                guard()
                assert not runner.gpu_pids(), 'Owned/unowned GPU process remains; refuse restart'
                with (out / 'lifecycle.private.log').open('ab') as log:
                    subprocess.run(['bash', str(runner.SCRIPT), 'start_strata'], stdout=log, stderr=log, check=True)
                runner.loaded(port=30001)
                assert runner.sha(runner.native(runner.EXE) / 'exe') == hashes[str(runner.EXE)]
                guard()
                runner.save(out / 'restored.public.json', dict(loaded=True, files=hashes))
            except BaseException as exc:
                if primary is not None and hasattr(primary, 'add_note'): primary.add_note('Restore failure: ' + repr(exc))
                try: print('RESTORE_ERROR', repr(exc), file=sys.stderr, flush=True)
                except BaseException: pass
                if primary is None: raise


if __name__ == '__main__': main()
