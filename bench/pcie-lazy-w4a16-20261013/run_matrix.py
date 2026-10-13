"""Bounded W4A16 PCIe/adaptation trials using the retained request harness."""
import argparse
from array import array
import copy
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

SOURCE = Path('/home/tc/Strata-nvfp4')
ROOT = SOURCE / 'bench/pcie-lazy-w4a16-20261013'
CONFIG = SOURCE / 'strata-qwen38-flash-next-nvidia-30001.json'
SCRIPT = Path('/home/tc/wildrig/gpu_schedule.sh')
PORT = 30114
SIZES = (8192, 32768, 131072)


def harness():
    sys.path.insert(0, str(SOURCE/'bench/prefill-residency-p0-p3-20261012/nvidia-next-20261012'))
    runner = importlib.import_module('run_cases')
    runner.PORT = PORT
    runner.CONFIG = CONFIG
    return runner


def option(args, name, value):
    if name in args:
        args[args.index(name)+1] = str(value)
    else:
        args.extend((name, str(value)))


def candidate_config(base, spec, out, label):
    cfg = copy.deepcopy(base)
    cfg.update(host='127.0.0.1', port=PORT, open_browser=False,
               log=str(out/(label+'-engine.private.log')))
    option(cfg['args'], '--pcie-mode', spec['mode'])
    option(cfg['args'], '--pcie-frac', spec['frac'])
    if spec.get('parallel') == 1:
        cfg['parallel'] = 1
        option(cfg['args'], '--batch', 1)  # Native parser explicitly normalizes this to zero.
    if spec.get('async'):
        option(cfg['args'], '--adapt-async', 1)
    if 'rotate' in spec:
        cfg['env']['STRATA_EXCHANGE_ROTATE']='1' if spec['rotate'] else '0'
    cfg['env'].update(STRATA_DECODE_TIMING='1', STRATA_PREFILL_NVFP4='fp16',
                      STRATA_NVFP4_W4A16='1', STRATA_NVFP4_FUSED_MOE='2',
                      STRATA_PREFILL_RING='256', STRATA_PREFILL_NVFP4_FP8='0')
    for key in ('STRATA_ADAPT_WAIT', 'STRATA_ADAPT_NOWAIT', 'STRATA_ADAPT_LAG',
                'STRATA_VERIFY_PROFILE', 'STRATA_PREFILL_LEDGER', 'STRATA_SLOT_LEDGER'):
        cfg['env'].pop(key, None)
    if spec.get('lag'):
        cfg['env'].update(STRATA_ADAPT_WAIT='1', STRATA_ADAPT_LAG=str(spec['lag']))
    cfg['env']['STRATA_DUMP_FIRST_LOGITS'] = str(out/(label+'-logits.private.bin'))
    return cfg


def means(rows):
    result = []
    for size in SIZES:
        selected = [r for r in rows if r['input_tokens'] == size]
        if len(selected) != 3:
            raise ValueError('Expected three complete cases per size')
        keys = ('prefill_tps', 'decode_native_tps', 'ttft_s', 'prompt_ms')
        result.append(dict(input_tokens=size, trials=3,
            mean={k:statistics.mean(r[k] for r in selected) for k in keys},
            sample_stdev={k:statistics.stdev(r[k] for r in selected) for k in keys},
            individual=[{k:r[k] for k in ('label', *keys, 'output_tokens', 'reused',
                'quality', 'output_sha256', 'payload_sha256', 'native')} for r in selected]))
    return result


def compare_logits(base, other):
    a, b = array('f'), array('f')
    a.frombytes(Path(base).read_bytes()); b.frombytes(Path(other).read_bytes())
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError('Invalid logits')
    aa = math.fsum(x*x for x in a); bb = math.fsum(x*x for x in b)
    sq = math.fsum((x-y)**2 for x,y in zip(a,b))
    return dict(values=len(a), finite=True,
        top1_equal=max(range(len(a)), key=a.__getitem__)==max(range(len(b)), key=b.__getitem__),
        maximum_absolute_error=max(abs(x-y) for x,y in zip(a,b)),
        rmse=math.sqrt(sq/len(a)), relative_rmse=math.sqrt(sq/aa) if aa else None,
        cosine=math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa*bb else None)


def control(action, out):
    with (out/'controller.private.log').open('ab') as log:
        subprocess.run(['bash', str(SCRIPT), action], stdout=log, stderr=log,
                       check=True, timeout=90)


def stop_owned(process, cfg, runner):
    """Teardown is bound to the process group we created, not to API reachability."""
    if process.poll() is None:
        proc = Path('/proc')/str(process.pid)
        args = (proc/'cmdline').read_bytes().split(b'\0')
        if (os.getpgid(process.pid) != process.pid or b'--config' not in args or
                args[args.index(b'--config')+1] != os.fsencode(cfg['_file'])):
            raise RuntimeError('Candidate process ownership changed')
    else:
        # The Python parent can exit while native children remain in its owned group.
        for pid in runner.gpu_pids():
            if os.getpgid(pid) != process.pid:
                raise RuntimeError('Unowned GPU process appeared during teardown')
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic()+20
    while time.monotonic() < deadline:
        owned = [pid for pid in runner.gpu_pids() if os.getpgid(pid) == process.pid]
        if not owned and process.poll() is not None:
            return
        time.sleep(.5)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)
    if any(os.getpgid(pid) == process.pid for pid in runner.gpu_pids()):
        raise RuntimeError('Owned GPU process did not exit')


def worker(config, output, name, label):
    runner = harness()
    cfg = json.loads(config.read_text()); cfg['_file'] = str(config)
    runner.EXE = Path(cfg['exe'])
    entries = json.loads((runner.INPUTS/'inputs.public.json').read_text())['requests']
    entry = next(e for e in entries if e['file'] == name)
    path = runner.INPUTS/'inputs'/name
    assert runner.sha(path) == entry['sha256']
    row = runner.measure(label, path, entry['expected'], cfg, output)
    segment = (output/(label+'-0.engine.log')).read_text()
    lines = [line for line in segment.splitlines() if line.startswith('strata decode timing:')]
    assert len(lines) == 1, 'Expected one GEN decode summary'
    groups = re.search(r'CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)',lines[0])
    assert groups, 'Decode plan counts missing'
    row.update(actual_decode_path='GEN', decode_summary=lines[0],
        per_layer_window=dict(zip(('cpu_experts','cpu_entries','vram_hits','pcie_experts'),map(float,groups.groups()))))
    runner.save(output/(label+'.public.json'),row)
    assert row['full_output'] and row['reused'] == 0
    assert not row['quality']['periodic_tail'] and not row['quality']['long_digit_loop']


def numerical_worker(config, output, label):
    harness()
    cfg=json.loads(config.read_text());cfg['_file']=str(config)
    importlib.import_module('capacity_checks').numerical(cfg, output, label, lambda:None)


def phase(spec, base, out, label, guard, baseline_logits=None, smoke_only=False):
    runner = harness()
    assert not runner.gpu_pids(), 'GPU occupied before trial'
    cfg = candidate_config(base, spec, out, label)
    path = out/(label+'-config.private.json'); runner.save(path, cfg)
    cfg['_file'] = str(path); digest = runner.sha(path)
    process = None
    memory = runner.MemorySampler(); memory.thread.start()
    primary = None
    try:
        with (out/(label+'-server.private.log')).open('xb') as log:
            process = subprocess.Popen([str(SOURCE/'.venv/bin/python'), '-m', 'serve.server',
                '--engine', 'strata', '--config', str(path), '--port', str(PORT)],
                cwd=SOURCE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True)
        runner.save(out/'progress.public.json', dict(stage='loading', label=label,
                    trial_api_pid=process.pid, orchestrator_pid=os.getpid()))
        runner.loaded(process, port=PORT)
        guard()
        native = runner.native(cfg['exe'])
        assert runner.sha(native/'exe') == runner.sha(cfg['exe'])
        engine = runner.api('metrics', port=PORT)['engine']
        assert engine['context'] == 262144 and engine['kv'] == 'int8'
        if spec.get('parallel') == 1:
            # INFO emits batch_slots only for o.batch>0; the frontend treats its
            # absence as zero. Also check the actual API concurrency cap.
            assert int(engine.get('batch_slots') or 0) == 0, 'Expected effective native batch=0'
            assert runner.api('v1/status',port=PORT)['concurrency']['serving'] == 1
        startup_log=Path(cfg['log']).read_text()
        if spec.get('async'):
            assert 'adaptive tier asynchronous (--adapt-async 1)' in startup_log, 'Async adaptation fell back'
        if spec.get('rotate'):
            assert 'exchange buffer rotation enabled:' in startup_log, 'RAM rotation fell back'
        env = dict(e.split('=',1) for e in (native/'environ').read_bytes().decode().split('\0') if '=' in e)
        assert 'NSYS_PROFILING_SESSION_ID' not in env and 'nsys' not in env.get('LD_PRELOAD','').lower()
        runner.save(out/(label+'-startup.public.json'), dict(spec=spec, engine=engine,
            effective_native_batch_slots=int(engine.get('batch_slots') or 0),
            native_pid=int(native.name), api_pid=process.pid, binary_sha256=runner.sha(native/'exe'),
            config_sha256=digest, env={k:v for k,v in env.items() if k.startswith('STRATA_')},
            native_args=(native/'cmdline').read_bytes().decode().split('\0')))
        print('READY', label, engine['expert_slots'], flush=True)

        # The first forced 8K/one-token request uses fresh profile state. Its worker
        # is bounded externally, including an unavailable or deadlocked candidate API.
        if not smoke_only:
            with (out/'worker.private.log').open('ab') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--numerical',str(path),
                    '--output',str(out),'--label',label],stdout=log,stderr=log,check=True,timeout=240)

        # A short multi-window API request has an external timeout.
        payload = dict(model=base['model_name'], messages=[dict(role='user',content=
            'Explain why a CPU and a GPU can run independent jobs at the same time. Write at least 150 words.')],
            temperature=0, max_tokens=64, stream=False, reasoning_effort='none')
        script = ('import json,urllib.request; p=json.loads(input()); '
            'r=urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions",'%PORT+
            'data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}); '
            'print(urllib.request.urlopen(r,timeout=80).read().decode())')
        smoke = subprocess.run([sys.executable, '-c', script], input=json.dumps(payload),
                               text=True, capture_output=True, timeout=90, check=True)
        response = json.loads(smoke.stdout)
        runner.save(out/(label+'-smoke.private.json'), response)
        runner.save(out/(label+'-smoke.public.json'), dict(complete=True, usage=response.get('usage'),
            metrics=runner.api('metrics',port=PORT)['requests'][0], timeout_s=90))
        if smoke_only:
            return

        comparisons = {}
        for size in SIZES:
            receipt=json.loads((out/(label+'-numerical-%d.public.json'%size)).read_text())
            if spec.get('lag'):
                assert receipt['diagnostic']['adapt_nowait']==0 and receipt['diagnostic']['adapt_lag']==spec['lag']
            values = out/(label+'-numerical-%d.private.bin'%size)
            if baseline_logits:
                comparisons[str(size)] = compare_logits(out/(baseline_logits+'-numerical-%d.private.bin'%size), values)
        if comparisons:
            runner.save(out/(label+'-logits-comparison.public.json'), comparisons)
        Path(cfg['env']['STRATA_DUMP_FIRST_LOGITS']).unlink(missing_ok=True)
        names = ['warmup-%d.json'%i for i in range(3)]
        for case in range(3):
            order = (SIZES, tuple(reversed(SIZES)), SIZES[1:]+SIZES[:1])[case]
            names += ['cold-%d-%d.json'%(size,case) for size in order]
        rows = []
        for name in names:
            guard(); assert runner.sha(path) == digest, 'Candidate config changed'
            case_label = label+'-'+Path(name).stem
            with (out/'worker.private.log').open('ab') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--measure', str(path),
                    '--output', str(out), '--name', name, '--label', case_label],
                    stdout=log, stderr=log, check=True, timeout=600)
            row = json.loads((out/(case_label+'.public.json')).read_text())
            if name.startswith('cold'):
                rows.append(row)
            print('MEASURED',case_label,round(row['prefill_tps'],1),row['decode_native_tps'],flush=True)
        runner.save(out/(label+'-means.public.json'), dict(complete=True, instrumented=False,
            native_host_timing_summary=True, spec=spec, rows=means(rows)))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors = []
        try:
            memory.finish(out/(label+'-memory.public.json'))
        except BaseException as exc:
            errors.append(repr(exc))
        if process is not None:
            try:
                stop_owned(process, cfg, runner)
            except BaseException as exc:
                errors.append(repr(exc))
        if errors:
            if primary is not None:
                if hasattr(primary,'add_note'): primary.add_note('Cleanup: '+repr(errors))
            else:
                raise RuntimeError(repr(errors))
        print('STOPPED',label,flush=True)


def main(args):
    runner = harness()
    os.umask(0o077)
    out = ROOT/args.run; out.mkdir(exist_ok=False)
    base_bytes = CONFIG.read_bytes(); base = json.loads(base_bytes)
    cfg = {**base, '_file':str(CONFIG)}
    paths = (CONFIG, SCRIPT, Path(base['exe']), runner.PROFILE, *runner.METADATA)
    hashes = {str(p):runner.sha(p) for p in paths}
    def guard():
        assert all(runner.sha(p)==digest for p,digest in ((Path(p),h) for p,h in hashes.items())), 'Protected target changed'
        assert not runner.miners(), 'External miner appeared'
        assert not (ROOT/'stop-request').exists(), 'Operator stopped trials'
    runner.save(out/'before.public.json', dict(time=time.time(), files=hashes,
        source_hashes={str(p):runner.sha(SOURCE/p) for p in (Path('src/program/generate.cpp'),
            Path('src/core/verify.cpp'),Path('src/core/expert_source.cpp'),Path('serve/server.py'))},
        helper_sha256=runner.sha(Path(__file__)), input_sha256=runner.sha(runner.INPUTS/'inputs.public.json')))
    for p in (CONFIG,SCRIPT):
        backup=out/('backup-'+p.name); backup.write_bytes(p.read_bytes())
        assert runner.sha(backup)==hashes[str(p)]
    plan=json.loads(args.plan.read_text()); runner.save(out/'plan.public.json',plan)
    lifecycle=False; primary=None
    try:
        guard()
        deadline=time.monotonic()+300
        while runner.api('v1/status',port=30001)['activity']['in_flight']:
            if time.monotonic()>deadline: raise RuntimeError('Production stayed busy')
            time.sleep(2)
        guard()
        owner=runner.server_owner(cfg)
        assert all(os.getpgid(pid)==int(owner.name) for pid in runner.gpu_pids()), 'Unowned GPU process'
        lifecycle=True
        control('stop_strata',out)
        guard(); assert not runner.gpu_pids()
        baseline=None
        for i,spec in enumerate(plan):
            label='%02d-%s'%(i,spec['label'])
            try:
                phase(spec,base,out,label,guard,baseline, args.smoke_only)
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError,RuntimeError,AssertionError) as exc:
                runner.save(out/(label+'-failed.public.json'),dict(complete=False,error=repr(exc),notes=getattr(exc,'__notes__',[])))
                if not args.continue_failed or runner.gpu_pids(): raise
                if i == 0: raise  # No candidate series without a working baseline.
                print('REJECTED',label,repr(exc),flush=True)
            if baseline is None and (out/(label+'-means.public.json')).exists(): baseline=label
        runner.save(out/'completed.public.json',dict(complete=True,scope='Plan executed; failed points explicitly retained'))
    except BaseException as exc:
        primary=exc; raise
    finally:
        if lifecycle:
            try:
                # Operator stop-request affects trials, not restoration. Never restart a miner.
                assert all(runner.sha(Path(p))==h for p,h in hashes.items()), 'Protected target changed; restoration refused'
                assert not runner.gpu_pids(), 'Unowned/undrained GPU workload blocks production restoration'
                control('start_strata',out);runner.loaded(port=30001)
                native=runner.native(base['exe'])
                assert runner.sha(native/'exe')==hashes[base['exe']]
                runner.save(out/'restored.public.json',dict(loaded=True,native_pid=int(native.name),files=hashes,
                    engine=runner.api('metrics',port=30001)['engine']))
                print('PRODUCTION_RESTORED',native.name,flush=True)
            except BaseException as exc:
                if primary is None: raise
                if hasattr(primary,'add_note'): primary.add_note('Restoration: '+repr(exc))
                try: print('RESTORE_ERROR',repr(exc),file=sys.stderr,flush=True)
                except BaseException: pass


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path);parser.add_argument('--run')
    parser.add_argument('--smoke-only',action='store_true');parser.add_argument('--continue-failed',action='store_true')
    parser.add_argument('--measure',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--numerical',type=Path)
    parser.add_argument('--name');parser.add_argument('--label')
    args=parser.parse_args()
    if args.measure:
        worker(args.measure,args.output,args.name,args.label)
    elif args.numerical:
        numerical_worker(args.numerical,args.output,args.label)
    else:
        if not args.plan or not args.run or Path(args.run).name != args.run: parser.error('plan and safe run name required')
        main(args)
