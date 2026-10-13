"""Verify the public deployment receipt separately from historical A/B/A evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence-public/production-20261013'


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def main():
    deployment = read('deployment.public.json')
    startup = read('startup.public.json')
    controls = dict(STRATA_PLE_EARLY_UPLOAD='1', STRATA_PLE_RAW_UPLOAD='1', STRATA_PLE_RAW_CACHE_MIB='64')
    assert deployment['complete'] and startup['loaded']
    assert deployment['controls'] == startup['ple_controls'] == controls
    assert deployment['removed_arguments'] == ['--vram-reserve-mib', '1536']
    for key in ('binary_sha256', 'config_sha256', 'script_sha256', 'native_pid'):
        assert deployment[key] == startup[key]
    assert deployment['binary_sha256'] == '4ae23d2d995866b3ddf98525b0de700d3fb48e289135fea39f5643d2cd55af78'
    for engine in (startup['engine'], deployment['engine']):
        assert engine['context'] == 262144 and engine['batch_slots'] == 2
        assert engine['kv'] == 'int8' and engine['images']
        assert engine['expert_slots'] == 10295
    assert not deployment['miner_running']
    repo = HERE.parents[1]
    for name, digest in deployment['source_sha256'].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    ownership = read('ownership.public.json')
    assert ownership['passed'] and ownership['only_intended_config_changes']
    assert '--vram-reserve-mib' not in ownership['native_args']
    for flag, value in (('--prefill', '8192'), ('--max-context', '262144'), ('--batch', '2')):
        assert ownership['native_args'][ownership['native_args'].index(flag) + 1] == value
    for name in ('api-smoke.public.json', 'gateway.public.json', 'repo-tests.public.json'):
        assert read(name)['complete']
    samples = read('long-input.public.json')
    assert samples['complete'] and len(samples['samples']) == 3
    assert sorted(r['input_tokens'] for r in samples['samples']) == [8192, 32768, 131072]
    for row in samples['samples']:
        assert row['full_output'] and row['output_tokens'] == 512 and row['reused'] == 0
        assert row['binary_sha256'] == deployment['binary_sha256']
        assert row['config_sha256'] == deployment['config_sha256']
        assert not row['quality']['periodic_tail'] and not row['quality']['long_digit_loop']
    capacity = read('capacity.public.json')
    assert capacity['complete'] and capacity['peak_running'] == 2 and len(capacity['checks']) == 2
    for row in capacity['checks']:
        assert row['usage']['input_tokens'] == 261624 and row['usage']['output_tokens'] == 512
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(EVIDENCE.iterdir()) if p.is_file()}
    result = dict(passed=True, sha256=hashes, native_pid=deployment['native_pid'],
                  binary_sha256=deployment['binary_sha256'], cold_full_output_samples=3,
                  full_context_slots=2,
                  scope='Saved deployment receipt/hash and API/capacity checks; not a new performance comparison')
    (HERE / 'deployment-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'sha256'}))


if __name__ == '__main__': main()
