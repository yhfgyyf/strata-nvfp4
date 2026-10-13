"""Validate the saved public campaign, without fetching private model responses."""
import hashlib
import json
from pathlib import Path
import statistics

from summarize import summarize


ROOT = Path(__file__).resolve().parent
SIZES = (8192, 32768, 131072)


def load(path):
    return json.loads(path.read_text())


def verify_run(name, expected_arms):
    folder = ROOT / 'evidence-public' / name
    before = load(folder / 'before.public.json')
    restored = load(folder / 'restored.public.json')
    assert load(folder / 'completed.public.json')['complete']
    assert restored['loaded'] and before['files'] == restored['files']
    helper = ROOT / ('evidence-public/runner-v1.py' if name == 'matrix-r1' else 'run_matrix.py')
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == before['helper_sha256']
    summary = summarize(folder)
    assert len(summary['arms']) == expected_arms and not summary['failures']
    assert summary == load(folder / 'summary.json')
    rows_checked = 0
    for arm in summary['arms']:
        label = arm['label']
        startup = load(folder / (label + '-startup.public.json'))
        env = startup['env']
        assert startup['binary_sha256'] == before['files'][startup['native_args'][0]]
        assert env['STRATA_NVFP4_W4A16'] == '1' and env['STRATA_NVFP4_FUSED_MOE'] == '2'
        assert env['STRATA_PREFILL_RING'] == '256' and env['STRATA_PREFILL_NVFP4'] == 'fp16'
        assert env['STRATA_PREFILL_NVFP4_FP8'] == '0'
        assert env['STRATA_PLE_EARLY_UPLOAD'] == '1' and env['STRATA_PLE_RAW_UPLOAD'] == '1'
        assert env['STRATA_PLE_RAW_CACHE_MIB'] == '64'
        assert startup['engine']['context'] == 262144 and startup['engine']['kv'] == 'int8'
        expected_slots = 0 if arm['spec'].get('parallel') == 1 else 2
        assert int(startup['engine'].get('batch_slots') or 0) == expected_slots
        measurements = load(folder / (label + '-means.public.json'))
        assert measurements['complete'] and measurements['instrumented'] is False
        for entry in measurements['rows']:
            for metric, value in entry['sample_stdev'].items():
                assert abs(statistics.stdev(r[metric] for r in entry['individual']) - value) < 1e-8
        for size in SIZES:
            numerical = load(folder / (label + '-numerical-%d.public.json' % size))
            assert numerical['input_tokens'] == size and numerical['output_tokens'] == 1
            assert numerical['reused'] == 0 and numerical['finite'] and numerical['logits_values'] == 248320
            assert numerical['diagnostic']['command'] == ('GEN' if expected_slots == 0 else 'BGEN')
            if arm['spec'].get('lag'):
                assert numerical['diagnostic']['adapt_nowait'] == 0
                assert numerical['diagnostic']['adapt_lag'] == arm['spec']['lag']
            for trial in range(3):
                row = load(folder / (label + '-cold-%d-%d.public.json' % (size, trial)))
                assert row['actual_decode_path'] == 'GEN' and row['input_tokens'] == size
                assert row['output_tokens'] == 512 and row['full_output'] and row['reused'] == 0
                assert row['quality']['score'] == 5 and row['quality']['maximum'] == 6
                assert not row['quality']['periodic_tail'] and not row['quality']['long_digit_loop']
                assert row['config_sha256'] == startup['config_sha256']
                assert row['binary_sha256'] == startup['binary_sha256']
                assert row['profile_sha256'] == before['files']['/home/tc/Strata-data/packs/nvidia-nvfp4-raw/expert-profile.bin']
                assert (row['per_layer_window']['pcie_experts'] > 0) == (arm['spec']['frac'] > 0)
                reference = load(ROOT / 'evidence-public/matrix-r1' /
                                 ('00-A0-kernel-0-cold-%d-%d.public.json' % (size, trial)))
                assert row['payload_sha256'] == reference['payload_sha256']
                rows_checked += 1
    return dict(run=name, arms=expected_arms, cold_requests=rows_checked,
                restoration_files_match=True, means_and_stdev_recomputed=True)


if __name__ == '__main__':
    results = [verify_run(name, count) for name, count in
               (('matrix-r1', 11), ('lazy-r1', 5), ('rotation-r1', 5))]
    folder = ROOT / 'evidence-public/accepted-rotation'
    deployment = load(folder / 'deployment.public.json')
    service = load(folder / 'service-check.public.json')
    assert deployment['applied'] and deployment['verified'] and service['passed']
    assert deployment['semantic_diff'] == {'env.STRATA_EXCHANGE_ROTATE': [None, '1']}
    assert service['env']['STRATA_EXCHANGE_ROTATE'] == '1' and service['peak_running'] == 2
    assert len(service['checks']) == 10 and all(row['passed'] for row in service['checks'])
    assert deployment['after_files'] == service['files']
    assert service['engine']['context'] == 262144 and service['engine']['batch_slots'] == 2
    assert service['engine']['expert_slots'] == 10295 and service['engine']['kv'] == 'int8'
    print(json.dumps(dict(passed=True, runs=results, saved_live_checks_validated=10,
        scope='Offline public-evidence validation; no GPU run or API health check'), indent=2))
