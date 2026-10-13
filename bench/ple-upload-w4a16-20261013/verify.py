"""Verify public hashes, three-trial arithmetic, numerical receipts and byte budgets."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def verify():
    manifest = read(HERE / 'manifest.json')
    for name, digest in manifest['sha256'].items():
        path = SOURCE / name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    build = read(HERE / 'evidence-public/step1-v3-r1/build.public.json')
    for obj in build['objects']:
        assert hashlib.sha256((SOURCE / obj['source']).read_bytes()).hexdigest() == obj['sha256']
    for name, digest in build['header_sha256'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    upload = read(HERE / 'evidence-public/upload-step1-v3/upload.public.json')
    for name in ('ple_upload_cuda_test.cu', 'ple_upload_fence_test.cpp'):
        path = HERE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == upload[str(path.relative_to(SOURCE))]
    records = 0
    identical_quality_pairs = 0
    quality_keys = ('score', 'maximum', 'correct', 'json_prefix_present', 'periodic_tail',
                    'long_digit_loop', 'tool_calls', 'valid_tool_json')
    for run, label in manifest['campaigns'].items():
        folder = HERE / 'evidence-public' / run
        summary = read(folder / 'means.public.json')
        assert summary['records'] == 27
        assert read(folder / 'restored.public.json')['loaded']
        for result in summary['comparisons']:
            size = result['input_tokens']
            for arm in ('A1', 'B', 'A2'):
                rows = [read(folder / f'{label}-{arm}-cold-{size}-{case}.public.json') for case in range(3)]
                assert all(r['input_tokens'] == size and r['reused'] == 0 and r['full_output'] and r['output_tokens'] == 512 for r in rows)
                for key, mean in result['mean_by_arm'][arm].items():
                    assert math.isclose(statistics.mean(r[key] for r in rows), mean, rel_tol=1e-12)
                    assert math.isclose(statistics.stdev(r[key] for r in rows), result['sample_stdev_by_arm'][arm][key], rel_tol=1e-12)
                records += len(rows)
            for pair in result['pairs']:
                rows = {arm: read(folder / f'{label}-{arm}-cold-{size}-{pair["case"]}.public.json') for arm in ('A1', 'B', 'A2')}
                assert len({r['payload_sha256'] for r in rows.values()}) == 1
                assert len({r['profile_sha256'] for r in rows.values()}) == 1
                same = all(len({json.dumps(r['quality'][key], sort_keys=True) for r in rows.values()}) == 1 for key in quality_keys)
                assert same == pair['quality_same']
                identical_quality_pairs += same
                assert all(not r['quality']['periodic_tail'] and not r['quality']['long_digit_loop'] for r in rows.values())
                assert rows['B']['quality']['score'] >= min(rows['A1']['quality']['score'], rows['A2']['quality']['score'])
                for key, delta in pair['change_percent'].items():
                    expected = 100 * (rows['B'][key] / statistics.mean([rows['A1'][key], rows['A2'][key]]) - 1)
                    assert math.isclose(expected, delta, abs_tol=1e-10)
            for key, delta in result['change_from_arm_means_percent'].items():
                means = result['mean_by_arm']
                expected = 100 * (means['B'][key] / statistics.mean([means['A1'][key], means['A2'][key]]) - 1)
                assert math.isclose(expected, delta, abs_tol=1e-10)
        for arm in ('B', 'A2'):
            numerical = read(folder / (label + '-' + arm + '-numerical-comparison.public.json'))['rows']
            assert len(numerical) == 3 and all(r['bit_equal'] and r['baseline_sha256'] == r['candidate_sha256'] for r in numerical)
    assert records == 135
    for run, label in [('step3-aba-1', 'W4A16-cache-B'), ('combined-checks-1', 'W4A16-combined')]:
        capacity = read(HERE / 'evidence-public' / run / (label + '-capacity.public.json'))
        assert capacity['complete']
        slots = [r for r in capacity['checks'] if r['name'].startswith('context-slot-')]
        assert len(slots) == 2 and all(r['usage']['input_tokens'] == 261624 and r['usage']['output_tokens'] == 512 and r['peak_running'] == 2 for r in slots)
    combined = HERE / 'evidence-public/combined-checks-1'
    assert all(r['bit_equal'] for r in read(combined / 'numerical-comparison.public.json')['rows'])
    row = read(combined / 'W4A16-combined-cold-8192-0.public.json')
    loan, = [r for r in row['ledger'] if r['kind'] == 'loan']
    raw, = [r for r in row['ledger'] if r['kind'] == 'ple_raw_rows']
    assert loan['cache_bytes'] == loan['cache_slots'] * 2765056
    assert loan['loan_span_bytes'] == loan['borrowed_experts'] * 2765056
    assert loan['borrowed_experts'] == (loan['workspace_bytes'] + 2765055) // 2765056
    assert loan['effective_resident'] == loan['cache_slots'] - loan['borrowed_experts']
    assert raw['cache_owned_bytes'] <= 64 << 20
    assert raw['fetched_rows'] == raw['cache_hits'] + raw['backing_rows']
    assert raw['raw_output_bytes'] == raw['decoded_elements'] == raw['occurrences'] * 160
    assert raw['backing_payload_bytes'] == raw['backing_rows'] * 160
    runtime = read(HERE / 'evidence-public/final-runtime.public.json')
    assert runtime['loaded'] and not runtime['miner_running'] and len(runtime['checks']) == 6 and all(r['passed'] for r in runtime['checks'])
    assert runtime['binary_sha256'] == '1b827daf441e701d17b12c77a67cee9b2ead8015288d1392e191f063aa469c39'
    assert read(HERE / 'evidence-public/humanize-cancellation.public.json')['status'] == 'cancelled_by_user'
    return dict(passed=True, hashed_files=len(manifest['sha256']), complete_cold_records=records,
                campaigns=5, tested_v3_sources_match=True, identical_quality_pairs=identical_quality_pairs, total_quality_pairs=45,
                combined_capacity=True, production_restored=True,
                scope='Public arithmetic/hash/receipt verification; GPU tests have their separately retained evidence')


if __name__ == '__main__':
    result = verify()
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
