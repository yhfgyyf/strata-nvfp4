"""Report three matched trials per length/arm as arithmetic means, retaining control drift."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(folder, label):
    metrics = ('prefill_tps', 'decode_native_tps', 'ttft_s', 'prompt_ms')
    quality_keys = ('score', 'maximum', 'correct', 'json_prefix_present', 'periodic_tail',
                    'long_digit_loop', 'tool_calls', 'valid_tool_json')
    comparisons = []
    files = {}
    for size in (8192, 32768, 131072):
        arms = {}
        for arm in ('A1', 'B', 'A2'):
            arms[arm] = []
            for case in range(3):
                path = folder / f'{label}-{arm}-cold-{size}-{case}.public.json'
                data = path.read_bytes()
                row = json.loads(data)
                assert row['input_tokens'] == size and row['reused'] == 0 and row['output_tokens'] == 512 and row['full_output']
                files[path.name] = hashlib.sha256(data).hexdigest()
                arms[arm].append(row)
        means = {arm: {key: statistics.mean(r[key] for r in rows) for key in metrics} for arm, rows in arms.items()}
        stdev = {arm: {key: statistics.stdev(r[key] for r in rows) for key in metrics} for arm, rows in arms.items()}
        pairs = []
        for case in range(3):
            rows = {arm: arms[arm][case] for arm in arms}
            assert len({r['payload_sha256'] for r in rows.values()}) == 1
            assert len({r['profile_sha256'] for r in rows.values()}) == 1
            pairs.append(dict(case=case, payload_sha256=rows['A1']['payload_sha256'],
                change_percent={key: 100 * (rows['B'][key] / statistics.mean([rows['A1'][key], rows['A2'][key]]) - 1) for key in metrics},
                control_drift_percent={key: 100 * (rows['A2'][key] / rows['A1'][key] - 1) for key in metrics},
                quality_same=all(len({json.dumps(r['quality'][key], sort_keys=True) for r in rows.values()}) == 1 for key in quality_keys)))
        comparisons.append(dict(input_tokens=size, trials_per_arm=3, mean_by_arm=means, sample_stdev_by_arm=stdev,
            change_from_arm_means_percent={key: 100 * (means['B'][key] / statistics.mean([means['A1'][key], means['A2'][key]]) - 1) for key in metrics},
            control_mean_drift_percent={key: 100 * (means['A2'][key] / means['A1'][key] - 1) for key in metrics}, pairs=pairs))
    result = dict(complete=True, records=27, reporting_rule='Arithmetic mean of3 matched fixed input seeds per length/arm; retain all individual rows and A/A drift',
                  comparisons=comparisons, source_files_sha256=files,
                  quality_scope='Six-task deterministic screening and loop checks, not general model accuracy')
    (folder / 'means.public.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    result = summarize(args.folder, args.label)
    print(json.dumps({k: result[k] for k in ('records', 'reporting_rule', 'comparisons')}))
