"""Extract cumulative adaptive exchanges from retained native request logs.

Run on the server. Only counters and hashes are printed; prompts/logits stay there.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


BLOB_BYTES = 2764816
COUNTER = re.compile(r'strata serve: resident RAM: ([\d.]+) GiB of experts in RAM, (\d+) exchanged with the VRAM tier, (\d+) blob reads from the file')


def collect(folder):
    plan = json.loads((folder / 'plan.public.json').read_text())
    arms = []
    for index, spec in enumerate(plan):
        label = '%02d-%s' % (index, spec['label'])
        startup = folder / (label + '-engine.private.log')
        activation = [line for line in startup.read_text().splitlines()
                      if line.startswith('FileExpertSource: exchange buffer rotation enabled:')]
        assert bool(activation) == bool(spec.get('rotate'))
        snapshots = []
        for path in folder.glob(label + '-*-0.engine.log'):
            if '-warmup-' not in path.name and '-cold-' not in path.name:
                continue
            row_path = path.with_name(path.name.removesuffix('-0.engine.log') + '.public.json')
            row = json.loads(row_path.read_text())
            matches = COUNTER.findall(path.read_text())
            assert len(matches) == 1, 'Missing or ambiguous exchange summary: ' + path.name
            ram, count, reads = matches[0]
            snapshots.append(dict(label=row['label'], time=row['native']['time'],
                cumulative_exchanges=int(count), resident_ram_gib=float(ram), file_blob_reads=int(reads),
                engine_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        snapshots.sort(key=lambda row: row['time'])
        assert len(snapshots) == 12
        assert all(b['cumulative_exchanges'] >= a['cumulative_exchanges']
                   for a, b in zip(snapshots, snapshots[1:]))
        boundary = next(row for row in snapshots if row['label'] == label + '-warmup-2')
        cold = [row for row in snapshots if '-cold-' in row['label']]
        assert len(cold) == 9 and all(row['time'] > boundary['time'] for row in cold)
        count = cold[-1]['cumulative_exchanges'] - boundary['cumulative_exchanges']
        arms.append(dict(label=label, rotation_activation=activation, snapshots=snapshots,
            cold_interval_committed_exchanges=count,
            avoided_host_memcpy_payload_bytes=count * BLOB_BYTES if activation else 0))
    return dict(expert_blob_bytes=BLOB_BYTES, arms=arms,
        scope='Committed adaptive exchanges between warmup-2 and last cold completion snapshots. '
              'Pending commits may cross request boundaries. Rotation skips host commit memcpy; '
              'payload is not physical DRAM traffic and does not remove H2D/D2H or prove critical-path savings.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.folder), indent=2))
