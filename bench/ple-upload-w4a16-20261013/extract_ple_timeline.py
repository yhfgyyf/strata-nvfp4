"""Correlate the single 8K PLE copy with its host API and prefill NVTX; keep SQLite on the server."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def extract(folder):
    results = []
    for row in json.loads((folder / 'pipeline-nvidia.public.json').read_text()):
        label = row['label']
        pid = row['startup']['native_pid']
        segment, = row['prefill_nvtx']
        start, end = segment['start_ns'], segment['end_ns']
        with sqlite3.connect('file:' + str(folder / (label + '.sqlite')) + '?mode=ro', uri=True) as db:
            global_pid, = db.execute('SELECT globalPid FROM PROCESSES WHERE pid=?', (pid,)).fetchone()
            strings = dict(db.execute('SELECT id,value FROM StringIds'))
            # Only one full PLE payload in the captured 8K prefill: N=2560, excluding the scalar seed tokens.
            copies = list(db.execute('SELECT start,end,bytes,streamId,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE globalPid=? AND copyKind=1 AND start>=? AND end<=? AND bytes>? ORDER BY start',
                (global_pid, start, end, 16 * 1024 * 1024)))
            copy, = copies
            a, b, size, stream, correlation = copy
            calls = list(db.execute('SELECT start,end,globalTid,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?', (correlation,)))
            calls = [c for c in calls if ((c[2] >> 24) & 0xffffff) == pid]
            host, = calls
            assert host[4] == 0
        group, = [g for g in row['h2d_groups'] if g['stream_id'] == stream and g['bytes_per_copy'] == size]
        results.append(dict(label=label, native_pid=pid, prefill_window_ms=(end-start)/1e6,
            ple_payload_bytes=size, stream_id=stream, correlation_id=correlation,
            host_api=strings[host[3]], host_submit_ms=(host[0]-start)/1e6,
            host_return_ms=(host[1]-start)/1e6, device_start_ms=(a-start)/1e6,
            device_end_ms=(b-start)/1e6, host_submit_to_device_start_ms=(a-host[0])/1e6,
            copy_ms=(b-a)/1e6, overlap_compute_ms=group['overlap_compute_ms'],
            prefill_partition_ns=segment['coverage'], report_sha256=row['report_sha256']))
    result = dict(captures=results, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One instrumented request per arm. Native CUDA activity, matched correlation and NVTX. No layer-level GPU marker or CPU sampling; activity overlap does not establish removable latency.')
    (folder / 'ple-timing.public.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.folder)))
