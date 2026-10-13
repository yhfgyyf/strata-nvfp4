"""Compare the retained R256 trace and the new early-upload pair without running GPU work."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/home/tc/Strata-nvfp4/bench')
sys.path.insert(0, str(ROOT / 'prefill-residency-p0-p3-20261012/nvidia-next-20261012'))
from analyze_nvidia_trace import partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=ROOT / 'ple-upload-w4a16-20261013/step1-trace-2')
    parser.add_argument('--label', default='W4A16-early-trace')
    args = parser.parse_args()
    old = ROOT / 'prefill-residency-p0-p3-20261012/nvidia-ring-trace-final-1'
    new = args.folder
    rows = []
    for folder, wanted in [(old, 'W4A16-NTRACE-B'), (new, args.label + '-A'), (new, args.label + '-B')]:
        row, = [r for r in json.loads((folder / 'pipeline-nvidia.public.json').read_text()) if r['label'] == wanted]
        segment, = row['prefill_nvtx']
        start, end = segment['start_ns'], segment['end_ns']
        cut = start + 160_000_000
        with sqlite3.connect('file:' + str(folder / (wanted + '.sqlite')) + '?mode=ro', uri=True) as db:
            pid, = db.execute('SELECT globalPid FROM PROCESSES WHERE pid=?', (row['native_pid'],)).fetchone()
            strings = dict(db.execute('SELECT id,value FROM StringIds'))
            kernels = list(db.execute('SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL '
                'WHERE globalPid=? AND end>? AND start<? ORDER BY start', (pid, start, cut)))
            compute = [(a,b) for a,b,n in kernels if 'wait_flag' not in strings[n]]
            waits = [(a,b) for a,b,n in kernels if 'wait_flag' in strings[n]]
            copies = list(db.execute('SELECT start,end,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE globalPid=? AND end>? AND start<?', (pid, start, cut)))
            h2d = [(a,b) for a,b,k in copies if k == 1]
            other = [(a,b) for a,b,k in copies if k != 1]
            other += list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET '
                'WHERE globalPid=? AND end>? AND start<?', (pid, start, cut)))
            small = []
            for a,b,size,stream,correlation in db.execute('SELECT start,end,bytes,streamId,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE globalPid=? AND start>=? AND start<? AND copyKind=1 AND bytes!=2764816 ORDER BY start', (pid,start,cut)):
                calls = [c for c in db.execute('SELECT start,end,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?', (correlation,))
                         if ((c[2] >> 24) & 0xffffff) == row['native_pid']]
                small.append(dict(payload_bytes=size, stream_id=stream, device_ns=[a-start,b-start],
                                  host_ns=[[c[0]-start,c[1]-start] for c in calls], correlation_id=correlation))
            experts = list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                'WHERE globalPid=? AND start>=? AND start<? AND copyKind=1 AND bytes=2764816 ORDER BY start', (pid,start,cut)))
            selected_experts = [dict(ordinal=i+1,device_ns=[a-start,b-start]) for i,(a,b) in enumerate(experts)
                                if i in (0,255,256,294,295,550,551)]
        prefix = partition(start, cut, compute, h2d, other, waits)
        full = segment['coverage']
        later = {k: full['parts_ns'].get(k, 0) - prefix['parts_ns'].get(k, 0) for k in set(full['parts_ns']) | set(prefix['parts_ns'])}
        assert all(v >= 0 for v in later.values()) and sum(later.values()) == end - cut
        bins = [partition(t, min(t+5_000_000, cut), compute, h2d, other, waits)['parts_ns'] for t in range(start, cut, 5_000_000)]
        rows.append(dict(label=wanted, binary_sha256=row['startup']['binary_sha256'],
            report_sha256=row['report_sha256'], whole=full, first160=prefix, after160_parts_ns=later, bins5ms=bins,
            initial_loan=next(x for x in row['ledger'] if x['kind'] == 'loan'),
            small_copies=small, selected_experts=selected_experts,
            gr_writes_first160=[dict(start_ms=(a-start)/1e6,end_ms=(b-start)/1e6) for a,b,n in kernels if 'gr_write_kernel' in strings[n]]))
    result = dict(rows=rows, scope='Only selected native CUDA activity from retained traces. Fixed time cuts are temporal regions, not matching layer boundaries. No GPU workload or new causal experiment.')
    (new / 'original-r256-comparison.public.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({r['label']: dict(first160=r['first160'], after160=r['after160_parts_ns'], gr_writes=r['gr_writes_first160']) for r in rows}))


if __name__ == '__main__': main()
