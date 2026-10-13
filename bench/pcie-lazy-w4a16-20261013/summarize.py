"""Recompute bracketed comparisons from public per-request evidence."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(folder):
    arms=[]
    for path in sorted(folder.glob('*-means.public.json')):
        data=json.loads(path.read_text());label=path.name.removesuffix('-means.public.json')
        startup=json.loads((folder/(label+'-startup.public.json')).read_text())
        rows=[]
        for entry in data['rows']:
            individual=entry['individual']
            assert len(individual)==3 and all(r['output_tokens']==512 and r['reused']==0 for r in individual)
            for metric,value in entry['mean'].items():
                assert abs(statistics.mean(r[metric] for r in individual)-value)<1e-8
            rows.append(dict(input_tokens=entry['input_tokens'],mean=entry['mean'],sample_stdev=entry['sample_stdev'],
                scores=[r['quality']['score'] for r in individual],
                spec_acceptance=[r['native']['drafts_accepted']/r['native']['drafts_offered']
                    if r['native']['drafts_offered'] else None for r in individual],
                pcie_share=[r['native']['pcie_share'] for r in individual],
                repetition=any(r['quality']['periodic_tail'] or r['quality']['long_digit_loop'] for r in individual)))
        arms.append(dict(label=label,spec=data['spec'],expert_slots=startup['engine']['expert_slots'],rows=rows))
    by_label={a['label']:a for a in arms};pairs=[]
    plan=json.loads((folder/'plan.public.json').read_text())
    for i,spec in enumerate(plan):
        if not spec['label'].startswith('B-'):continue
        labels=['%02d-%s'%(j,plan[j]['label']) for j in (i-1,i,i+1)]
        if not all(label in by_label for label in labels):continue
        before,candidate,after=[by_label[label] for label in labels];comparison=[]
        for a,b,c in zip(before['rows'],candidate['rows'],after['rows']):
            assert a['input_tokens']==b['input_tokens']==c['input_tokens']
            change={};drift={}
            for metric in ('prefill_tps','decode_native_tps','ttft_s'):
                ref=(a['mean'][metric]+c['mean'][metric])/2
                change[metric]=100*(b['mean'][metric]/ref-1)
                drift[metric]=100*(c['mean'][metric]/a['mean'][metric]-1)
            comparison.append(dict(input_tokens=b['input_tokens'],candidate_change_percent=change,
                control_drift_percent=drift,quality_candidate=b['scores'],quality_before=a['scores'],quality_after=c['scores']))
        pairs.append(dict(labels=labels,rows=comparison))
    return dict(arms=arms,bracketed=pairs,
        failures=[dict(file=p.name,**json.loads(p.read_text())) for p in sorted(folder.glob('*-failed.public.json'))],
        statistics='Each arm has three trials per size; B compared to mean of adjacent A means; drift is A-after vs A-before')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    print(json.dumps(summarize(args.folder),indent=2))
