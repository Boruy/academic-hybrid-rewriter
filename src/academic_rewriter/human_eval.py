"""Export blinded human-review packets or summarize completed independent ratings."""
import argparse
import hashlib
import itertools
import json
import random
import secrets
import statistics
from pathlib import Path
from .evaluation_common import (read_json,read_jsonl,write_json,load_aligned,words,
                                clustered_interval,paired_difference,quadratic_kappa)
PROJECT=Path(__file__).resolve().parents[2]
RATINGS=['meaning','academic','fluency','academic_improvement','major_fact_error']


def text_digest(source,candidate):
    return hashlib.sha256((source+'\0'+candidate).encode()).hexdigest()


def validate_rating(row):
    if any(row.get(k) is None for k in RATINGS):return False
    for key in ['meaning','academic','fluency']:
        if type(row[key]) is not int or not 1<=row[key]<=5:raise ValueError('Ratings must be integers from 1 to 5')
    if type(row['academic_improvement']) is not int or not -2<=row['academic_improvement']<=2:
        raise ValueError('Academic improvement must be an integer from -2 to +2')
    if type(row['major_fact_error']) is not bool:raise ValueError('Major fact error must be an explicit boolean')
    return True


def successful_rewrite(rows,source,candidate,thresholds):
    if len(rows)<thresholds['required_raters'] or source.strip()==candidate.strip():return False
    return all(validate_rating(r) and r['meaning']>=thresholds['minimum_meaning'] and
        r['academic']>=thresholds['minimum_academic'] and r['fluency']>=thresholds['minimum_fluency'] and
        r['academic_improvement']>=thresholds['minimum_improvement'] and not r['major_fact_error'] for r in rows)


def export(args):
    config=read_json(PROJECT/args.config)
    data,candidates,labels=load_aligned(PROJECT,config)
    if args.sample<1 or args.sample>len(data) or args.raters<2:
        raise ValueError(f'Sample must be 1..{len(data)}; at least two independent raters required')
    out=PROJECT/args.output; keypath=PROJECT/args.key
    if keypath.exists() or out.exists() and any(out.iterdir()):raise ValueError('Use new packet and private-key paths; never overwrite an ongoing review')
    rng=random.Random(config['seed']);chosen=rng.sample(list(range(len(data))),args.sample)
    salt=secrets.token_hex(16);items={};packet=[]
    for i in chosen:
        case=data[i]
        for model,texts in candidates.items():
            candidate=texts[i]
            uid=hashlib.sha256((salt+case['id']+'\0'+model).encode()).hexdigest()[:20]
            items[uid]=dict(case_id=case['id'],model=model,group_id=case['group_id'],
                digest=text_digest(case['source'],candidate),source=case['source'],candidate=candidate)
            packet.append(dict(item_id=uid,source=case['source'],candidate=candidate,
                **{k:None for k in RATINGS},comment=''))
    template=(PROJECT/'scripts/templates/human_rating.html').read_text(encoding='utf8')
    out.mkdir(parents=True,exist_ok=True)
    for rater in range(1,args.raters+1):
        name=f'rater_{rater}'
        rows=[dict(rater_id=name,**r) for r in packet];random.Random(config['seed']+rater).shuffle(rows)
        with (out/(name+'.jsonl')).open('w',encoding='utf8',newline='\n') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        payload=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
        (out/(name+'.html')).write_text(template.replace('__RATING_DATA__',payload),encoding='utf8',newline='\n')
    key=dict(schema_version=1,items=items,labels=labels,raters=[f'rater_{i}' for i in range(1,args.raters+1)],
        thresholds=config['human'],evaluation_status=config['evaluation_status'],case_ids=[data[i]['id'] for i in chosen])
    write_json(keypath,key)
    write_json(out/'study.json',dict(sample_cases=args.sample,available_cases=len(data),candidates_per_case=len(candidates),
        raters=args.raters,items_per_rater=len(packet),seed=config['seed'],evaluation_status=config['evaluation_status'],
        private_key_sha256=hashlib.sha256(keypath.read_bytes()).hexdigest(),ratings_status='unfilled',
        note='These are already inspected diagnostic examples. Packets contain no system names. Send only a rater packet and rubric to a reviewer, never the private key.'))
    print(f'Created {args.raters} blinded packets of {len(packet)} candidates; private mapping remains at {keypath}',flush=True)


def summarize(args):
    key=read_json(PROJECT/args.key);items=key['items'];raters=key['raters'];all_rows={};incomplete=0
    for path in args.ratings:
        for row in read_jsonl(PROJECT/path):
            who=row.get('rater_id');uid=row.get('item_id')
            if who not in raters or uid not in items:raise ValueError('Unknown rater or item; use the matching private key')
            if (who,uid) in all_rows:raise ValueError('Duplicate rater/item rating')
            item=items[uid]
            if text_digest(row['source'],row['candidate'])!=item['digest']:raise ValueError('Rating text was changed or belongs to a different packet')
            complete=validate_rating(row)
            all_rows[(who,uid)]=row if complete else None
            incomplete+=not complete
    complete_items={uid:[all_rows[(who,uid)] for who in raters] for uid in items if all(all_rows.get((who,uid)) is not None for who in raters)}
    agreement={}
    for a,b in itertools.combinations(raters,2):
        common=[uid for uid in items if all_rows.get((a,uid)) is not None and all_rows.get((b,uid)) is not None]
        result={}
        if common:
            for field in ['meaning','academic','fluency','academic_improvement']:
                lo,hi=(-2,2) if field=='academic_improvement' else (1,5)
                result[field]=quadratic_kappa([all_rows[(a,u)][field] for u in common],[all_rows[(b,u)][field] for u in common],lo,hi)
        agreement[a+'_vs_'+b]=dict(n_common=len(common),quadratic_weighted_kappa=result,
            note='Null kappa means undefined constant marginals or no shared completed items; it is not perfect agreement.')
    models={};per_model={}
    for model,label in key['labels'].items():
        expected=[u for u,v in items.items() if v['model']==model]
        completed=[u for u in expected if u in complete_items]
        fully_complete=len(completed)==len(expected) and len(raters)>=2
        result=dict(label=label,expected=len(expected),completed=len(completed),successful_rewrite_rate=None)
        if completed:
            result['mean_ratings']={field:statistics.mean(r[field] for u in completed for r in complete_items[u]) for field in ['meaning','academic','fluency','academic_improvement']}
        if fully_complete:
            success=[float(successful_rewrite(complete_items[u],items[u]['source'],items[u]['candidate'],key['thresholds'])) for u in completed]
            result['provisional_successful_rewrite_rate']=clustered_interval(success,[items[u]['group_id'] for u in completed])
            if key['thresholds']['thresholds_calibrated']:result['successful_rewrite_rate']=result['provisional_successful_rewrite_rate']
            per_model[model]={items[u]['case_id']:(v,items[u]['group_id']) for u,v in zip(completed,success)}
        models[model]=result
    paired={}
    baseline=next((m for m in key['labels'] if m not in {'copy_source','human_reference'}),None)
    if baseline in per_model:
        for model,values in per_model.items():
            if model==baseline:continue
            ids=sorted(set(values).intersection(per_model[baseline]))
            if len(ids)==len(key['case_ids']):
                paired[model+'_minus_'+baseline]=paired_difference([values[i][0] for i in ids],[per_model[baseline][i][0] for i in ids],[values[i][1] for i in ids])
    summary=dict(status='complete_pilot' if len(complete_items)==len(items) else 'pending_ratings',
        evaluation_status=key['evaluation_status'],thresholds=key['thresholds'],raters=len(raters),
        complete_candidates=len(complete_items),expected_candidates=len(items),incomplete_rows=incomplete,
        models=models,agreement=agreement,paired_success_differences=paired,
        note='Success requires every rater to meet all thresholds, positive academic improvement, no major fact error, and a changed output. Thresholds are experimental; no human scores are imputed. Independent identities are an organizational requirement, not verifiable from files alone.')
    write_json(PROJECT/args.summary,summary)
    print(f'Human review: {summary["status"]}, {len(complete_items)}/{len(items)} fully rated candidates',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('export');p.add_argument('--config',default='configs/evaluation.json');p.add_argument('--sample',type=int,default=24);p.add_argument('--raters',type=int,default=2);p.add_argument('--output',default='output/human-evaluation');p.add_argument('--key',default='.cache/human-evaluation/key.json');p.set_defaults(run=export)
    p=sub.add_parser('summarize');p.add_argument('--key',default='.cache/human-evaluation/key.json');p.add_argument('--ratings',nargs='+',required=True);p.add_argument('--summary',default='output/human-evaluation/summary.json');p.set_defaults(run=summarize)
    args=parser.parse_args();args.run(args)

if __name__=='__main__':main()
