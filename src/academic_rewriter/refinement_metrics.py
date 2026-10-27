"""Summarize real, independent output ratings by the source's revision need."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import statistics

from .evaluation_common import clustered_interval, paired_difference, read_jsonl, write_json, quadratic_kappa
from .human_eval import successful_rewrite, validate_rating, text_digest

THRESHOLDS=dict(required_raters=2,minimum_meaning=4,minimum_academic=4,minimum_fluency=4,minimum_improvement=1)

def summarize(cases,ratings):
    """cases carry source-only human judgments made before model outputs exist."""
    lookup={}; grouped=defaultdict(list); source_judgments={}
    for case in cases:
        key=(case['id'],case['model'])
        if key in lookup: raise ValueError('Duplicate case/model')
        if type(case.get('needs_revision')) is not bool or case.get('source_judgment_human_verified') is not True: raise ValueError('A prior human source-only need judgment is required')
        judgment=(case['source'],case['doc_id'],case['needs_revision'])
        if case['id'] in source_judgments and source_judgments[case['id']]!=judgment: raise ValueError('Models must use identical source-only need judgments')
        source_judgments[case['id']]=judgment
        lookup[key]=case
    seen=set()
    for row in ratings:
        key=(row['id'],row['model']); identity=(*key,row['rater_id'])
        if key not in lookup or identity in seen: raise ValueError('Unknown item or repeated reviewer/item')
        seen.add(identity); case=lookup[key]
        if not isinstance(row['rater_id'],str) or not row['rater_id'].strip(): raise ValueError('A reviewer identity is required')
        if row.get('reviewer_type')!='human': raise ValueError('Only actual human output ratings are eligible')
        if row.get('text_digest')!=text_digest(case['source'],case['prediction']): raise ValueError('Rating is not bound to the generated text')
        if not validate_rating(row): continue
        if not case['needs_revision'] and type(row.get('unnecessary_edit')) is not bool: continue
        grouped[key].append(row)
    models={}; rates={}
    for model in sorted({c['model'] for c in cases}):
        expected=[c for c in cases if c['model']==model]; complete=[c for c in expected if len(grouped[(c['id'],model)])>=2]
        result=dict(expected_cases=len(expected),complete_cases=len(complete),status='complete_pilot' if len(expected)==len(complete) else 'pending_ratings',effective_revision_rate=None,unnecessary_edit_rate=None,academic_tone_improvement=None,meaning_preservation=None,thresholds_calibrated=False)
        if complete:
            result['descriptive_mean_ratings']={field:statistics.mean(r[field] for c in complete for r in grouped[(c['id'],model)]) for field in ['meaning','academic','fluency','academic_improvement']}
            result['major_fact_error_candidate_count']=sum(any(r['major_fact_error'] for r in grouped[(c['id'],model)]) for c in complete)
            by_rater=defaultdict(dict)
            for case in complete:
                for row in grouped[(case['id'],model)]:by_rater[row['rater_id']][case['id']]=row
            result['agreement']={}
            for a,b in itertools.combinations(sorted(by_rater),2):
                common=sorted(set(by_rater[a])&set(by_rater[b]))
                result['agreement'][a+'_vs_'+b]=dict(common_cases=len(common),quadratic_weighted_kappa={field:quadratic_kappa([by_rater[a][i][field] for i in common],[by_rater[b][i][field] for i in common],-2 if field=='academic_improvement' else 1,2 if field=='academic_improvement' else 5) if common else None for field in ['meaning','academic','fluency','academic_improvement']})
        if len(complete)==len(expected) and complete:
            needs=[c for c in complete if c['needs_revision']];suitable=[c for c in complete if not c['needs_revision']]
            for name,subset in [('effective_revision_rate',needs),('unnecessary_edit_rate',suitable)]:
                if not subset:continue
                values=[]
                for case in subset:
                    rs=grouped[(case['id'],model)];changed=case['source'].strip()!=case['prediction'].strip()
                    values.append(float(successful_rewrite(rs,case['source'],case['prediction'],THRESHOLDS)) if name=='effective_revision_rate' else float(changed and any(r['unnecessary_edit'] for r in rs)))
                result['provisional_'+name]=clustered_interval(values,[c['doc_id'] for c in subset])
                rates[(model,name)]={c['id']:(value,c['doc_id']) for c,value in zip(subset,values)}
            result['academic_tone_improvement']=result['descriptive_mean_ratings']['academic_improvement']
            result['meaning_preservation']=result['descriptive_mean_ratings']['meaning']
        models[model]=result
    paired={}
    for a,b in itertools.combinations(sorted(models),2):
        for name in ['effective_revision_rate','unnecessary_edit_rate']:
            x=rates.get((a,name),{});y=rates.get((b,name),{})
            if x and set(x)==set(y):
                ids=sorted(x);paired[a+'_minus_'+b+'_'+name]=paired_difference([x[i][0] for i in ids],[y[i][0] for i in ids],[x[i][1] for i in ids])
    return dict(status='complete_pilot' if models and all(m['status']=='complete_pilot' for m in models.values()) else 'pending_ratings',models=models,paired_provisional_rate_differences=paired,thresholds=THRESHOLDS,note='Rates remain provisional until thresholds are calibrated independently. Effective revisions are measured only on sources needing refinement; unnecessary edits only on already-suitable sources and require an explicit unnecessary-edit judgment. Missing/partial ratings do not produce a zero rate. Null kappa is undefined, not perfect agreement. Independent rater identity is an organizational requirement.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',required=True);p.add_argument('--ratings',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=summarize(read_jsonl(Path(args.cases)),[r for path in args.ratings for r in read_jsonl(Path(path))]);write_json(Path(args.output),result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
