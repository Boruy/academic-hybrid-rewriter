"""Review and materialize meaning-preserving academic-refinement data.

Candidate text is local-only. Two human decisions are required to admit a pair;
AI triage, upstream intent labels and automatic guards cannot approve a pair.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .evaluation_common import read_jsonl, write_json

ROOT=Path(__file__).resolve().parents[2]
CATEGORIES=['Academic Style','Grammar / Fluency','Content Change','No Meaningful Improvement']
FIELDS=['category','meaning','fluency','source_fluency','source_suitable','source_needs_revision','target_suitable','major_fact_error']

def checksum(text): return hashlib.sha256(text.encode()).hexdigest()
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8',newline='\n')

def load_candidates():
    rows=[r for sp in ['train','validation','test'] for r in read_jsonl(ROOT/f'.cache/academic-refinement/candidates/{sp}.jsonl')]
    refs={r['id']:r for r in read_jsonl(ROOT/'data/curation/arxivedits/references.jsonl')}
    for row in rows:
        ref=refs[row['id']]
        if not ref['task_candidate']: raise ValueError('Not an eligible upstream candidate')
        for key in ['doc_id','split','source_sha256','target_sha256']:
            if row[key]!=ref[key]: raise ValueError('Candidate does not match frozen reference metadata')
        if any(checksum(row[k])!=ref[k+'_sha256'] for k in ['source','target']): raise ValueError('Candidate text differs from frozen source')
    return rows

def complete_review(row):
    if any(row.get(field) is None for field in FIELDS): return False
    if row['category'] not in CATEGORIES: raise ValueError('Unknown task category')
    for field in ['meaning','fluency','source_fluency']:
        if type(row[field]) is not int or not 1<=row[field]<=5: raise ValueError('Meaning and fluency must be integers 1..5')
    for field in ['source_needs_revision','source_suitable','target_suitable','major_fact_error']:
        if type(row[field]) is not bool: raise ValueError('Review decisions must be explicit booleans')
    if row.get('reviewer_type')!='human': raise ValueError('Only actual human reviews can approve target data')
    if not isinstance(row.get('rater_id'),str) or not row['rater_id'].strip(): raise ValueError('A reviewer identity is required')
    return True

def approve_pair(reviews):
    return len(reviews)>=2 and len({r['rater_id'] for r in reviews})==len(reviews) and all(complete_review(r) and r['category']=='Academic Style' and r['meaning']>=4 and r['fluency']>=4 and r['source_needs_revision'] and r['target_suitable'] and not r['major_fact_error'] for r in reviews)

def approve_no_edit(reviews):
    return len(reviews)>=2 and len({r['rater_id'] for r in reviews})==len(reviews) and all(complete_review(r) and not r['source_needs_revision'] and r['source_suitable'] and r['source_fluency']>=4 for r in reviews)

def validate_augmentation(row,parent):
    if parent['split']!='train' or row.get('split')!='train': raise ValueError('Synthetic pairs may only derive from training documents and enter train')
    if row.get('target')!=parent['target'] or row.get('doc_id')!=parent['doc_id']: raise ValueError('Synthetic target/document must match the approved parent')
    if row['source'].strip()==row['target'].strip(): raise ValueError('Augmentation must change the source')
    # Import the exact same guards used when screening real revision pairs.
    import importlib.util
    spec=importlib.util.spec_from_file_location('refinement_preparation',ROOT/'scripts/prepare_refinement.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.preservation_flags(row['source'],row['target']): raise ValueError('Synthetic pair fails preservation guards')
    terms=row.get('protected_terms')
    if not isinstance(terms,list) or any(not isinstance(term,str) or not term.strip() for term in terms): raise ValueError('Explicit protected technical terms are required (an empty list is allowed if none apply)')
    for term in terms:
        if term.casefold() not in row['target'].casefold(): raise ValueError('Protected technical term must occur in the target')
        if row['source'].casefold().count(term.casefold())!=row['target'].casefold().count(term.casefold()): raise ValueError('Synthetic pair changes a protected technical term')

def export(args):
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new review-packet directory')
    rows=load_candidates()
    # Source content is not opened by this exporter. Test packets go only to the
    # independent data curators; development review must not browse those files.
    for split in ['train','validation','test']:
        chosen=[r for r in rows if r['split']==split]
        for number in [1,2]:
            packet=[dict(id=r['id'],doc_id=r['doc_id'],split=split,source=r['source'],target=r['target'],rater_id=f'curator_{number}',reviewer_type='human',source_sha256=r['source_sha256'],target_sha256=r['target_sha256'],**{k:None for k in FIELDS},comment='') for r in chosen]
            write_rows(out/split/f'curator_{number}.jsonl',packet)
            payload=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
            template=(ROOT/'scripts/templates/curation_review.html').read_text(encoding='utf8')
            (out/split/f'curator_{number}.html').write_text(template.replace('__REVIEW_DATA__',payload),encoding='utf8')
    write_json(out/'study.json',dict(status='unfilled',purpose='Independent data curation, not model evaluation',required_curators=2,source='Frozen arXivEdits candidates',test_policy='Test packets are for independent data curators only. Do not use them for model/threshold selection.'))
    print(f'Created blank train/validation/test packets at {out}; no approvals were imputed.')

def build(args):
    candidates={r['id']:r for r in load_candidates()}; decisions=defaultdict(list); seen=set()
    for path in args.reviews:
        for row in read_jsonl(ROOT/path):
            if row['id'] not in candidates: raise ValueError('Unknown review item')
            candidate=candidates[row['id']]; who=row.get('rater_id'); key=(row['id'],who)
            if key in seen: raise ValueError('Duplicate reviewer/item');
            seen.add(key)
            if any(row.get(k)!=candidate[k] for k in ['source','target','source_sha256','target_sha256','doc_id','split']): raise ValueError('Review text/metadata does not match frozen candidate')
            if complete_review(row): decisions[row['id']].append(row)
    accepted=[dict(r,needs_revision=True,human_verified=True,review_status='accepted_by_independent_curators',curation_raters=sorted(d['rater_id'] for d in decisions[r['id']])) for r in candidates.values() if approve_pair(decisions[r['id']])]
    controls=[dict(r,target=r['source'],target_sha256=r['source_sha256'],curation_original_target_sha256=r['target_sha256'],needs_revision=False,human_verified=True,review_status='source_already_suitable',curation_raters=sorted(d['rater_id'] for d in decisions[r['id']])) for r in candidates.values() if approve_no_edit(decisions[r['id']])]
    by_id={r['id']:r for r in accepted}
    if args.augmentation:
        for row in read_jsonl(ROOT/args.augmentation):
            if row['id'] in candidates or row['id'] in by_id: raise ValueError('Duplicate augmentation ID')
            parent=by_id.get(row.get('parent_id'))
            if parent is None or parent.get('synthetic'): raise ValueError('Augmentation requires an approved real training parent')
            validate_augmentation(row,parent)
            reviews=row.get('independent_reviews',[])
            if any(r.get('source_sha256')!=checksum(row['source']) or r.get('target_sha256')!=checksum(row['target']) for r in reviews): raise ValueError('Synthetic review is not bound to the pair text')
            if not approve_pair(reviews): raise ValueError('Synthetic pair needs two complete independent human reviews')
            enriched=dict(row,synthetic=True,needs_revision=True,group_id=parent['group_id'],source_sha256=checksum(row['source']),target_sha256=checksum(row['target']),human_verified=True,curation_raters=sorted(r['rater_id'] for r in reviews),review_status='accepted_by_independent_curators');accepted.append(enriched);by_id[row['id']]=enriched
    splits={sp:[r for r in accepted if r['split']==sp] for sp in ['train','validation','test']}
    status='ready_for_training' if all(splits.values()) else 'pending_curation'
    summary=dict(status=status,pairs_by_partition={sp:len(rows) for sp,rows in splits.items()},already_suitable_controls_by_partition={sp:sum(r['split']==sp for r in controls) for sp in splits},synthetic_pairs=sum(bool(r.get('synthetic')) for r in accepted),reviewed_target_goal=[200,500],minimum_independent_curators=2,review_files_sha256={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in args.reviews},note='Agreement in files does not independently establish rater identity. No model-quality conclusion follows from data approval. Already-suitable controls are separate from positive refinement pairs.')
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new dataset output directory')
    out.mkdir(parents=True)
    for sp,rows in splits.items(): write_rows(out/(sp+'.jsonl'),rows)
    for sp in splits: write_rows(out/(sp+'_already_suitable.jsonl'),[r for r in controls if r['split']==sp])
    summary['data_sha256']={sp:hashlib.sha256((out/(sp+'.jsonl')).read_bytes()).hexdigest() for sp in splits}
    summary['control_sha256']={sp:hashlib.sha256((out/(sp+'_already_suitable.jsonl')).read_bytes()).hexdigest() for sp in splits}
    write_json(out/'manifest.json',summary);print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--output',default='.cache/academic-refinement/review-v2');e.set_defaults(run=export)
    b=sub.add_parser('build');b.add_argument('--reviews',nargs='+',required=True);b.add_argument('--augmentation');b.add_argument('--output',default='.cache/academic-refinement/approved-v1');b.set_defaults(run=build)
    args=p.parse_args();args.run(args)

if __name__=='__main__': main()
