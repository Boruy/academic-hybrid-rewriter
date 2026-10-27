"""Rebuild the English data-quality audit and next-stage readiness report."""
from collections import Counter
import difflib
import hashlib
import html
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/academic-refinement'

def read_rows(path):return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def diff(source,target):
    a=source.split();b=target.split();old=[];new=[]
    for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        x=html.escape(' '.join(a[i:j]));y=html.escape(' '.join(b[k:l]))
        if x:old.append(x if op=='equal' else '<del>'+x+'</del>')
        if y:new.append(y if op=='equal' else '<ins>'+y+'</ins>')
    return ' '.join(old),' '.join(new)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=read_rows(ROOT/'data/curation/iterater_audit_100.jsonl')
    all_rows=[dict(r,original_split=sp) for sp in ['train','validation','test'] for r in read_rows(ROOT/f'data/processed/{sp}.jsonl')]
    sampled=random.Random(42).sample(all_rows,100)
    if len(audit)!=100 or [r['id'] for r in audit]!=[r['id'] for r in sampled]:raise ValueError('Audit does not match the seeded sample')
    for row,raw in zip(audit,sampled):
        if any(row[k]!=raw[k] for k in ['source','target','doc_id','original_split']):raise ValueError('Audit source was changed')
        if row['human_verified'] or row['reviewer_type']!='ai_assisted' or row['eligible_for_training']:raise ValueError('Preliminary audit must not impersonate independent approval')
        row['source_diff'],row['target_diff']=diff(row['source'],row['target'])
    summary=json.loads((ROOT/'data/curation/arxivedits/summary.json').read_text(encoding='utf8'))
    triage=read_rows(ROOT/'data/curation/arxivedits/ai_triage.jsonl');references=read_rows(ROOT/'data/curation/arxivedits/references.jsonl')
    lookup={r['id']:r for r in references}
    for row in triage:
        ref=lookup[row['id']]
        if ref['split']=='test' or any(row[k]!=ref[k] for k in ['source_sha256','target_sha256']):raise ValueError('Triage must preserve frozen development references and exclude test')
    counts=Counter(r['task_category'] for r in audit);split_counts=Counter(r['original_split'] for r in audit)
    sources=['data/curation/iterater_audit_100.jsonl','data/curation/arxivedits/summary.json','data/curation/arxivedits/references.jsonl','data/curation/arxivedits/ai_triage.jsonl','data/curation/arxivedits/frozen_partition.json','configs/refinement_experiment.json','scripts/report_refinement.py','scripts/templates/refinement_report.html']
    result=dict(task=summary['task'],audit=dict(sample_size=100,population_size=len(all_rows),seed=42,categories=dict(counts),original_splits=dict(split_counts),status='AI-assisted preliminary review; no independent human gold labels'),arxivedits=summary,development_triage=dict(reviewed=len(triage),shortlisted=sum(r['decision']=='shortlist_for_human_review' for r in triage),human_verified=0),new_model_training_status='Not started: independently reviewed target data is not yet available.',new_quality_evidence_status='Pending; no new model predictions or successful-revision rate are claimed.',source_sha256={p:sha(ROOT/p) for p in sources},cases=audit)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#dce3df','text.color':'#152c3c','axes.labelcolor':'#152c3c','xtick.color':'#637580','ytick.color':'#152c3c','figure.facecolor':'#faf8f3','axes.facecolor':'#faf8f3','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(11,5.2));labels=['Academic Style','Grammar / Fluency','Content Change / Uncertain','No Meaningful Improvement'];colors=['#0c8b81','#4473c5','#d16c50','#8a9498']
    bars=ax.barh(range(4),[counts[k] for k in labels],color=colors,height=.55)
    ax.set_yticks(range(4),['Academic style candidate','Grammar / fluency','Content change / uncertain','No clear improvement']);ax.invert_yaxis();ax.set_xlim(0,57);ax.set_xlabel('Pairs in the 100-pair sample');ax.tick_params(axis='y',length=0,pad=12)
    for bar in bars:ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,str(int(bar.get_width())),va='center',weight='bold')
    fig.suptitle('The training signal does not yet match the narrow task',x=.04,ha='left',fontsize=18,fontweight='bold',y=.98);fig.text(.04,.90,'Seed 42 · 100 of 676 IteraTeR pairs · AI-assisted preliminary audit',fontsize=11,color='#637580');fig.text(.04,.025,'Task-specific judgments, not upstream intent labels. All 100 require independent human verification.',fontsize=9,color='#637580');fig.subplots_adjust(left=.29,right=.95,top=.81,bottom=.17)
    for extension in ['svg','png']:
        path=OUT/('data_audit.'+extension);fig.savefig(path,dpi=180)
        if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
    plt.close(fig)
    template=(ROOT/'scripts/templates/refinement_report.html').read_text(encoding='utf8')
    payload=json.dumps(result,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    document=template.replace('__REFINEMENT_DATA__',payload).replace('__AUDIT_SVG__',(OUT/'data_audit.svg').read_text(encoding='utf8').split('?>',1)[-1])
    (OUT/'index.html').write_text(document,encoding='utf8',newline='\n')
    print(f'Refinement report: audit={dict(counts)}, screened={summary["screened_candidates"]}, AI shortlist={result["development_triage"]["shortlisted"]}; human verified=0; new training=pending')

if __name__=='__main__':main()
