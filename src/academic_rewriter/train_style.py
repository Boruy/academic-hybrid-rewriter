"""Train an independent academic-style classifier from human-labeled JSONL.

Requires text, academic (0/1), doc_id, split (train/validation/test),
label_source='human', and guideline_version. Revision intent is not an academic label.
"""
import argparse
import hashlib
import math
import random
from pathlib import Path
from .evaluation_common import read_json,read_jsonl,write_json,fingerprint

PROJECT=Path(__file__).resolve().parents[2]


def text_hash(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def validate_annotations(rows,forbidden_docs,forbidden_hashes):
    if not rows: raise ValueError('Human academic/non-academic annotations are required')
    docs={}
    hashes={}
    for row in rows:
        if not isinstance(row.get('text'),str) or not row['text'].strip() or row.get('label_source')!='human' or not row.get('guideline_version'):
            raise ValueError('Nonempty text, human label provenance and guideline_version required')
        if type(row.get('academic')) is not int or row['academic'] not in {0,1}:
            raise ValueError('Academic label must be an explicit human 0/1, not revision intent')
        split=row.get('split')
        if split not in {'train','validation','test'} or not row.get('doc_id'):
            raise ValueError('Document ID and train/validation/test split required')
        doc=row['doc_id']; key=text_hash(row['text'])
        if doc in forbidden_docs or key in forbidden_hashes:
            raise ValueError('Classifier annotation overlaps generator train/validation/test data')
        if doc in docs and docs[doc]!=split:
            raise ValueError('Classifier document leakage across splits')
        if key in hashes:
            raise ValueError('Duplicate annotation text, possibly across splits')
        docs[doc]=split; hashes[key]=split
    partitions={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    if any({r['academic'] for r in part}!={0,1} for part in partitions.values()):
        raise ValueError('Each classifier split must contain both human academic labels')
    return partitions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',default='.cache/academic-style-classifier')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.batch_size)<1 or not math.isfinite(a.lr) or a.lr<=0:p.error('Positive finite training parameters required')
    forbidden_docs=set(); forbidden_hashes=set()
    generator_sources={}
    for split in ['train','validation','test']:
        path=PROJECT/'data/processed'/f'{split}.jsonl'
        generator_sources[str(path.relative_to(PROJECT))]=fingerprint(path)
        for row in read_jsonl(path):
            forbidden_docs.add(row['doc_id'])
            forbidden_hashes.update(text_hash(row[key]) for key in ['source','target'])
    out=PROJECT/a.output
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory to preserve previous classifiers')
    rows=read_jsonl(PROJECT/a.data)
    parts=validate_annotations(rows,forbidden_docs,forbidden_hashes)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from .assess import resolve_metric_model,require_untruncated
    torch.manual_seed(a.seed); random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    spec=read_json(PROJECT/'configs/evaluation.json')['bertscore']
    base=resolve_metric_model(spec)
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(base,local_files_only=True,num_labels=2,
        id2label={0:'non_academic',1:'academic'},label2id={'non_academic':0,'academic':1}).to(device)
    limit=512
    require_untruncated(tok,[r['text'] for r in rows],limit)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    def batch(chosen):
        x=tok([r['text'] for r in chosen],padding=True,truncation=False,return_tensors='pt')
        return {k:v.to(device) for k,v in x.items()},torch.tensor([r['academic'] for r in chosen],device=device)
    @torch.no_grad()
    def logits(part):
        model.eval(); output=[]
        for start in range(0,len(part),a.batch_size):
            x,_=batch(part[start:start+a.batch_size]);output.append(model(**x).logits.cpu())
        return torch.cat(output)
    def stats(output,part,temperature=1.):
        labels=torch.tensor([r['academic'] for r in part])
        z=output/temperature; probs=z.softmax(-1)[:,1]
        return dict(n=len(part),cross_entropy=torch.nn.functional.cross_entropy(z,labels).item(),
            accuracy=(z.argmax(-1)==labels).float().mean().item(),brier=((probs-labels)**2).mean().item(),
            majority_baseline=max(sum(r['academic']==i for r in part) for i in [0,1])/len(part))
    best=None; best_score=float('inf'); best_epoch=None; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); shuffled=parts['train'][:];random.shuffle(shuffled)
        for start in range(0,len(shuffled),a.batch_size):
            x,y=batch(shuffled[start:start+a.batch_size]);optimizer.zero_grad(set_to_none=True)
            loss=model(**x,labels=y).loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        result=stats(logits(parts['validation']),parts['validation']);history.append(dict(epoch=epoch,validation=result))
        if result['cross_entropy']<best_score:
            best_score=result['cross_entropy'];best_epoch=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'Epoch {epoch}: validation CE {result["cross_entropy"]:.4f}',flush=True)
    model.load_state_dict(best)
    val_logits=logits(parts['validation'])
    temperatures=[.5,.75,1.,1.5,2.,3.,4.]
    temperature=min(temperatures,key=lambda t:stats(val_logits,parts['validation'],t)['cross_entropy'])
    out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
    manifest=dict(model_kind='academic_style_binary_classifier',base_model=spec,positive_label=1,max_length=limit,
        seed=a.seed,lr=a.lr,best_epoch=best_epoch,temperature=temperature,
        validation=stats(val_logits,parts['validation'],temperature),test=stats(logits(parts['test']),parts['test'],temperature),
        history=history,annotated_document_ids=sorted({r['doc_id'] for r in rows}),text_hashes=sorted(text_hash(r['text']) for r in rows),
        annotation_sha256=fingerprint(PROJECT/a.data),generator_data_sha256=generator_sources,
        limitations=['Validation selects both checkpoint and temperature; small-sample calibration may overfit.',
            'Binary academic probability is not a validated rating of precision, objectivity or appropriate hedging.',
            'Use topic-diverse human labels and independent human review to test generalization.'])
    write_json(out/'style_manifest.json',manifest)
    print(f'Saved independently annotated classifier to {out}',flush=True)

if __name__=='__main__':main()
