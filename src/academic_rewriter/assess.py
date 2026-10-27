"""Multidimensional revision diagnostics. Neural scoring is local and explicitly enabled.

python -m academic_rewriter.assess --neural
python -m academic_rewriter.assess --output output/evaluation-lexical
"""
import argparse
import gc
import importlib.metadata
import math
import re
import statistics
from pathlib import Path

from .evaluation_common import (read_json, write_json, fingerprint, load_aligned, words,
                                 edit_distance, clustered_interval, paired_difference)
from .evaluate import lcs
from .prepare import numbers

PROJECT=Path(__file__).resolve().parents[2]


def resolve_metric_model(spec):
    from huggingface_hub import snapshot_download
    if not re.fullmatch(r'[0-9a-f]{40}',spec.get('revision') or ''):
        raise ValueError('Evaluation models require an exact 40-character commit revision')
    return Path(snapshot_download(spec['model_id'],revision=spec['revision'],
        cache_dir=str(PROJECT/'.cache/evaluation-models'),local_files_only=True))


def model_provenance(spec,path):
    files=[p for p in path.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors','.bin','.model','.txt'}]
    return dict(model_id=spec['model_id'],revision=spec['revision'],
        files={p.name:fingerprint(p) for p in files})


def lexical(source,prediction,reference):
    a,b=words(source),words(prediction)
    p,t=prediction.lower().split(),reference.lower().split()
    common=lcs(p,t)
    distance=edit_distance(a,b)
    neg=r'\b(?:not|no|never|neither|without|cannot)\b|n.t\b'
    hedge=r'\b(?:may|might|could|possibly|probably|suggests?|appears?|potentially)\b'
    citation=r'\[\d+(?:\s*[,;-]\s*\d+)*\]|\\cite\w*\{[^}]+\}'
    return dict(reference_rouge_l=2*common/(len(p)+len(t)) if p or t else 0.,
        exact_copy=float(source.strip()==prediction.strip()),same_word_sequence=float(a==b),
        word_edit_distance=distance,word_edit_rate=distance/max(1,len(a)),
        word_length_ratio=len(b)/max(1,len(a)),numbers_preserved=float(numbers(source)==numbers(prediction)),
        negation_changed=float(re.findall(neg,source.lower())!=re.findall(neg,prediction.lower())),
        hedging_changed=float(re.findall(hedge,source.lower())!=re.findall(hedge,prediction.lower())),
        citations_changed=float(re.findall(citation,source)!=re.findall(citation,prediction)))


def require_untruncated(tokenizer,texts,limit,pairs=None):
    lengths=[]
    for i,text in enumerate(texts):
        encoded=tokenizer(text, pairs[i] if pairs is not None else None, truncation=False)
        lengths.append(len(encoded['input_ids']))
    if any(n>limit for n in lengths):
        raise ValueError(f'Metric would silently truncate {sum(n>limit for n in lengths)} inputs; max {max(lengths)}, limit {limit}')
    return lengths


def nli_probabilities(logits,labels):
    """Numerically stable softmax; labels are verified against the model card/config."""
    maximum=max(logits)
    exp=[math.exp(float(x)-maximum) for x in logits]
    denominator=sum(exp)
    return {name:exp[index]/denominator for name,index in labels.items()}


def run_neural(records,config,batch_size,device):
    import torch
    from bert_score import BERTScorer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    specs=config['bertscore']
    path=resolve_metric_model(specs)
    print('BERTScore: loading independent DistilRoBERTa encoder',flush=True)
    scorer=BERTScorer(model_type=str(path),num_layers=specs['num_layers'],batch_size=batch_size,
        device=device,lang='en',idf=False,rescale_with_baseline=False,use_fast_tokenizer=specs['use_fast_tokenizer'])
    source=[r['source'] for r in records]
    output=[r['prediction'] for r in records]
    reference=[r['reference'] for r in records]
    limit=min(scorer._tokenizer.model_max_length,512)
    for texts in [source,output,reference]: require_untruncated(scorer._tokenizer,texts,limit)
    for field,refs in [('bertscore_source_f1',source),('bertscore_reference_f1',reference)]:
        _,_,f=scorer.score(output,refs,batch_size=batch_size)
        for row,value in zip(records,f.tolist()): row['metrics'][field]=value
    provenance={'bertscore':dict(**model_provenance(specs,path),num_layers=specs['num_layers'],idf=False,
        rescale_with_baseline=False,use_fast_tokenizer=specs['use_fast_tokenizer'],
        scorer_hash=scorer.hash.replace(str(path),specs['model_id']+'@'+specs['revision']),
        package_version=importlib.metadata.version('bert-score'),truncated_inputs=0)}
    del scorer
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    specs=config['nli']
    path=resolve_metric_model(specs)
    print('NLI: source -> output and output -> source',flush=True)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(device).eval()
    labels=specs['label_ids']
    if set(labels)!={'contradiction','entailment','neutral'} or set(labels.values())!={0,1,2} or model.config.num_labels!=3:
        raise ValueError('Explicit three-class NLI mapping required')
    for index,name in model.config.id2label.items():
        if name.lower() in labels and labels[name.lower()]!=int(index):
            raise ValueError('NLI label mapping conflicts with model configuration')
    require_untruncated(tok,source,specs['max_length'],pairs=output)
    require_untruncated(tok,output,specs['max_length'],pairs=source)
    for direction,premises,hypotheses in [('forward',source,output),('reverse',output,source)]:
        for start in range(0,len(records),batch_size):
            batch=tok(premises[start:start+batch_size],hypotheses[start:start+batch_size],padding=True,truncation=False,return_tensors='pt')
            with torch.no_grad(): logits=model(**{k:v.to(device) for k,v in batch.items()}).logits.cpu().tolist()
            for row,values in zip(records[start:start+batch_size],logits):
                probs=nli_probabilities(values,labels)
                row['metrics']['nli_'+direction+'_entailment']=probs['entailment']
                row['metrics']['nli_'+direction+'_contradiction']=probs['contradiction']
            print(f'NLI {direction}: {min(start+batch_size,len(records))}/{len(records)}',flush=True)
    for row in records:
        m=row['metrics']
        m['bidirectional_entailment_min']=min(m['nli_forward_entailment'],m['nli_reverse_entailment'])
        m['bidirectional_entailment_geomean']=math.sqrt(m['nli_forward_entailment']*m['nli_reverse_entailment'])
    provenance['nli']=dict(**model_provenance(specs,path),label_ids=labels,max_length=specs['max_length'],
        truncated_pairs=0,note='Generic bidirectional NLI; not the specialized published MIS model. No equivalence threshold calibrated.')
    del model
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return provenance


def score_style(records,config,batch_size,device):
    spec=config.get('style_classifier')
    if spec is None:
        return dict(status='pending_independent_labels',reason='No calibrated independent academic/non-academic classifier supplied. Revision intent labels and formality classifiers are not academic-style labels.')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root=PROJECT/spec['checkpoint']
    manifest=read_json(root/'style_manifest.json')
    # Never score with an evaluator trained on generator data.
    forbidden_docs=set()
    forbidden_texts=set()
    import hashlib
    from .evaluation_common import read_jsonl
    for split in ['train','validation','test']:
        for row in read_jsonl(PROJECT/'data/processed'/f'{split}.jsonl'):
            forbidden_docs.add(row['doc_id'])
            for key in ['source','target']:
                forbidden_texts.add(hashlib.sha256(' '.join(row[key].lower().split()).encode()).hexdigest())
    for row in records:
        forbidden_docs.add(row['doc_id'])
        for field in ['source','reference']:
            forbidden_texts.add(hashlib.sha256(' '.join(row[field].lower().split()).encode()).hexdigest())
    if forbidden_docs.intersection(manifest['annotated_document_ids']) or forbidden_texts.intersection(manifest['text_hashes']):
        raise ValueError('Style classifier annotations overlap generator data')
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(root,local_files_only=True).to(device).eval()
    temperature=manifest['temperature']
    if manifest.get('model_kind')!='academic_style_binary_classifier' or manifest.get('positive_label')!=1 or not math.isfinite(temperature) or temperature<=0 or model.config.num_labels!=2:
        raise ValueError('Invalid academic-style classifier manifest or temperature')
    for field,key in [('academic_source_probability','source'),('academic_output_probability','prediction')]:
        texts=[r[key] for r in records]
        require_untruncated(tok,texts,manifest['max_length'])
        for start in range(0,len(records),batch_size):
            batch=tok(texts[start:start+batch_size],padding=True,truncation=False,return_tensors='pt')
            with torch.no_grad(): probs=(model(**{k:v.to(device) for k,v in batch.items()}).logits/temperature).softmax(-1)[:,1].cpu().tolist()
            for row,p in zip(records[start:start+batch_size],probs): row['metrics'][field]=p
    for row in records:
        row['metrics']['academic_probability_gain']=row['metrics']['academic_output_probability']-row['metrics']['academic_source_probability']
    return dict(status='scored_pilot',manifest=manifest,note='Academic class probability is not a validated multidimensional academic-quality rating. Independent human calibration remains necessary.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/evaluation.json')
    p.add_argument('--output',default='output/multidimensional')
    p.add_argument('--neural',action='store_true',help='Require and run local pinned BERTScore and NLI models')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default=None)
    args=p.parse_args()
    if args.batch_size<1: p.error('batch-size must be positive')
    config=read_json(PROJECT/args.config)
    data,candidates,labels=load_aligned(PROJECT,config)
    records=[]
    for model,texts in candidates.items():
        for case,text in zip(data,texts):
            records.append(dict(id=case['id'],model=model,group_id=case['group_id'],doc_id=case['doc_id'],
                source=case['source'],reference=case['target'],prediction=text,
                metrics=lexical(case['source'],text,case['target'])))
    device=args.device
    if device is None:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    neural=run_neural(records,config,args.batch_size,device) if args.neural else {'status':'not_requested'}
    style=score_style(records,config,args.batch_size,device)
    groups=[r['group_id'] for r in data]
    by_model={key:[r for r in records if r['model']==key] for key in candidates}
    summaries={}
    for key,rows in by_model.items():
        fields=rows[0]['metrics'].keys()
        summaries[key]=dict(label=labels[key],n=len(rows),metrics={name:clustered_interval([r['metrics'][name] for r in rows],groups,config['bootstrap_samples'],config['seed']) for name in fields})
    from sacrebleu.metrics import BLEU
    for key,rows in by_model.items():
        metric=BLEU(tokenize='13a')
        value=metric.corpus_score([r['prediction'] for r in rows],[[r['reference'] for r in rows]])
        summaries[key]['corpus_bleu']=dict(score=value.score,signature=str(metric.get_signature()),package_version=importlib.metadata.version('sacrebleu'))
    baseline=config['models'][0]['id']
    paired={}
    for key,rows in by_model.items():
        if key==baseline: continue
        paired[key+'_minus_'+baseline]={name:paired_difference([r['metrics'][name] for r in rows],[r['metrics'][name] for r in by_model[baseline]],groups,config['bootstrap_samples'],config['seed']) for name in rows[0]['metrics']}
    source_paths=[args.config,config['data'],'src/academic_rewriter/assess.py','src/academic_rewriter/evaluation_common.py']
    source_paths += ['data/processed/'+split+'.jsonl' for split in ['train','validation','test']]
    source_paths += [e['predictions'] for e in config['models']]
    import platform
    runtime=dict(python=platform.python_version(),device=device,packages={name:importlib.metadata.version(name) for name in ['torch','transformers','tokenizers','numpy','bert-score','sacrebleu','sentencepiece']})
    summary=dict(schema_version=1,n=len(data),groups=len(set(groups)),status=config['evaluation_status'],runtime=runtime,
        dimensions=dict(semantic=neural,academic_style=style,fluency={'status':'pending_human_ratings','note':'No calibrated grammar/fluency model or external causal LM has been provided. Generator CE is not independent perplexity.'},
            human={'status':'pending_independent_raters','successful_rewrite_rate':None,'thresholds_calibrated':False}),
        models=summaries,paired_differences=paired,
        limitations=['Similarity and NLI can reward copying. Style gain and human review are separate requirements.',
            'NLI probabilities are not guarantees for negation, numbers, citations, technical terms or hedging.',
            'No automatic or human success rate is inferred from incomplete dimensions.',
            ('This test set was inspected; use fresh held-out material for confirmatory evaluation.' if config['evaluation_status']=='inspected_test_diagnostic' else 'Freshness is checked against known project splits; independent sampling and undisclosed external overlap still require a study audit.')],
        sources={path:fingerprint(PROJECT/path) for path in source_paths})
    out=PROJECT/args.output
    write_json(out/'summary.json',summary)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'cases.jsonl').open('w',encoding='utf8',newline='\n') as stream:
        import json
        for row in records: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'Wrote {len(records)} scored candidates across {len(candidates)} systems/controls to {out}',flush=True)


if __name__=='__main__':main()
