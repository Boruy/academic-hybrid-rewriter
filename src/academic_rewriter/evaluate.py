"""Evaluate aligned JSONL predictions. No model downloads required."""
import argparse,collections,json,re
from pathlib import Path
from .prepare import numbers

def lcs(a,b):
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b): current.append(previous[j]+1 if x==y else max(previous[j+1],current[-1]))
        previous=current
    return previous[-1]
def evaluate(rows):
    if not rows: raise ValueError('No predictions')
    scores=[]
    for r in rows:
        p=r['prediction'].lower().split(); t=r['reference'].lower().split(); common=lcs(p,t)
        precision=common/len(p) if p else 0; recall=common/len(t) if t else 0
        scores.append(2*precision*recall/(precision+recall) if precision+recall else 0)
    out={'n':len(rows),'whitespace_rouge_l_f1':sum(scores)/len(scores),'number_multiset_preservation_rate':sum(numbers(r['source'])==numbers(r['prediction']) for r in rows)/len(rows),'copy_rate':sum(r['source'].strip()==r['prediction'].strip() for r in rows)/len(rows),'warning':'ROUGE is a whitespace diagnostic. Number preservation is not semantic equivalence. Use independent human labels for academic tone.'}
    try:
        from sacrebleu import corpus_bleu
        metric=corpus_bleu([r['prediction'] for r in rows],[[r['reference'] for r in rows]])
        out['sacrebleu']=metric.score
    except ImportError: out['sacrebleu']=None
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('predictions');a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.predictions).read_text(encoding='utf8').splitlines() if line.strip()]
    print(json.dumps(evaluate(rows),indent=2))
