"""Prepare auditable, document-disjoint academic revision data. Standard library only."""
import argparse, collections, hashlib, json, random, re, unicodedata
from pathlib import Path

def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC', s).split())

def numbers(s):
    return sorted(re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|e[+-]?\d+)?', s.lower()))

class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b): self.parent[self.find(a)] = self.find(b)

def prepare(raw_dir, output, seed=42):
    raw_dir, output = Path(raw_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, raw_counts, rejection = [], {}, collections.Counter()
    for split in ('train', 'dev', 'test'):
        content = (raw_dir / (split + '.json')).read_text(encoding='utf8')
        items = [json.loads(line) for line in content.splitlines() if line.strip()]
        raw_counts[split] = len(items)
        for i, r in enumerate(items):
            r.update(upstream_split=split, upstream_line=i+1)
            rows.append(r)
    kept, seen = [], set()
    for r in rows:
        doc = str(r['doc_id'])
        if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?', doc):
            rejection['non_arxiv_document'] += 1; continue
        intent = r['labels'].strip().lower()
        if intent not in {'style','clarity','fluency'}:
            rejection['excluded_intent'] += 1; continue
        s,t = normalize(r['before_sent']),normalize(r['after_sent'])
        if not s or not t or s.casefold() == t.casefold():
            rejection['empty_or_identity'] += 1; continue
        if not (5 <= len(s.split()) <= 100 and 5 <= len(t.split()) <= 100):
            rejection['word_length'] += 1; continue
        if not 0.5 <= len(t.split()) / len(s.split()) <= 2:
            rejection['length_ratio'] += 1; continue
        if numbers(s) != numbers(t):
            rejection['number_change'] += 1; continue
        pair = (s.casefold(), t.casefold())
        if pair in seen: rejection['duplicate_pair'] += 1; continue
        seen.add(pair)
        kept.append(dict(id=hashlib.sha256(('\0'.join(pair)).encode()).hexdigest()[:16],
                         source=s,target=t,intent=intent,doc_id=re.sub(r'v\d+$','',doc),
                         upstream_split=r['upstream_split'],upstream_line=r['upstream_line']))
    uf, text_owner = UnionFind(), {}
    for r in kept:
        uf.find(r['doc_id'])
        for text in (r['source'],r['target']):
            key=text.casefold()
            if key in text_owner: uf.union(r['doc_id'],text_owner[key])
            else: text_owner[key]=r['doc_id']
    groups=collections.defaultdict(list)
    for r in kept: groups[uf.find(r['doc_id'])].append(r)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    if len(keys)<3: raise ValueError('At least 3 independent document components are required')
    n_train=max(1,int(len(keys)*.8)); n_val=max(1,int(len(keys)*.1))
    if n_train+n_val>=len(keys): n_train=len(keys)-n_val-1
    assignments={k:('train' if i<n_train else 'validation' if i<n_train+n_val else 'test') for i,k in enumerate(keys)}
    splits={k:[] for k in ('train','validation','test')}
    for k, records in groups.items():
        for r in records: r['group_id']=k; splits[assignments[k]].append(r)
    for name, records in splits.items():
        records.sort(key=lambda r:r['id'])
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
    leakage={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        ds=lambda name:{r['doc_id'] for r in splits[name]}
        ts=lambda name:{t.casefold() for r in splits[name] for t in (r['source'],r['target'])}
        leakage[a+'__'+b]={'shared_docs':len(ds(a)&ds(b)),'shared_exact_texts':len(ts(a)&ts(b))}
        assert leakage[a+'__'+b] == {'shared_docs':0,'shared_exact_texts':0}
    upstream_docs={s:{str(r['doc_id']) for r in rows if r['upstream_split']==s} for s in raw_counts}
    stats={'seed':seed,'raw_counts':raw_counts,'raw_total':len(rows),'kept_total':len(kept),
           'rejections':dict(rejection),'components':len(groups),'split_policy':'80/10/10 of connected document groups, not sentence counts',
           'splits':{name:{'pairs':len(rs),'documents':len({r['doc_id'] for r in rs}),'intents':dict(collections.Counter(r['intent'] for r in rs))} for name,rs in splits.items()},
           'leakage':leakage,'upstream_shared_documents':{a+'__'+b:len(upstream_docs[a]&upstream_docs[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]},
           'limitation':'Exact text checks do not detect semantic near duplicates. Intent labels do not guarantee academic tone or factual equivalence.'}
    (output/'stats.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    examples=sorted(splits['train'],key=lambda r:(r['intent']!='style',len(r['source'])))[:5]
    (output/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(stats,indent=2)); return stats

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',default='data/raw/iterater_human_sent'); p.add_argument('--output',default='data/processed'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); prepare(a.raw_dir,a.output,a.seed)
