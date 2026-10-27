"""Pin arXivEdits, freeze document partitions, and prepare private review candidates.

Raw paper text stays in ignored .cache because the upstream release has mixed
paper-specific licenses. Public manifests contain references and hashes only.
Upstream edit intentions are candidate signals, never academic-quality labels.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REVISION='113913d68522c1c8428d8e191d5c9bed633da6ce'
REPOSITORY='https://github.com/chaojiang06/arXivEdits'
SEEN_DOCUMENTS={'0806.2805','1612.06149','1301.2079','hep-ph-0405286','1507.07370','nucl-th-0306024'}
STYLE_SIGNALS={'Lang-professional-Improve-style','Lang-improve-readability-Simplify','Lang-accurate-spefific'}
PROTECTED=set('may might could can cannot not no never neither nor only all some most approximately almost possibly probably significant significantly'.split())

def digest(raw): return hashlib.sha256(raw).hexdigest()
def norm(text): return ' '.join(re.findall(r'\w+',text.casefold()))
def text_hash(text): return digest(norm(text).encode())
def read_rows(path): return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf8',newline='\n')

def preservation_flags(source,target):
    flags=[]
    if Counter(re.findall(r'\d+(?:[.,]\d+)*',source))!=Counter(re.findall(r'\d+(?:[.,]\d+)*',target)): flags.append('number_change')
    for label,pattern in [('placeholder_change',r'\[(?:MATH|CITATION|REF|EQUATION)\]'),('acronym_change',r'\b[A-Z][A-Z0-9-]+\b')]:
        if Counter(re.findall(pattern,source))!=Counter(re.findall(pattern,target)): flags.append(label)
    if Counter(w for w in norm(source).split() if w in PROTECTED)!=Counter(w for w in norm(target).split() if w in PROTECTED): flags.append('hedge_negation_quantifier_change')
    if '\\' in source+target or '%DIF' in source+target: flags.append('extraction_artifact')
    if norm(source)==norm(target): flags.append('no_word_change')
    return flags

def prepare(download=False):
    cache=ROOT/'.cache/arxivedits'; cache.mkdir(parents=True,exist_ok=True)
    files={}; papers={}; edits=[]
    for split in ['train','dev','test']:
        for kind in ['edits','sentence_alignment']:
            name=split+('_alignment' if kind=='sentence_alignment' else '')+'.json'; path=cache/name
            relative=f'data/{kind}/{split}.json'
            if not path.exists():
                if not download: raise ValueError(f'Missing {path}; rerun with --download')
                path.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/chaojiang06/arXivEdits/{REVISION}/{relative}',timeout=180).read())
            raw=path.read_bytes(); files[relative]={'sha256':digest(raw),'bytes':len(raw)}; content=json.loads(raw)
            if kind=='sentence_alignment':
                for doc_id,paper in content.items():
                    if doc_id in papers: raise ValueError('Upstream alignment document overlap')
                    papers[doc_id]=paper
            else:
                edits.extend((split,key,row) for key,row in content.items())
    old=[r for split in ['train','validation','test'] for r in read_rows(ROOT/f'data/processed/{split}.jsonl')]
    old_docs={r['doc_id'] for r in old}; old_text={text_hash(r[k]) for r in old for k in ['source','target']}
    # Cross-document exact-text bridges are quarantined rather than moved across
    # upstream partitions. Assignment happens before task labels or model outputs.
    text_partitions=defaultdict(set); doc_partitions=defaultdict(set)
    for split,key,row in edits:
        doc_partitions[row['arxiv-id']].add(split)
        for k in ['sentence-1','sentence-2']: text_partitions[text_hash(row[k])].add(split)
    references=[]; candidates=[]; seen_pairs=set()
    for upstream_split,key,row in edits:
        source,target=row['sentence-1'],row['sentence-2']; doc_id=row['arxiv-id']; paper=papers.get(doc_id,{})
        intents=sorted({edit['intention'] for k,v in row.items() if k.startswith('edits-combination') for edit in v.values() if edit.get('intention')})
        reasons=[]
        if doc_id in old_docs or any(text_hash(text) in old_text for text in [source,target]): reasons.append('overlap_with_previous_experiments')
        if doc_id in SEEN_DOCUMENTS: reasons.append('text_inspected_before_freeze')
        if len(doc_partitions[doc_id])>1 or any(len(text_partitions[text_hash(text)])>1 for text in [source,target]): reasons.append('cross_partition_document_or_text')
        partition={'train':'train','dev':'validation','test':'test'}[upstream_split]
        pair=(text_hash(source),text_hash(target))
        if pair in seen_pairs: reasons.append('duplicate_pair')
        seen_pairs.add(pair)
        flags=preservation_flags(source,target)
        proposed=bool(STYLE_SIGNALS.intersection(intents)) and 'Content' not in intents
        identifier=f'arxivedits-{upstream_split}-{key}'
        licenses={str(level):paper.get('license',{}).get(str(level)) for level in [row['sentence-1-level'],row['sentence-2-level']]}
        reference=dict(id=identifier,doc_id=doc_id,split=partition,upstream_split=upstream_split,upstream_key=key,upstream_intents=intents,source_sha256=digest(source.encode()),target_sha256=digest(target.encode()),normalized_source_sha256=text_hash(source),normalized_target_sha256=text_hash(target),versions=[row['sentence-1-level'],row['sentence-2-level']],paper_url=paper.get('arxiv_link'),licenses=licenses,exclusion_reasons=reasons,preservation_flags=flags,task_candidate=proposed and not reasons and not flags,review_status='unreviewed',human_verified=False)
        references.append(reference)
        if reference['task_candidate']:
            candidates.append(dict(reference,source=source,target=target,group_id=doc_id,synthetic=False))
    public=ROOT/'data/curation/arxivedits'
    frozen={'revision':REVISION,'file_sha256':files,'partition_rule':'Preserve upstream document-disjoint train/dev/test; dev becomes validation. Quarantine previous overlap, inspected documents and cross-partition exact-text bridges.','documents':{doc:({'train':'train','dev':'validation','test':'test'}[next(iter(splits))] if len(splits)==1 else 'quarantine') for doc,splits in sorted(doc_partitions.items())},'previous_data_sha256':{sp:digest((ROOT/f'data/processed/{sp}.jsonl').read_bytes()) for sp in ['train','validation','test']},'previously_inspected_documents':sorted(SEEN_DOCUMENTS)}
    frozen_path=public/'frozen_partition.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf8'))!=frozen: raise ValueError('Frozen partition/provenance changed; create a separately versioned study')
    write_json(frozen_path,frozen); write_rows(public/'references.jsonl',references)
    for split in ['train','validation','test']:
        write_rows(ROOT/f'.cache/academic-refinement/candidates/{split}.jsonl',[r for r in candidates if r['split']==split])
    summary={'task':'Academic Style Enhancement with Meaning Preservation','repository':REPOSITORY,'revision':REVISION,'upstream_pairs':len(edits),'upstream_documents':len(doc_partitions),'previous_document_overlap':len(set(doc_partitions)&old_docs),'intention_only_candidates':sum(bool(STYLE_SIGNALS.intersection(r['upstream_intents'])) and 'Content' not in r['upstream_intents'] for r in references),'screened_candidates':len(candidates),'candidates_by_partition':dict(Counter(r['split'] for r in candidates)),'human_verified_pairs':0,'target_reviewed_pairs':[200,500],'test_status':'Frozen and quarantined; no model evaluation or public text preview. Human curation is still required.','text_storage':'Ignored local cache only. Mixed version-specific paper licenses; references are not a blanket redistribution license.','filter_note':'Numeric/hedge/acronym/placeholder guards are conservative screens, not semantic verification.'}
    write_json(public/'summary.json',summary); print(json.dumps(summary,indent=2))
    return references,candidates

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');args=p.parse_args();prepare(args.download)
