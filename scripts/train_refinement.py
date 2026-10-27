"""Run the four matched arms once independently curated target data exists."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def validate_data(data):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='ready_for_training': raise ValueError('Target data is not approved; complete independent curation first')
    documents={};texts={}
    old=[json.loads(line) for sp in ['train','validation','test'] for line in (ROOT/f'data/processed/{sp}.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    old_documents={r['doc_id'] for r in old}
    for split in ['train','validation','test']:
        path=data/(split+'.jsonl');raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['data_sha256'][split]: raise ValueError('Approved data fingerprint changed')
        for row in [json.loads(line) for line in raw.decode('utf8').splitlines() if line.strip()]:
            if not row.get('human_verified') or len(set(row.get('curation_raters',[])))<2: raise ValueError('Missing independent data approval')
            if row['split']!=split or row.get('synthetic') and split!='train': raise ValueError('Split/synthetic provenance violation')
            doc=row['doc_id']
            if doc in old_documents: raise ValueError('New target data overlaps a previous experiment document')
            if doc in documents and documents[doc]!=split: raise ValueError('Cross-split document leakage')
            documents[doc]=split
            for key in ['source','target']:
                text=' '.join(__import__('re').findall(r'\w+',row[key].casefold()))
                if text in texts and texts[text]!=split: raise ValueError('Cross-split exact-text leakage')
                texts[text]=split
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/refinement_experiment.json');p.add_argument('--print-commands',action='store_true');args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf8'));data=ROOT/config['data']
    if not args.print_commands: validate_data(data)
    for arm in config['arms']:
        settings=dict(config['common'],model=arm['model'],data=config['data'],output=config['output']+'/'+arm['name'])
        if 'init_checkpoint' in arm:settings['init_checkpoint']=arm['init_checkpoint']
        command=[sys.executable,'-m','academic_rewriter.run']
        for key,value in settings.items():
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value: command.append(flag)
            else: command.extend([flag,str(value)])
        print(subprocess.list2cmdline(command),flush=True)
        if not args.print_commands: subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__': main()
