import urllib.request,json,hashlib,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'data/raw/iterater_human_sent'; out.mkdir(parents=True,exist_ok=True)
def fetch(url):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=60) as r: return r.read()
  except Exception:
   if attempt==2: raise
   time.sleep(2)
rev='e22e0371dac444239b944f9293f5b491d62b73f0'; manifest={'dataset':'wanyu/IteraTeR_human_sent','revision':rev,'files':{}}
for name in ['train.json','dev.json','test.json','README.md']:
 b=fetch(f'https://huggingface.co/datasets/wanyu/IteraTeR_human_sent/resolve/{rev}/{name}')
 (out/name).write_bytes(b)
 manifest['files'][name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 print(name,len(b),flush=True)
(root/'data/raw/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('REVISION',rev)
print((out/'dev.json').read_text(encoding='utf8')[:2500])
