"""Fetch only pinned public metric models into the ignored project cache."""
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi,snapshot_download
root=Path(__file__).resolve().parents[1]
config=json.loads((root/'configs/evaluation.json').read_text(encoding='utf8'))
for key in ['bertscore','nli']:
    spec=config[key]
    if not spec.get('revision') or len(spec['revision'])!=40: raise ValueError('Pin an exact revision first')
    info=HfApi().model_info(spec['model_id'],revision=spec['revision'])
    weights='model.safetensors' if any(s.rfilename=='model.safetensors' for s in info.siblings) else 'pytorch_model.bin'
    print(f"Fetching {spec['model_id']}@{spec['revision']}",flush=True)
    snapshot_download(spec['model_id'],revision=spec['revision'],cache_dir=str(root/'.cache/evaluation-models'),
        allow_patterns=[weights,'config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','spm.model','added_tokens.json','README.md','LICENSE'],max_workers=2)
print('Pinned metric models are ready for local-only inference.',flush=True)
