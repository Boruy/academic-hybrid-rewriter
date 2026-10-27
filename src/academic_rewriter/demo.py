"""Local checkpoint inference with transparent warnings and author review."""
import argparse,difflib,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,BartConfig,BartForConditionalGeneration
from .models import HybridRewriter
from .detect import detect
from .prepare import numbers

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--text',required=True);p.add_argument('--force',action='store_true');p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
    if a.max_new_tokens<1: p.error('--max-new-tokens must be positive')
    root=Path(a.checkpoint);meta=json.loads((root/'result.json').read_text(encoding='utf8'))
    tok=AutoTokenizer.from_pretrained(root/'tokenizer');base=BartForConditionalGeneration(BartConfig.from_pretrained(root/'backbone_config'))
    model=base if meta['model']=='transformer' else HybridRewriter(base,meta['fusion'],meta['backend'])
    model.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True)); model.eval()
    detection=detect(a.text)
    if detection['revise'] or a.force:
        b=tok(a.text,return_tensors='pt',truncation=True,max_length=meta['max_length']); b={k:v for k,v in b.items() if k in {'input_ids','attention_mask'}}
        with torch.no_grad(): suggestion=tok.decode(model.generate(**b,max_new_tokens=a.max_new_tokens,num_beams=1,do_sample=False,early_stopping=False)[0],skip_special_tokens=True)
    else: suggestion=a.text
    print(json.dumps({'detection':detection,'source':a.text,'suggestion':suggestion,'numbers_preserved':numbers(a.text)==numbers(suggestion),'review_required':True,'max_new_tokens':a.max_new_tokens,'checkpoint_type':meta['experiment_type'],'warning':'Tiny smoke checkpoints are unsuitable for academic rewriting. Inspect facts, numbers, hedging and citations before accepting any output.'},indent=2))
if __name__=='__main__':main()
