"""Run a local checkpoint on a held-out split and save predictions and metrics."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

from .evaluate import evaluate
from .models import HybridRewriter
from .run import batch, read_rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/processed')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--output', help='Default: CHECKPOINT/evaluation/SPLIT')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='0 evaluates the entire split')
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--device', help='Default: cuda when available, otherwise cpu')
    return p


@torch.no_grad()
def run(args):
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        raise ValueError('batch-size and max-new-tokens must be positive; limit must be nonnegative')
    root = Path(args.checkpoint)
    meta = json.loads((root/'result.json').read_text(encoding='utf8'))
    rows = read_rows(Path(args.data)/(args.split+'.jsonl'))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty evaluation split')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.output) if args.output else root/'evaluation'/args.split
    out.mkdir(parents=True, exist_ok=True)
    print(f'Loading {root} on {device}; evaluating {len(rows)} {args.split} pairs', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
    tokenizer.padding_side = 'right'
    backbone = BartForConditionalGeneration(BartConfig.from_pretrained(root/'backbone_config', local_files_only=True))
    model = backbone if meta['model']=='transformer' else HybridRewriter(backbone, meta['fusion'], meta['backend'])
    model.load_state_dict(torch.load(root/'weights.pt', map_location='cpu', weights_only=True))
    model.to(device).eval()
    predictions = []
    numerator = denominator = 0
    with (out/'predictions.jsonl').open('w', encoding='utf8') as stream:
        for i in range(0, len(rows), args.batch_size):
            chosen = rows[i:i+args.batch_size]
            inputs = batch(chosen, tokenizer, device, meta['max_length'])
            tokens = (inputs['labels']!=-100).sum().item()
            numerator += model(**inputs).loss.item()*tokens
            denominator += tokens
            generated = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                       max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False)
            for row, prediction in zip(chosen, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                result = dict(id=row['id'], source=row['source'], reference=row['target'], prediction=prediction)
                predictions.append(result)
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
            stream.flush()
            print(f'[{args.split} {len(predictions)}/{len(rows)}] cross_entropy={numerator/denominator:.4f}', flush=True)
    metrics = evaluate(predictions)
    metrics.update(checkpoint=str(root.resolve()), split=args.split, device=str(device),
                   cross_entropy=numerator/denominator, max_length=meta['max_length'],
                   max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False,
                   test_set_used=args.split=='test')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f'Evaluation complete. Results: {out.resolve()}', flush=True)
    return metrics


if __name__=='__main__':
    run(parser().parse_args())
