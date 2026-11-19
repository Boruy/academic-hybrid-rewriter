"""Rewrite one sentence with the word module or a depth-specific neural checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .lexical import LexicalFormalizer
from .prepare import numbers


def rewrite_neural(text, depth, checkpoint, device=None, max_new_tokens=None):
    import torch
    from transformers import AutoTokenizer

    from .six_layer import SixLayerConfig, SixLayerRewriter

    root = Path(checkpoint)
    config = SixLayerConfig.load(root/'six_layer_config.json')
    if config.depth != depth:
        raise ValueError(f'Use a {depth} checkpoint. This checkpoint uses {config.depth} depth.')
    meta = json.loads((root/'result.json').read_text(encoding='utf8'))
    if meta['model'] != 'six_layer':
        raise ValueError('A completed six-layer checkpoint is required')

    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
    inputs = tokenizer(text, return_tensors='pt')
    source_length = inputs['input_ids'].size(1)
    if source_length > meta['max_length']:
        raise ValueError(f'Source uses {source_length} tokens and exceeds the checkpoint input limit of {meta["max_length"]}')
    remaining = config.max_positions - source_length
    budget = min(meta['max_length'], remaining) if max_new_tokens is None else max_new_tokens
    if not 1 <= budget <= remaining:
        raise ValueError(f'Generation budget must be between 1 and {remaining} tokens for this input')

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = SixLayerRewriter(config)
    model.load_state_dict(torch.load(root/'weights.pt', map_location='cpu', weights_only=True))
    model.to(device).eval()
    generated = model.generate(input_ids=inputs['input_ids'].to(device),
                               attention_mask=inputs['attention_mask'].to(device), max_new_tokens=budget)[0]
    tokens = generated.tolist()[1:]  # Skip the initial generation marker.
    candidate = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if not candidate:
        fallback = 'empty_candidate'
    elif config.eos_token_id not in tokens:
        fallback = 'no_end_token'
    elif numbers(text) != numbers(candidate):
        fallback = 'changed_numbers'
    else:
        fallback = None
    result = dict(source=text, prediction=text if fallback else candidate, depth=depth,
                  active_sequence_layers=model.active_layer_count, checkpoint=str(root),
                  fallback=fallback)
    if fallback:
        result['candidate'] = candidate
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', choices=['shallow', 'medium', 'deep'], required=True)
    parser.add_argument('--text', required=True)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--lexical-model', help='Word model JSON for shallow rewriting')
    model_group.add_argument('--checkpoint', help='Trained medium or deep checkpoint directory')
    parser.add_argument('--device', help='Neural device, for example cpu or cuda')
    parser.add_argument('--max-new-tokens', type=int, help='Optional neural decoding limit')
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error('--text must contain a sentence')
    if args.depth == 'shallow':
        if not args.lexical_model or args.device is not None or args.max_new_tokens is not None:
            parser.error('Shallow rewriting requires --lexical-model and takes no neural options')
    elif not args.checkpoint:
        parser.error('Medium and deep rewriting require --checkpoint')

    try:
        if args.depth == 'shallow':
            prediction, edits = LexicalFormalizer.load(args.lexical_model).rewrite(args.text)
            result = dict(source=args.text, prediction=prediction, depth=args.depth,
                          word_edits=[asdict(edit) for edit in edits])
        else:
            result = rewrite_neural(args.text, args.depth, args.checkpoint,
                                    args.device, args.max_new_tokens)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
