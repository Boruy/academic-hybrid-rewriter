"""Generate a final sentence rewrite from a local checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .lexical import LexicalFormalizer
from .prepare import numbers


def load_checkpoint(checkpoint, device):
    import torch
    from transformers import AutoTokenizer

    from .six_layer import SixLayerConfig, SixLayerRewriter

    root = Path(checkpoint)
    meta = json.loads((root/'result.json').read_text(encoding='utf8'))
    if meta['model'] != 'six_layer':
        raise ValueError('A completed six-layer checkpoint is required')

    tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True)
    tokenizer.padding_side = 'right'
    model = SixLayerRewriter(SixLayerConfig.load(root/'six_layer_config.json'))
    if model.config.span_edits and len(tokenizer) != model.config.text_vocab_size:
        raise ValueError('Checkpoint and tokenizer text vocabularies differ')
    model.load_state_dict(torch.load(root/'weights.pt', map_location='cpu', weights_only=True))
    model.to(device).eval()
    return model, tokenizer, meta


def generate_predictions(model, tokenizer, sources, inputs, max_new_tokens, *, depth=None):
    from .span_edits import apply_edit_program, word_units

    edit_mode = model.config.span_edits
    counts = [len(word_units(source)) for source in sources] if edit_mode else None
    sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                               max_new_tokens=max_new_tokens, depth=depth, word_guidance=inputs.get('word_guidance'),
                               edit_unit_counts=counts)
    if not edit_mode:
        predictions = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [(dict(prediction=text), tokens) for text, tokens in zip(predictions, sequences.tolist(), strict=True)]

    results = []
    for source, tokens in zip(sources, sequences.tolist(), strict=True):
        try:
            prediction, edits = apply_edit_program(source, tokens[1:], tokenizer, model.config)
            result = dict(prediction=prediction, edits=[asdict(edit) for edit in edits], edit_error=None)
        except ValueError as error:
            result = dict(prediction=source, edits=[], edit_error=str(error))
        results.append((result, tokens))
    return results


def rewrite_neural(text, depth, checkpoint, device=None, max_new_tokens=None):
    import torch

    from .joint_words import WordTask
    from .six_layer import NEURAL_DEPTHS, WORD_LAYERS

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, tokenizer, meta = load_checkpoint(checkpoint, device)
    config = model.config
    depth = model.sentence_depths[-1] if depth is None else depth
    if config.fuse_stage_outputs and depth != 'deep':
        raise ValueError('This checkpoint generates only the fused final rewrite. Omit --depth.')
    word_task = WordTask(model, tokenizer, device, meta['max_length']) if model.word_head is not None else None
    if depth == 'shallow':
        if word_task is None:
            raise ValueError('Shallow mode requires a joint checkpoint or a standalone word model')
        prediction, edits = word_task.rewrite(text)
        return dict(source=text, prediction=prediction, depth=depth, checkpoint=str(checkpoint),
                    active_sequence_layers=WORD_LAYERS, word_edits=[asdict(edit) for edit in edits])
    if depth not in model.sentence_depths:
        raise ValueError(f'{depth} generation is unavailable for a {config.depth} checkpoint')
    inputs = tokenizer(text, return_tensors='pt', return_offsets_mapping=word_task is not None)
    source_length = inputs['input_ids'].size(1)
    if source_length > meta['max_length']:
        raise ValueError(f'Source uses {source_length} tokens and exceeds the checkpoint input limit of {meta["max_length"]}')
    remaining = config.max_positions - source_length
    default_budget = meta['max_length'] + (2 if config.span_edits else 0)
    budget = min(default_budget, remaining) if max_new_tokens is None else max_new_tokens
    if not 1 <= budget <= remaining:
        raise ValueError(f'Generation budget must be between 1 and {remaining} tokens for this input')

    with torch.no_grad():
        model_inputs = dict(input_ids=inputs['input_ids'].to(device),
                            attention_mask=inputs['attention_mask'].to(device),
                            word_guidance=word_task.guidance([text], inputs) if word_task else None)
        (generated, tokens), = generate_predictions(model, tokenizer, [text], model_inputs, budget, depth=depth)
    candidate = generated['prediction'] if config.span_edits else generated['prediction'].strip()
    if not candidate:
        fallback = 'empty_candidate'
    elif config.eos_token_id not in tokens[1:]:
        fallback = 'no_end_token'
    elif generated.get('edit_error'):
        fallback = 'invalid_edit_program'
    elif numbers(text) != numbers(candidate):
        fallback = 'changed_numbers'
    else:
        fallback = None
    result = dict(source=text, prediction=text if fallback else candidate,
                  depth='final' if config.fuse_stage_outputs else depth,
                  active_sequence_layers=NEURAL_DEPTHS[depth], checkpoint=str(checkpoint),
                  fallback=fallback)
    if fallback and not config.fuse_stage_outputs:
        result['candidate'] = candidate
    if config.span_edits:
        result.update(edits=[] if fallback else generated['edits'], edit_error=generated['edit_error'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', choices=['shallow', 'medium', 'deep'],
                        help='Override the default output for an earlier checkpoint')
    parser.add_argument('--text', required=True)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--lexical-model', help='Legacy standalone word model JSON')
    model_group.add_argument('--checkpoint', help='Trained checkpoint directory')
    parser.add_argument('--device', help='Neural device, for example cpu or cuda')
    parser.add_argument('--max-new-tokens', type=int, help='Optional neural decoding limit')
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error('--text must contain a sentence')
    if args.lexical_model and (args.depth not in (None, 'shallow') or args.device is not None):
        parser.error('The standalone --lexical-model is only used with shallow mode on CPU')
    if (args.depth == 'shallow' or args.lexical_model) and args.max_new_tokens is not None:
        parser.error('--max-new-tokens applies to sentence generation only')

    try:
        if args.lexical_model:
            prediction, edits = LexicalFormalizer.load(args.lexical_model).rewrite(args.text)
            result = dict(source=args.text, prediction=prediction, depth='shallow',
                          word_edits=[asdict(edit) for edit in edits])
        else:
            result = rewrite_neural(args.text, args.depth, args.checkpoint,
                                    args.device, args.max_new_tokens)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
