"""Generate a final sentence rewrite from a local checkpoint."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re

from .lexical import LexicalFormalizer
from .prepare import numbers


_NUMBER_WORDS = frozenset(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
    'fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy '
    'eighty ninety hundred thousand million billion trillion dozen'.split()
)
_WORDS = re.compile(r'\w+')
_IDENTIFIERS = re.compile(r'\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b')
_NUMERIC_RELATIONS = re.compile(
    r'\b(?:up\s*to|at\s+(?:least|most)|(?:(?:times|fold)\s+)?'
    r'(?:faster|slower|higher|lower|greater|smaller|more|less|better|worse)\s+than)\b'
)


def span_fact_error(source, candidate):
    """Return the first protected expression changed by a span rewrite."""
    if source == candidate:
        return None
    source_numbers, candidate_numbers = numbers(source), numbers(candidate)
    if source_numbers != candidate_numbers:
        return 'changed_numbers'
    source_counts, candidate_counts = (
        Counter(word for word in _WORDS.findall(value.casefold()) if word in _NUMBER_WORDS)
        for value in (source, candidate)
    )
    if source_counts != candidate_counts:
        return 'changed_number_words'
    if source_numbers or source_counts:
        source_relations, candidate_relations = (
            Counter(''.join(phrase.split()) for phrase in _NUMERIC_RELATIONS.findall(value.casefold()))
            for value in (source, candidate)
        )
        if source_relations != candidate_relations:
            return 'changed_numeric_comparison'

    source_terms, candidate_terms = _IDENTIFIERS.findall(source), _IDENTIFIERS.findall(candidate)
    protected = {term.casefold() for term in source_terms + candidate_terms
                 if sum(char.isupper() for char in term) >= 2 or any(char.isdigit() for char in term)}
    original = Counter(term.casefold() for term in source_terms)
    revised = Counter(term.casefold() for term in candidate_terms)
    if any(original[term] != revised[term] for term in protected):
        return 'changed_technical_identifier'
    return None


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
        except ValueError as error:
            result = dict(prediction=source, edits=[], edit_error=str(error), fact_error=None)
        else:
            result = dict(prediction=prediction, edits=[asdict(edit) for edit in edits],
                          edit_error=None, fact_error=span_fact_error(source, prediction))
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
    elif config.span_edits:
        fallback = generated['fact_error']
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
        result.update(edits=[] if fallback else generated['edits'], edit_error=generated['edit_error'],
                      fact_error=generated['fact_error'])
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
