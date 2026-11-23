"""Evaluate a local checkpoint on a full held-out split."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch
from .evaluate import evaluate
from .joint_words import WordTask
from .lexical import evaluate_lexical, read_lexical_evaluation
from .rewrite import generate_predictions, load_checkpoint
from .run import batch, read_rows, target_token_count
from .six_layer import NEURAL_DEPTHS, WORD_LAYERS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/academic-pairs-v1')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--output', help='Default: CHECKPOINT/evaluation/SPLIT-BUDGET')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='0 evaluates the entire split')
    p.add_argument('--word-test', help='Optional independently reviewed word pairs for the joint checkpoint')
    p.add_argument('--no-word-guidance', action='store_true', help='Disable word guidance for neural sentence rewrites')
    p.add_argument('--max-new-tokens', type=int, default=192, help='Maximum generated tokens per example')
    p.add_argument('--span-copy-scale', type=float, default=1., help='Scale the learned span payload copy gate from 0 to 1')
    p.add_argument('--device', help='Default: cuda when available, otherwise cpu')
    return p


def evaluation_output_path(args):
    folder = f'{args.split}-{args.max_new_tokens}'
    if args.no_word_guidance:
        folder += '-no-word-guidance'
    if args.span_copy_scale != 1.:
        folder += f'-copy-scale-{args.span_copy_scale:g}'
    return Path(args.output) if args.output else Path(args.checkpoint)/'evaluation'/folder


def generation_diagnostics(sequence, tokenizer, source, reference, max_length, max_new_tokens):
    """Record an observed cap hit, not a claim that every capped output is incomplete."""
    tokens = list(sequence)[1:]  # Exclude the model's initial BOS marker.
    while tokens and tokens[-1] == tokenizer.pad_token_id:
        tokens.pop()
    source_tokens = len(tokenizer(source)['input_ids'])
    reference_tokens = len(tokenizer(reference)['input_ids'])
    return dict(generated_tokens=len(tokens), generation_limit_reached=len(tokens) >= max_new_tokens,
                source_tokens=source_tokens, reference_tokens=reference_tokens,
                input_truncated=source_tokens > max_length, target_truncated=reference_tokens > max_length)


def sentence_metrics(predictions, depth, cross_entropy):
    metrics = evaluate(predictions)
    metrics.update(cross_entropy=cross_entropy, active_sequence_layers=NEURAL_DEPTHS[depth],
                   generation_limit_reached_count=sum(r['generation_limit_reached'] for r in predictions),
                   input_truncated_count=sum(r['input_truncated'] for r in predictions),
                   target_truncated_count=sum(r['target_truncated'] for r in predictions),
                   diagnostic_note='A cap hit means the output used the complete generation budget; '
                                   'it does not prove every such sentence is incomplete.')
    if any('fact_error' in row for row in predictions):
        metrics['fact_error_counts'] = dict(Counter(row['fact_error'] for row in predictions if row.get('fact_error')))
    if predictions and 'edit_status' in predictions[0]:
        edits = [edit for row in predictions for edit in row['edits']]
        metrics.update(
            edit_status_counts=dict(Counter(row['edit_status'] for row in predictions)),
            edit_operation_counts=dict(
                insert=sum(not edit['source'] for edit in edits),
                delete=sum(not edit['target'] for edit in edits),
                replace=sum(bool(edit['source']) and bool(edit['target']) for edit in edits),
            ),
            deletion_only_count=sum(bool(row['edits']) and all(not edit['target'] for edit in row['edits'])
                                    for row in predictions),
            unchanged_when_reference_changed_count=sum(
                row['prediction'] == row['source'] and row['reference'] != row['source'] for row in predictions
            ),
        )
    return metrics


@torch.no_grad()
def run(args):
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        raise ValueError('batch-size and max-new-tokens must be positive; limit must be nonnegative')
    root = Path(args.checkpoint)
    rows = read_rows(Path(args.data)/(args.split+'.jsonl'))
    if any(row.get('synthetic') or row.get('training_only') or row.get('split') == 'train' for row in rows):
        raise ValueError('Sentence evaluation requires a nonsynthetic held-out split')
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Empty evaluation split')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, tokenizer, meta = load_checkpoint(root, device)
    model.validate_span_copy_scale(args.span_copy_scale)
    if model.config.span_edits and args.max_new_tokens < 2:
        raise ValueError('Span edits require a generation budget of at least two tokens')
    max_length = meta['max_length']
    fused = model.config.fuse_stage_outputs
    depths = ('deep',) if fused else model.sentence_depths
    word_task = WordTask(model, tokenizer, device, max_length) if model.word_head is not None else None
    sentence_word_task = None if args.no_word_guidance else word_task
    compare_depths = word_task is not None and not fused
    word_rows = None
    if args.word_test:
        if word_task is None:
            raise ValueError('--word-test requires a joint checkpoint')
        word_rows = read_lexical_evaluation(args.word_test, meta.get('word_training_data_sha256'))
    out = evaluation_output_path(args)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Loading {root} on {device}; evaluating {len(rows)} {args.split} pairs', flush=True)
    predictions = {depth: [] for depth in depths}
    shallow_predictions = []
    numerators = dict.fromkeys(depths, 0.)
    denominators = dict.fromkeys(depths, 0)
    columns = (['id', 'source', 'reference', 'shallow_prediction', 'medium_prediction', 'deep_prediction']
               if compare_depths else ['id', 'source', 'prediction', 'reference'])
    with (
        (out/'predictions.jsonl').open('w', encoding='utf8', newline='\n') as stream,
        (out/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as csv_stream,
    ):
        comparison = csv.DictWriter(csv_stream, fieldnames=columns)
        comparison.writeheader()
        for i in range(0, len(rows), args.batch_size):
            chosen = rows[i:i+args.batch_size]
            inputs = batch(chosen, tokenizer, device, max_length, sentence_word_task, model_config=model.config)
            losses = {'deep': model(**inputs).loss} if fused else model.sequence_losses(**inputs)
            for name, loss in losses.items():
                tokens = target_token_count(inputs, name)
                numerators[name] += loss.item()*tokens
                denominators[name] += tokens
            sources = [row['source'] for row in chosen]
            generated = {}
            for depth in depths:
                generated[depth] = generate_predictions(model, tokenizer, sources, inputs, args.max_new_tokens,
                                                        depth=depth, span_copy_scale=args.span_copy_scale)
            shallow = word_task.rewrite_many(sources) if compare_depths else None
            for j, row in enumerate(chosen):
                base = dict(id=row['id'], source=row['source'], reference=row['target'])
                result = base.copy()
                for depth in depths:
                    decoded, sequence = generated[depth][j]
                    prediction = decoded['prediction']
                    diagnostics = generation_diagnostics(sequence, tokenizer, row['source'], row['target'],
                                                         max_length, args.max_new_tokens)
                    predictions[depth].append(dict(base, **decoded, **diagnostics))
                    if compare_depths:
                        result[f'{depth}_prediction'] = prediction
                        result[f'{depth}_diagnostics'] = diagnostics
                    else:
                        result.update(decoded, **diagnostics)
                if compare_depths:
                    prediction, edits = shallow[j]
                    shallow_predictions.append(dict(base, prediction=prediction, edit_count=len(edits)))
                    result.update(shallow_prediction=prediction, shallow_word_edits=[asdict(edit) for edit in edits],
                                  prediction=result['deep_prediction'])
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                comparison.writerow({field: result[field] for field in columns})
            stream.flush()
            csv_stream.flush()
            loss_by_depth = {name: numerators[name] / denominators[name] for name in depths}
            average = sum(loss_by_depth.values()) / len(loss_by_depth)
            print(f'[{args.split} {i + len(chosen)}/{len(rows)}] cross_entropy={average:.4f}', flush=True)
    by_depth = {depth: sentence_metrics(predictions[depth], depth, loss_by_depth[depth])
                for depth in depths}
    if compare_depths:
        copy = evaluate([dict(row, prediction=row['source']) for row in shallow_predictions])
        shallow_metrics = evaluate(shallow_predictions)
        shallow_metrics.update(active_sequence_layers=WORD_LAYERS,
                               edit_count=sum(row['edit_count'] for row in shallow_predictions))
        metrics = dict(n=len(rows), modes=dict(copy_baseline=copy, shallow=shallow_metrics, **by_depth),
                       prediction_stage_layers=list(model.prediction_stage_layers),
                       word_test=evaluate_lexical(word_task, word_rows) if word_rows is not None else None)
    else:
        metrics = by_depth[depths[0]]
        if fused:
            metrics.update(fuse_stage_outputs=True, candidate_guidance=model.config.candidate_guidance,
                           source_copy=model.config.source_copy, contextual_edits=model.config.contextual_edits,
                           span_edits=model.config.span_edits,
                           prediction_stage_layers=list(model.prediction_stage_layers),
                           copy_baseline=evaluate([dict(row, prediction=row['source']) for row in predictions['deep']]))
            if word_rows is not None:
                metrics['word_test'] = evaluate_lexical(word_task, word_rows)
    metrics.update(checkpoint=str(root.resolve()), split=args.split, device=str(device),
                   rewrite_depth='final' if fused else model.config.depth, active_sequence_layers=model.active_layer_count,
                   cross_entropy=average, max_length=max_length,
                   max_new_tokens=args.max_new_tokens, num_beams=1, do_sample=False,
                   word_guidance_enabled=sentence_word_task is not None,
                   test_set_used=args.split=='test')
    if model.config.span_edits:
        metrics['span_copy_scale'] = args.span_copy_scale
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf8', newline='\n')
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f'Evaluation complete. Results: {out.resolve()}', flush=True)
    return metrics


if __name__=='__main__':
    run(parser().parse_args())
