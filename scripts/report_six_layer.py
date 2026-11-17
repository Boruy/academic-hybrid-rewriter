"""Verify and render the six-layer synthetic-training study without inventing quality scores."""
import argparse
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from academic_rewriter.evaluate import evaluate
from academic_rewriter.evaluation_common import edit_distance, fingerprint, read_json, read_jsonl, words, write_json
from academic_rewriter.synthetic import validate_pair

ROOT = Path(__file__).resolve().parents[1]


def verified_diagnostics(run, split, data, budget):
    folder = run/'evaluation'/f'{split}-{budget}'
    if not (folder/'predictions.jsonl').exists():
        return None, []
    records = read_jsonl(folder/'predictions.jsonl')
    cases = read_jsonl(data/(split+'.jsonl'))
    if len(records) != len(cases) or [r['id'] for r in records] != [r['id'] for r in cases]:
        raise ValueError('Incomplete or misaligned real evaluation')
    for record, case in zip(records, cases):
        if record['source'] != case['source'] or record['reference'] != case['target']:
            raise ValueError('Source or reference changed')
    calculated = evaluate(records)
    saved = read_json(folder/'metrics.json')
    for field in ['n', 'whitespace_rouge_l_f1', 'number_multiset_preservation_rate', 'copy_rate']:
        if abs(calculated[field] - saved[field]) > 1e-9:
            raise ValueError('Saved metrics disagree with predictions: '+field)
    calculated.update(cross_entropy=saved['cross_entropy'],
                      same_word_sequence=sum(words(r['source']) == words(r['prediction']) for r in records),
                      mean_word_edit_rate=statistics.mean(edit_distance(words(r['source']), words(r['prediction'])) /
                                                         max(1, len(words(r['source']))) for r in records),
                      generation_limit_reached_count=sum(r['generation_limit_reached'] for r in records),
                      input_truncated_count=sum(r['input_truncated'] for r in records),
                      target_truncated_count=sum(r['target_truncated'] for r in records),
                      status='inspected_real_'+split+'_diagnostic')
    return calculated, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/six_layer_experiment.json')
    parser.add_argument('--run', help='Checkpoint directory when the training output was overridden')
    parser.add_argument('--output', default='output/six-layer-v1')
    parser.add_argument('--budget', type=int, default=192)
    args = parser.parse_args()
    config = read_json(ROOT/args.config)
    if args.run:
        config['output'] = args.run
    data, run, out = ROOT/config['data'], ROOT/config['output'], ROOT/args.output
    manifest = read_json(data/'manifest.json')
    if fingerprint(ROOT/'src/academic_rewriter/synthetic.py') != manifest['generator_code_sha256']:
        raise ValueError('Stale generator provenance')
    train = read_jsonl(data/'train.jsonl')
    synthetic = [r for r in train if r.get('synthetic')]
    for row in synthetic:
        validate_pair(row)
    if len({row['id'] for row in train}) != len(train):
        raise ValueError('Duplicate training IDs')
    if len(train) != manifest['counts']['total_training'] or len(synthetic) != manifest['counts']['synthetic_training']:
        raise ValueError('Data composition disagrees with manifest')
    for split, expected in manifest['data_sha256'].items():
        if fingerprint(data/(split+'.jsonl')) != expected:
            raise ValueError('Changed data bytes: '+split)
        if split != 'train' and (data/(split+'.jsonl')).read_bytes() != (ROOT/'data/processed'/(split+'.jsonl')).read_bytes():
            raise ValueError('Real held-out data changed')
    metadata = read_json(run/'result.json') if (run/'result.json').exists() else None
    if metadata:
        if metadata['sequence_layers'] != 6 or metadata['layer_layout'] != config['layer_layout']:
            raise ValueError('Recorded architecture does not match the six-layer study')
        if metadata.get('training_data_manifest_sha256') != fingerprint(data/'manifest.json'):
            raise ValueError('Checkpoint trained on a different manifest')
        for split, expected in metadata['data_sha256'].items():
            if fingerprint(data/(split+'.jsonl')) != expected:
                raise ValueError('Checkpoint trained on different split bytes')
    diagnostics, predictions = {}, {}
    for split in ['validation', 'test']:
        diagnostics[split], predictions[split] = verified_diagnostics(run, split, data, args.budget)
    sources = [args.config, config['data']+'/manifest.json', config['data']+'/train.jsonl',
               config['data']+'/validation.jsonl', config['data']+'/test.jsonl',
               'src/academic_rewriter/six_layer.py', 'src/academic_rewriter/synthetic.py',
               'scripts/report_six_layer.py']
    for name in ['result.json', 'weights.pt', 'six_layer_config.json', 'training_log.json', 'validation_log.json']:
        if (run/name).exists():
            sources.append(str((run/name).relative_to(ROOT)).replace('\\', '/'))
    for split in ['validation', 'test']:
        folder = run/'evaluation'/f'{split}-{args.budget}'
        for name in ['predictions.jsonl', 'metrics.json']:
            if (folder/name).exists():
                sources.append(str((folder/name).relative_to(ROOT)).replace('\\', '/'))
    result = dict(study=config['study'], data=manifest, model=metadata,
                  sequence_layers=config['layer_layout'], diagnostics=diagnostics,
                  generation_budget=args.budget, academic_improvement=None, successful_rewrite_rate=None,
                  quality_status='Independent academic, fluency and factual-quality evidence remains unavailable.',
                  synthetic_examples=synthetic[:24],
                  source_sha256={name: fingerprint(ROOT/name) for name in sources})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'summary.json', result)
    if metadata:
        training = read_json(run/'training_log.json')
        validation = read_json(run/'validation_log.json')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
        axes[0].plot([r['step'] for r in training], [r['train_loss'] for r in training], color='#bddbd6', linewidth=.65)
        axes[0].plot([r['step'] for r in training], [r['train_loss_ma20'] for r in training], color='#087c72', linewidth=1.5)
        axes[0].set(title='Synthetic-augmented training', xlabel='Optimizer update', ylabel='Target-token cross-entropy')
        axes[1].plot([r['step'] for r in validation], [r['validation_cross_entropy'] for r in validation],
                     marker='o', color='#c26847', linewidth=1.5)
        axes[1].axvline(metadata['selected_step'], color='#637580', linestyle='--', label='Selected checkpoint')
        axes[1].set(title='Unchanged real validation set (98 pairs)', xlabel='Optimizer update', ylabel='Target-token cross-entropy')
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', alpha=.15)
        fig.suptitle('Six sequence blocks: 2 LSTM + 2 Transformer + 2 Mamba', fontsize=14, fontweight='bold')
        fig.savefig(out/'learning_curve.png', dpi=170)
        plt.close(fig)
    escape = html.escape
    cards = ''.join(f'<div class="layer {kind}"><small>Layer {i}</small><strong>{kind.upper()}</strong></div>'
                    for i, kind in enumerate(config['layer_layout'], 1))
    count = manifest['counts']
    status = (f'Trained for {metadata["steps"]:,} updates; checkpoint selected at update {metadata["selected_step"]:,}.'
              if metadata else 'Training has not yet completed.')
    table_rows = ''
    for split, metrics in diagnostics.items():
        if metrics:
            table_rows += (f'<tr><td>{split.title()}</td><td>{metrics["n"]}</td>'
                           f'<td>{metrics["same_word_sequence"]}/{metrics["n"]}</td>'
                           f'<td>{metrics["number_multiset_preservation_rate"]:.1%}</td>'
                           f'<td>{metrics["mean_word_edit_rate"]:.1%}</td>'
                           f'<td>{metrics["generation_limit_reached_count"]}</td></tr>')
    cases = ''
    for split, records in predictions.items():
        if not records:
            continue
        cases += f'<h3>{split.title()} outputs · all {len(records)} cases</h3>'
        for index, row in enumerate(records, 1):
            cases += (f'<details><summary>{index:02d} · {escape(row["id"])}</summary>'
                      f'<b>Source</b><p>{escape(row["source"])}</p><b>Model output</b>'
                      f'<p>{escape(row["prediction"])}</p><b>Human revision reference</b>'
                      f'<p>{escape(row["reference"])}</p></details>')
    examples = ''.join(f'<tr><td>{escape(row["source"])}</td><td>{escape(row["target"])}</td></tr>'
                       for row in synthetic[:12])
    figure = '<img src="learning_curve.png" alt="Training loss and real validation loss">' if metadata else ''
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Six-layer academic rewriter</title>
<style>body{{margin:0;background:#faf8f3;color:#152c3c;font:16px/1.65 'Segoe UI',Arial,sans-serif}}
main{{max-width:1140px;margin:auto;padding:45px 28px}}h1{{font-size:42px;line-height:1.15}}h2{{margin-top:44px}}
a{{color:#087c72}}.muted,small{{color:#637580}}.stack{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:25px 0}}
.layer{{padding:18px 12px;border-top:5px solid #087c72;background:white}}.layer small,.layer strong{{display:block}}
.transformer{{border-color:#4473c5}}.mamba{{border-color:#c26847}}.note{{padding:18px 22px;background:#e8efeb;border-left:4px solid #087c72;margin:22px 0}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:13px;border-bottom:1px solid #dce3df}}
img{{width:100%;height:auto}}details{{padding:14px 18px;background:white;border:1px solid #dce3df;margin:10px 0}}summary{{cursor:pointer}}p{{overflow-wrap:anywhere}}
@media(max-width:700px){{.stack{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:32px}}main{{padding:28px 18px}}}}</style></head>
<body><main><a href="../academic-refinement/index.html">Academic data audit</a> · <a href="../../docs/Six_Layer_Model.md">Model and protocol</a>
<p class="muted">Versioned exploratory study · {escape(config['study'])}</p><h1>Six layers.<br>Paired academic expression training.</h1>
<div class="stack">{cards}</div><p>Two LSTM blocks, two causal self-attention + feed-forward Transformer blocks, then two Mamba blocks.
Token/position/segment embeddings and a tied vocabulary projection surround these six blocks. There is no additional BART encoder or decoder.</p>
<div class="note">{escape(status)} <strong>Useful academic rewriting has not been independently established.</strong></div>
<h2>Training data that actually exists</h2><p><strong>{count['synthetic_training']:,} synthetic pairs + {count['real_training']:,} existing real revisions = {count['total_training']:,} training pairs.</strong>
The synthetic pairs use {manifest['template_families']} authored semantic frames across {manifest['domains']} technical domains.
Both sides use shared facts, quantities and protected terms. All findings are hypothetical. Automatic guards do not establish semantic equivalence.</p>
<p>All {count['validation']} validation and {count['test']} test cases are real existing revisions; their exact bytes are unchanged.
Synthetic examples are training-only. Template diversity is limited, and the original real pairs are mixed-intent revisions.</p>
<h2>Measured diagnostics</h2><p>These real sets have already been inspected. They are not fresh confirmatory evaluations.
Copying, editing and number preservation are diagnostics, not academic-quality scores. Different tokenization also prevents direct CE comparison with historical BART runs.</p>
<div class="scroll"><table><thead><tr><th>Real split</th><th>Pairs</th><th>Same word sequence</th><th>Numbers preserved</th><th>Word edit rate</th><th>Output cap hits</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p class="muted">Academic improvement: unavailable · Independent fluency/meaning ratings: unavailable · Successful rewriting rate: unavailable</p>{figure}
<h2>Inspect generated pairs</h2><p>Training examples below are generated data, not model predictions or independent human examples.</p>
<div class="scroll"><table><thead><tr><th>Less academic expression</th><th>Academic target</th></tr></thead><tbody>{examples}</tbody></table></div>
<h2>Inspect actual model outputs</h2>{cases}
<p><a href="summary.json">Verified numerical sources and SHA-256 fingerprints</a></p></main></body></html>'''
    (out/'index.html').write_text(document, encoding='utf8', newline='\n')
    print(json.dumps(dict(training_pairs=count['total_training'], trained=metadata is not None,
                         diagnostics=diagnostics, report=str(out/'index.html')), indent=2))


if __name__ == '__main__':
    main()
