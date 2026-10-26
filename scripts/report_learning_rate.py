"""Summarize measured loss noise and validation-selected learning-rate trials."""
import argparse
import json
import math
import statistics
from pathlib import Path

from tune_learning_rate import loss_statistics


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def resolve_run_path(project, value):
    # Historical records may contain Windows paths from the original machine.
    parts = [part for part in str(value).replace(chr(92), '/').split('/') if part]
    if 'runs' in parts:
        return project.joinpath(*parts[parts.index('runs'):])
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project/candidate


def moving_average(values, window=20):
    return [statistics.mean(values[max(0,i-window+1):i+1]) for i in range(len(values))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', default='runs/lr_tuning_offline/comparison.json')
    p.add_argument('--output', default='output/learning-rate-analysis')
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    report = read_json(project/a.comparison)
    all_runs = report['runs']+([report['adaptive_trial']] if 'adaptive_trial' in report else [])
    out = project/a.output
    out.mkdir(parents=True, exist_ok=True)
    baseline_root = resolve_run_path(project, report['reference_checkpoint'])
    reference = read_json(baseline_root/'training_log.json')
    frozen = read_json(project/'runs/bart_adapter_visible/training_log.json')
    x = [r['train_loss'] for r in reference]
    y = [r['train_loss'] for r in frozen[:len(x)]]
    xm, ym = statistics.mean(x), statistics.mean(y)
    correlation = sum((u-xm)*(v-ym) for u,v in zip(x,y))/math.sqrt(sum((u-xm)**2 for u in x)*sum((v-ym)**2 for v in y))
    evidence = dict(unfrozen_reference=loss_statistics(reference), frozen_reference=loss_statistics(frozen),
                    frozen_unfrozen_same_step_loss_correlation=correlation,
                    interpretation='Correlated spikes and decreasing window means support batch-difficulty noise; they do not establish an excessively high LR.',
                    comparison=report)
    (out/'analysis.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    colors = ['#3468a8', '#da7b24', '#2c8c72', '#9a55a5']
    for color, row in zip(colors, all_runs):
        root = resolve_run_path(project, row['checkpoint'])
        log = read_json(root/'training_log.json')
        values = [entry['train_loss'] for entry in log]
        steps = [entry['step'] for entry in log]
        label = f"LR {row['learning_rate']:.0e}"+(' plateau' if row.get('lr_scheduler')=='plateau' else '')
        if root==baseline_root:
            axes[0].plot(steps, values, color=color, alpha=.22, linewidth=.7, label=label+' raw')
        axes[0].plot(steps, moving_average(values), color=color, linewidth=1.6, label=label+' MA20')
        history_path = root/'validation_log.json'
        if history_path.exists():
            history = read_json(history_path)
            axes[1].plot([v['step'] for v in history], [v['validation_cross_entropy'] for v in history], marker='o', markersize=4, color=color, label=label)
        else:
            axes[1].scatter([len(log)], [row['validation_cross_entropy']], marker='X', s=65, color=color, label=label+' final only')
    axes[0].set(title='Mini-batch noise and trend', xlabel='Optimizer step', ylabel='Training cross-entropy')
    axes[1].set(title='Validation loss (98 pairs)', xlabel='Optimizer step', ylabel='Token-weighted cross-entropy')
    labels = [f"{row['learning_rate']:.0e}"+('\nplateau' if row.get('lr_scheduler')=='plateau' else '\nconstant') for row in all_runs]
    values = [row['validation_cross_entropy'] for row in all_runs]
    axes[2].bar(labels, values, color=colors[:len(values)])
    for i, value in enumerate(values):
        axes[2].text(i, value+.012, f'{value:.4f}', ha='center', va='bottom')
    axes[2].set(title='Final loss at equal training budget', xlabel='Initial learning rate / schedule', ylabel='Validation cross-entropy', ylim=(0,max(values)*1.13))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle('Full BART + Mamba fine-tuning | same seed, batches and 300 steps | no test-set selection')
    fig.savefig(out/'loss_comparison.png', dpi=180)
    fig.savefig(out/'loss_comparison.pdf')
    plt.close(fig)

    rows = '\n'.join(f"| {r['learning_rate']:.0e} | {r.get('lr_scheduler','constant')} | {r.get('last_applied_learning_rate',r['learning_rate']):.0e} | {r['loss_statistics']['first_50_mean']:.4f} | {r['loss_statistics']['last_50_mean']:.4f} | {r['loss_statistics']['std']:.4f} | {r['validation_cross_entropy']:.4f} |" for r in all_runs)
    selected = report['recommended_learning_rate']
    selected_scheduler = report.get('recommended_scheduler','constant')
    selected_patience = report.get('recommended_lr_patience',1)
    scheduler_flags = f' --lr-patience {selected_patience} --lr-drop-factor 0.5 --min-lr 1e-6' if selected_scheduler=='plateau' else ''
    markdown = f'''# Learning rate and loss analysis

Measured on 26 October 2026. These are single-seed pilot experiments.

## Loss variation

The full hybrid at 2e-5 has a first-50-step mean loss of {statistics.mean(x[:50]):.4f} and a last-50-step mean of {statistics.mean(x[-50:]):.4f}. The maximum individual loss is {max(x):.4f}. Each update samples only two training pairs, so successive losses measure different batches.

The frozen and full hybrid losses have same-step correlation {correlation:.4f}. Their shared spikes support batch difficulty as an explanation. They do not, by themselves, establish an excessively high learning rate or an architectural advantage.

## Controlled learning-rate comparison

The runs use facebook/bart-base, hybrid/reference/gated, seed 42, 491 training pairs, 98 validation pairs, batch size 2, a 128-token input limit and 300 updates. Constant-rate trials change only the learning rate. New trials evaluate validation loss every 50 steps. The reference run has only a final validation value, which the plot marks explicitly.

| Initial LR | Schedule | Last applied LR | First 50 mean | Last 50 mean | Step loss SD | Final validation CE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{rows}

The lowest final validation loss among constant-rate trials selects {selected:.0e}. Considering the additional scheduler trials, the current recommendation for this limited comparison is {selected:.0e} with a {selected_scheduler} schedule. Smoother training loss is not the selection criterion. The comparison does not use test scores and does not demonstrate better revision quality.

## Training controls

- training_progress.jsonl records raw loss, a 20-step moving average, the actual learning rate and the gradient norm before clipping.
- --eval-every 50 records validation token CE and fixed-batch training loss at step 0 and every 50 steps.
- --lr-scheduler plateau responds to validation loss, with relative improvement threshold 0.1%, a configurable patience, drop factor and minimum LR. Individual training spikes do not trigger it.
- A scheduler change after the final validation has not trained any weights. The table therefore reports the last learning rate actually applied.
- weights.pt contains the final-step weights. best_recorded_validation_cross_entropy records an observation, not a saved best checkpoint. Validation-based checkpoint saving and early stopping remain future work.
- Default learning rates are 1e-3 for tiny smoke runs, 1e-4 for a frozen hybrid and 2e-5 for full fine-tuning. An explicit --lr overrides these defaults.

The patience-1 diagnostic changes the rate only after step 300. The patience-0 trial applies the decrease after step 250 and uses it for the last 50 updates. These trials remain in runs/bart_hybrid_lr_adjusted and runs/bart_hybrid_lr_adjusted_patience0.

## Reproduction

~~~powershell
.\.venv\Scripts\python.exe -u -m academic_rewriter.run --model hybrid --backend reference --fusion gated --output runs/hybrid_repeat --steps 300 --train-limit 0 --val-limit 0 --batch-size 2 --lr {selected:.0e} --eval-every 50 --lr-scheduler {selected_scheduler}{scheduler_flags} --log-every 10 --local-files-only
.\.venv\Scripts\python.exe scripts/report_learning_rate.py
~~~

The offline flag requires an existing cached pretrained backbone. Raw comparisons are in runs/lr_tuning_offline/comparison.json. Figures and their numerical source are in output/learning-rate-analysis.
'''
    report_path = project/'docs/Learning_Rate_Analysis.md'
    report_path.write_text(markdown, encoding='utf8')
    print(json.dumps(dict(recommended_learning_rate=selected, output=str(out), report=str(report_path.relative_to(project))), indent=2))


if __name__=='__main__':
    main()
