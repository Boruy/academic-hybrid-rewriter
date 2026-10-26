"""Compare constant learning rates on validation data with identical training draws."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def loss_statistics(log):
    values = [row['train_loss'] for row in log]
    return {
        'steps': len(values),
        'mean': statistics.mean(values),
        'std': statistics.pstdev(values),
        'min': min(values),
        'max': max(values),
        'first_50_mean': statistics.mean(values[:50]),
        'last_50_mean': statistics.mean(values[-50:]),
        'mean_absolute_step_change': statistics.mean(abs(a-b) for a,b in zip(values, values[1:])) if len(values)>1 else 0,
        'steps_over_2': sum(value>2 for value in values),
    }


def collect(root):
    meta = json.loads((root/'result.json').read_text(encoding='utf8'))
    log = json.loads((root/'training_log.json').read_text(encoding='utf8'))
    return dict(checkpoint=str(root), learning_rate=meta['learning_rate'],
                final_learning_rate=meta.get('final_learning_rate',meta['learning_rate']),
                last_applied_learning_rate=log[-1].get('learning_rate',meta['learning_rate']),
                lr_scheduler=meta.get('lr_scheduler','constant'),
                validation_cross_entropy=meta['validation_cross_entropy'],
                fixed_train_batch_loss_after=meta['fixed_train_batch_loss_after'],
                loss_statistics=loss_statistics(log), test_set_used=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/lr_tuning')
    p.add_argument('--reference-checkpoint', default='runs/bart_hybrid_unfrozen')
    p.add_argument('--rates', type=float, nargs='+', default=[1e-5, 5e-6])
    a = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    reference = project/a.reference_checkpoint
    meta = json.loads((reference/'result.json').read_text(encoding='utf8'))
    if meta['model']!='hybrid' or meta['args']['tiny'] or meta['args']['freeze_backbone']:
        raise ValueError('Reference must be a full fine-tuning hybrid run')
    output = project/a.output
    output.mkdir(parents=True, exist_ok=True)
    results = [collect(reference)]
    for rate in a.rates:
        if rate<=0:
            raise ValueError('Learning rates must be positive')
        run = output/('hybrid_lr_'+format(rate, '.0e').replace('-', 'm').replace('+', 'p'))
        if run.exists():
            raise FileExistsError(f'Choose a new --output; preserving existing run: {run}')
        cmd = [sys.executable, '-u', '-m', 'academic_rewriter.run', '--model', 'hybrid',
               '--pretrained', meta['args']['pretrained'], '--backend', meta['args']['backend'],
               '--fusion', meta['args']['fusion'], '--data', meta['args']['data'],
               '--output', str(run), '--steps', str(meta['steps']),
               '--train-limit', str(meta['args']['train_limit']), '--val-limit', str(meta['args']['val_limit']),
               '--batch-size', str(meta['batch_size']), '--max-length', str(meta['max_length']),
               '--seed', str(meta['seed']), '--lr', str(rate), '--lr-scheduler', 'constant',
               '--eval-every', '50', '--log-every', '25', '--local-files-only']
        print(f'Comparing LR={rate:.2e}; checkpoint={run}', flush=True)
        (output/(run.name+'_command.json')).write_text(json.dumps(cmd, indent=2), encoding='utf8')
        with (output/(run.name+'.log')).open('w', encoding='utf8') as stream:
            subprocess.run(cmd, cwd=project, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = collect(run)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        (output/'comparison.partial.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    best = min(results, key=lambda row:row['validation_cross_entropy'])
    report = dict(protocol=f"same data, initialization, seed, random batches, {meta['steps']}-step reference budget; constant LR; final validation CE selection",
                  reference_checkpoint=str(reference), runs=results,
                  recommended_learning_rate=best['learning_rate'], selected_checkpoint=best['checkpoint'],
                  limitation='Single seed and fixed training budget; raw mini-batch loss variance is not the selection criterion.',
                  test_set_used=False)
    (output/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    main()
