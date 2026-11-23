"""Launch the versioned six-layer experiment after verifying its data manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/six_layer_experiment.json')
    parser.add_argument('--output', help='Checkpoint directory (overwritten with the default config)')
    parser.add_argument('--init-checkpoint', help='Override the initial checkpoint in the experiment config')
    parser.add_argument('--depth', choices=['joint', 'medium', 'deep'],
                        help='Default is joint; medium and deep train a single output')
    parser.add_argument('--word-guidance', choices=['position', 'candidates'],
                        help='Override the joint word guidance in the experiment config')
    parser.add_argument('--log-every', type=int, help='Print progress every N updates')
    parser.add_argument('--steps', type=int, help='Override the optimizer update budget')
    parser.add_argument('--seed', type=int, help='Override the training and sampling seed')
    parser.add_argument('--span-payload-weight', type=float, help='Override the training-only span replacement token weight')
    parser.add_argument('--checkpoint-selection', choices=['final', 'best-validation'],
                        help='Override how the checkpoint is chosen')
    parser.add_argument('--early-stopping-patience', type=int,
                        help='Stop after this many unimproved validation checks (0 disables stopping)')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    if args.log_every is not None and args.log_every < 1:
        parser.error('--log-every must be positive')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    if args.span_payload_weight is not None and (not math.isfinite(args.span_payload_weight) or args.span_payload_weight < 1.):
        parser.error('--span-payload-weight must be finite and at least 1')
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        parser.error('--early-stopping-patience must be nonnegative')
    config = json.loads((ROOT/args.config).read_text(encoding='utf8'))
    if args.output:
        config['output'] = args.output
    data = ROOT/config['data']
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf8'))
    if manifest['generator_code_sha256'] != hashlib.sha256(
            (ROOT/'src/academic_rewriter/synthetic.py').read_bytes()).hexdigest():
        raise ValueError('Generator code changed; use a new versioned study')
    for split in ['train', 'validation', 'test']:
        path = data/(split+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['data_sha256'][split]:
            raise ValueError('Prepared split fingerprint changed')
        if split != 'train':
            if path.read_bytes() != (ROOT/'data/processed'/path.name).read_bytes():
                raise ValueError('Original real held-out bytes must be preserved')
            if any(json.loads(line).get('synthetic') for line in path.read_text(encoding='utf8').splitlines()):
                raise ValueError('Synthetic data cannot enter held-out sets')
    settings = dict(config['settings'], model=config['model'], data=config['data'], output=config['output'])
    if args.init_checkpoint:
        settings['init_checkpoint'] = args.init_checkpoint
    if args.depth:
        settings['depth'] = args.depth
        if args.depth != 'joint':
            settings.pop('source_copy', None)
            settings.pop('contextual_edits', None)
            settings.pop('span_edits', None)
    if args.word_guidance:
        settings['word_guidance'] = args.word_guidance
    if args.log_every is not None:
        settings['log_every'] = args.log_every
    if args.steps is not None:
        settings['steps'] = args.steps
    if args.seed is not None:
        settings['seed'] = args.seed
    if args.span_payload_weight is not None:
        settings['span_payload_weight'] = args.span_payload_weight
    if args.checkpoint_selection:
        settings['checkpoint_selection'] = args.checkpoint_selection
    if args.early_stopping_patience is not None:
        settings['early_stopping_patience'] = args.early_stopping_patience
    if settings.get('depth', 'joint') == 'joint' and 'word_data_sha256' in config:
        word_path = ROOT/settings['word_data']
        if hashlib.sha256(word_path.read_bytes()).hexdigest() != config['word_data_sha256']:
            raise ValueError('Word training fingerprint changed; use a new versioned study')
    if settings.get('grammar_data'):
        grammar_path = ROOT/settings['grammar_data']
        if hashlib.sha256(grammar_path.read_bytes()).hexdigest() != config.get('grammar_data_sha256'):
            raise ValueError('Grammar training fingerprint changed; use a new versioned supplement')
    command = [sys.executable, '-m', 'academic_rewriter.run']
    for key, value in settings.items():
        flag = '--'+key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    print(subprocess.list2cmdline(command), flush=True)
    if not args.print_command:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
