"""Launch the versioned six-layer experiment after verifying its data manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/six_layer_experiment.json')
    parser.add_argument('--output', help='Checkpoint directory (overwritten with the default config)')
    parser.add_argument('--depth', choices=['joint', 'medium', 'deep'],
                        help='Default is joint; medium and deep reproduce legacy single-output runs')
    parser.add_argument('--word-guidance', choices=['position', 'candidates'],
                        help='Override the joint word guidance in the experiment config')
    parser.add_argument('--log-every', type=int, help='Print progress every N updates')
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    if args.log_every is not None and args.log_every < 1:
        parser.error('--log-every must be positive')
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
    if args.depth:
        settings['depth'] = args.depth
    if args.word_guidance:
        settings['word_guidance'] = args.word_guidance
    if args.log_every is not None:
        settings['log_every'] = args.log_every
    if settings.get('depth', 'joint') == 'joint' and 'word_data_sha256' in config:
        word_path = ROOT/settings['word_data']
        if hashlib.sha256(word_path.read_bytes()).hexdigest() != config['word_data_sha256']:
            raise ValueError('Word training fingerprint changed; use a new versioned study')
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
