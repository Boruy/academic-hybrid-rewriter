"""Add reviewed arXivEdits training pairs to a new local LexiFAR study."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')
SCORES = ('meaning', 'fluency', 'source_fluency')
DECISIONS = ('source_suitable', 'source_needs_revision', 'target_suitable', 'major_fact_error')
CATEGORIES = ('Academic Style', 'Grammar / Fluency', 'Content Change', 'No Meaningful Improvement')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_lines(data):
    return [json.loads(line) for line in data.decode('utf-8-sig').split('\n') if line.strip()]


def normalized_hash(text):
    # Match the normalization used by the frozen arXivEdits references.
    return sha256(' '.join(re.findall(r'\w+', text.casefold())).encode('utf8'))


def approves(review):
    if any(review.get(field) is None for field in ('category', *SCORES, *DECISIONS)):
        return False
    if review.get('reviewer_type') != 'human' or review['category'] not in CATEGORIES:
        raise ValueError('A completed review requires a human reviewer and a known category')
    if any(type(review[key]) is not int or not 1 <= review[key] <= 5 for key in SCORES):
        raise ValueError('Meaning and fluency scores must be integers from 1 to 5')
    if any(type(review[key]) is not bool for key in DECISIONS):
        raise ValueError('Suitability, revision need and factual-error judgments must be booleans')
    return (review['category'] == 'Academic Style' and review['meaning'] >= 4
            and review['fluency'] >= 4 and review['source_needs_revision']
            and review['target_suitable'] and not review['major_fact_error'])


def read_curator(data, candidates):
    rows = json_lines(data)
    if not rows:
        raise ValueError('A curator file must contain scored training reviews')
    rater = rows[0].get('rater_id')
    if not isinstance(rater, str) or not rater.strip():
        raise ValueError('Each file needs a curator ID')

    seen = set()
    accepted = {}
    for row in rows:
        identifier = row.get('id')
        ref = candidates.get(identifier)
        if ref is None:
            raise ValueError(f'{identifier}: expected a screened candidate from the frozen train partition')
        if row.get('rater_id') != rater or identifier in seen:
            raise ValueError('Each file must contain one curator and at most one review per candidate')
        seen.add(identifier)
        for key in ('doc_id', 'split', 'source_sha256', 'target_sha256'):
            if row.get(key) != ref[key]:
                raise ValueError(f'{identifier}: review metadata differs from its frozen reference')
        for field in ('source', 'target'):
            text = row.get(field)
            if not isinstance(text, str) or sha256(text.encode('utf8')) != ref[field + '_sha256']:
                raise ValueError(f'{identifier}: review text differs from its frozen reference')
        if approves(row):
            accepted[identifier] = row
    return rater, accepted


def approved_additions(review_files, references, partition):
    candidates = {ref['id']: ref for ref in references
                  if ref['task_candidate'] and ref['split'] == 'train'
                  and partition['documents'].get(ref['doc_id']) == 'train'}
    (first_rater, first), (second_rater, second) = (
        read_curator(data, candidates) for data in review_files
    )
    raters = (first_rater, second_rater)
    if first_rater.strip().casefold() == second_rater.strip().casefold():
        raise ValueError('Two distinct curator IDs are required for independent reviews')

    additions = []
    for identifier in sorted(first.keys() & second.keys()):
        ref = candidates[identifier]
        additions.append({**ref, 'source': first[identifier]['source'], 'target': first[identifier]['target'],
                          'group_id': ref['doc_id'], 'intent': 'style', 'synthetic': False,
                          'needs_revision': True, 'human_verified': True,
                          'review_status': 'accepted_by_independent_curators', 'curation_raters': sorted(raters)})
    return additions, dict(zip(raters, map(sha256, review_files), strict=True))


def check_isolation(additions, base_rows, references, partition):
    def pair_key(row):
        return normalized_hash(row['source']), normalized_hash(row['target'])

    original_rows = [row for split in SPLITS for row in base_rows[split]]
    forbidden_docs = set(partition['previously_inspected_documents'])
    forbidden_docs.update(row.get('doc_id') for row in original_rows)
    held_out_text = {key for split in ('validation', 'test') for row in base_rows[split] for key in pair_key(row)}
    held_out_text.update(ref[field] for ref in references if ref['split'] != 'train'
                         for field in ('normalized_source_sha256', 'normalized_target_sha256'))
    ids = {row['id'] for row in original_rows}
    pairs = {pair_key(row) for row in base_rows['train']}
    for row in additions:
        key = pair_key(row)
        if row['doc_id'] in forbidden_docs or held_out_text.intersection(key):
            raise ValueError(f"{row['id']}: overlap with a protected document or held-out text")
        if row['id'] in ids or key in pairs:
            raise ValueError(f"{row['id']}: duplicate training pair or ID")
        ids.add(row['id'])
        pairs.add(key)


def build(root, version, review_paths):
    study = f'academic-pairs-v{version}'
    dataset_path = Path('.cache') / study
    output = root / dataset_path
    if output.exists():
        raise FileExistsError(f'{dataset_path} already exists. Use a new --version to preserve its bytes.')

    curation = root / 'data/curation/arxivedits'
    frozen_bytes = (curation / 'frozen_partition.json').read_bytes()
    partition = json.loads(frozen_bytes)
    reference_bytes = (curation / 'references.jsonl').read_bytes()
    references = json_lines(reference_bytes)
    review_files = [path.read_bytes() for path in review_paths]
    additions, review_hashes = approved_additions(review_files, references, partition)
    if not additions:
        raise ValueError('No training pairs passed both independent reviews. No expanded dataset was created.')

    base = root / 'data/academic-pairs-v1'
    base_manifest_bytes = (base / 'manifest.json').read_bytes()
    base_manifest = json.loads(base_manifest_bytes)
    generator_hash = base_manifest['generator_code_sha256']
    if sha256((root / 'src/academic_rewriter/synthetic.py').read_bytes()) != generator_hash:
        raise ValueError('The frozen synthetic generator fingerprint changed')
    data = {}
    for split in SPLITS:
        name = split + '.jsonl'
        data[split] = (base / name).read_bytes()
        original = (root / 'data/processed' / name).read_bytes()
        if sha256(original) != partition['previous_data_sha256'][split]:
            raise ValueError(f'The original IteraTeR {split} fingerprint changed')
        if sha256(data[split]) != base_manifest['data_sha256'][split]:
            raise ValueError(f'The frozen v1 {split} fingerprint changed')
        if split != 'train' and data[split] != original:
            raise ValueError(f'The original held-out {split} bytes must be preserved')

    check_isolation(additions, {split: json_lines(raw) for split, raw in data.items()}, references, partition)
    extra = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in additions).encode('utf8')
    separator = b'' if data['train'].endswith(b'\n') else b'\n'
    data['train'] += separator + extra
    counts = dict(base_manifest['counts'])
    counts['real_training'] += len(additions)
    counts['total_training'] += len(additions)

    config_path = root / 'configs/six_layer_span_edits.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config.update(study=f'six-layer-v8-{study}', data=dataset_path.as_posix(),
                  output=f"runs/six_layer_joint_v8_{study.replace('-', '_')}")
    manifest = {
        'version': study,
        'base_study': 'data/academic-pairs-v1',
        'base_manifest_sha256': sha256(base_manifest_bytes),
        'base_train_sha256': base_manifest['data_sha256']['train'],
        'generator_code_sha256': generator_hash,
        'base_experiment_sha256': sha256(config_bytes),
        'data_sha256': {split: sha256(raw) for split, raw in data.items()},
        'counts': counts,
        'arxivedits': {
            'revision': partition['revision'],
            'references_sha256': sha256(reference_bytes),
            'frozen_partition_sha256': sha256(frozen_bytes),
            'review_files_sha256': review_hashes,
            'approved_training': len(additions),
            'approved_documents': len({row['doc_id'] for row in additions}),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='lexifar-expansion-', dir=output.parent) as temporary:
        stage = Path(temporary) / study
        stage.mkdir()
        for split in SPLITS:
            (stage / (split + '.jsonl')).write_bytes(data[split])
        for name, value in (('manifest.json', manifest), ('experiment.json', config)):
            (stage / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf8', newline='\n')
        stage.rename(output)
    print(f"Added {len(additions)} approved pairs. {counts['real_training']} real training pairs in {dataset_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs=2, required=True, type=Path, metavar='JSONL',
                        help='Training review exports from two independent human curators')
    parser.add_argument('--version', type=int, default=2, help='A new version number for the local study')
    args = parser.parse_args()
    if args.version < 2:
        parser.error('--version must be at least 2')
    try:
        build(ROOT, args.version, args.reviews)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
