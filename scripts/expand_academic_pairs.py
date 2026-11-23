"""Add arXivEdits training pairs to a new local LexiFAR study."""

import argparse
from collections import Counter
from difflib import SequenceMatcher
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
ALIGNMENT_FILE = 'data/sentence_alignment/train.json'
MIN_REAL_PAIRS = 10_000
MIN_WORD_OVERLAP = .75
WORD = re.compile(r'\w+')
ACRONYM = re.compile(r'\b[A-Z][A-Z0-9-]+\b')
PLACEHOLDER = re.compile(r'\[(?:MATH|CITATION|REF|EQUATION)\]')
NUMERIC_PART = re.compile(r'\d+(?:[.,]\d+)*')
PLACEHOLDER_NAMES = frozenset(('MATH', 'CITATION', 'REF', 'EQUATION'))
PROTECTED_WORDS = frozenset(
    'may might could can cannot not no never neither nor only all some most approximately almost '
    'possibly probably significant significantly'.split()
)
EXTRACTION_ARTIFACTS = re.compile(r'\\|%DIF|\[(?:PAD|BOS|EOS|UNK)\]|\ufffd')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_lines(data):
    return [json.loads(line) for line in data.decode('utf-8-sig').split('\n') if line.strip()]


def normalized_hash(text):
    # Match the normalization used by the frozen arXivEdits references.
    return sha256(' '.join(WORD.findall(text.casefold())).encode('utf8'))


def document_key(doc_id):
    return re.sub(r'v\d+$', '', doc_id.casefold()).replace('/', '-')


def pair_key(row):
    return normalized_hash(row['source']), normalized_hash(row['target'])


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


def isolation_keys(base_rows, references, partition):
    forbidden_docs = {document_key(doc) for doc in partition['previously_inspected_documents']}
    forbidden_docs.update(document_key(doc) for doc, split in partition['documents'].items() if split != 'train')
    forbidden_docs.update(document_key(row['doc_id']) for split in SPLITS for row in base_rows[split]
                          if row.get('doc_id'))
    held_out_text = {key for split in ('validation', 'test') for row in base_rows[split] for key in pair_key(row)}
    held_out_text.update(ref[field] for ref in references if ref['split'] != 'train'
                         for field in ('normalized_source_sha256', 'normalized_target_sha256'))
    return forbidden_docs, held_out_text, {pair_key(row) for row in base_rows['train']}


def check_isolation(additions, base_rows, references, partition):
    forbidden_docs, held_out_text, pairs = isolation_keys(base_rows, references, partition)
    ids = {row['id'] for split in SPLITS for row in base_rows[split]}
    for row in additions:
        key = pair_key(row)
        if document_key(row['doc_id']) in forbidden_docs or held_out_text.intersection(key):
            raise ValueError(f"{row['id']}: overlap with a protected document or held-out text")
        if row['id'] in ids or key in pairs:
            raise ValueError(f"{row['id']}: duplicate training pair or ID")
        ids.add(row['id'])
        pairs.add(key)


def aligned_revisions(paper):
    sentences = {identifier: (int(version), text) for version, entries in paper.items()
                 if version.isdecimal() for identifier, text in entries.items()}
    edges = [(source, target) for source, target in paper['alignment']
             if source in sentences and target in sentences and sentences[source][0] + 1 == sentences[target][0]]
    outgoing = Counter(source for source, _ in edges)
    incoming = Counter(target for _, target in edges)
    for source, target in edges:
        if outgoing[source] == incoming[target] == 1:
            yield source, target, sentences[source], sentences[target]


def author_candidates(papers, base_rows, references, partition):
    from academic_rewriter.prepare import normalize, numbers

    forbidden, held_out, seen = isolation_keys(base_rows, references, partition)
    candidates = []
    for doc_id, paper in sorted(papers.items()):
        if document_key(doc_id) in forbidden:
            continue
        for source_id, target_id, (start, raw_source), (end, raw_target) in aligned_revisions(paper):
            if raw_source == raw_target:
                continue
            source, target = normalize(raw_source), normalize(raw_target)
            n, m = len(source.split()), len(target.split())
            if source == target or not (5 <= n <= 100 and 5 <= m <= 100 and .5 <= m / n <= 2):
                continue
            if EXTRACTION_ARTIFACTS.search(source) or EXTRACTION_ARTIFACTS.search(target):
                continue
            if (numbers(source) != numbers(target)
                    or Counter(NUMERIC_PART.findall(source)) != Counter(NUMERIC_PART.findall(target))):
                continue
            if Counter(PLACEHOLDER.findall(source)) != Counter(PLACEHOLDER.findall(target)):
                continue
            terms = {word.casefold() for word in ACRONYM.findall(source + ' ' + target) if word not in PLACEHOLDER_NAMES}
            protected = terms | PROTECTED_WORDS
            before, after = WORD.findall(source.casefold()), WORD.findall(target.casefold())
            if Counter(word for word in before if word in protected) != Counter(word for word in after if word in protected):
                continue
            similarity = SequenceMatcher(None, before, after, autojunk=False).ratio()
            if similarity < MIN_WORD_OVERLAP:
                continue
            key = (sha256(' '.join(before).encode('utf8')), sha256(' '.join(after).encode('utf8')))
            if key in seen or held_out.intersection((*key, normalized_hash(raw_source), normalized_hash(raw_target))):
                continue
            licenses = paper.get('license', {})
            version_licenses = {str(v): licenses.get(str(v)) for v in (start, end)}
            if not all(version_licenses.values()):
                continue
            seen.add(key)
            identifier = sha256(('\0'.join((partition['revision'], doc_id, source_id, target_id))).encode('utf8'))[:24]
            candidates.append({
                'id': 'arxivedits-aligned-' + identifier, 'doc_id': doc_id, 'group_id': document_key(doc_id),
                'source': source, 'target': target, 'split': 'train', 'upstream_split': 'train',
                'synthetic': False, 'training_only': True, 'human_verified': False,
                'review_status': 'automatically_screened', 'source_corpus': 'arXivEdits',
                'edit_kind': 'punctuation_or_case' if before == after else 'word_change',
                'sentence_ids': [source_id, target_id], 'versions': [start, end],
                'paper_url': paper.get('arxiv_link'), 'licenses': version_licenses, 'word_overlap': similarity,
                'source_sha256': sha256(raw_source.encode('utf8')), 'target_sha256': sha256(raw_target.encode('utf8')),
            })
    return candidates


def compatible_author_pairs(root, candidates, base_rows, config, base_real_count):
    from academic_rewriter.joint_words import word_replacements
    from academic_rewriter.lexical import read_lexical_training
    from academic_rewriter.run import build_subword_tokenizer
    from academic_rewriter.six_layer import SixLayerConfig
    from academic_rewriter.span_edits import encode_edit_program

    settings = config['settings']
    maximum = settings['max_length']
    word_rows, word_hash = read_lexical_training(root / settings['word_data'])
    grammar_bytes = (root / settings['grammar_data']).read_bytes()
    if word_hash != config['word_data_sha256'] or sha256(grammar_bytes) != config['grammar_data_sha256']:
        raise ValueError('The v8 word or grammar supplement fingerprint changed')
    grammar_rows = json_lines(grammar_bytes)
    base_train = base_rows['train']
    fixed_rows = base_train + base_rows['validation'] + base_rows['test'] + grammar_rows
    replacements = word_replacements(word_rows)

    def span_error(row, tokenizer, span_config):
        if any(len(tokenizer(row[field])['input_ids']) > maximum for field in ('source', 'target')):
            return f'source or target exceeds {maximum} tokens'
        try:
            encode_edit_program(row['source'], row['target'], tokenizer, span_config, maximum)
        except ValueError as error:
            return str(error)
        return None

    cache = root / '.cache'
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='lexifar-span-screen-', dir=cache) as temporary:
        while base_real_count + len(candidates) >= MIN_REAL_PAIRS:
            print(f'Checking v8 encoding with {len(candidates)} screened author pairs', flush=True)
            tokenizer = build_subword_tokenizer(base_train + candidates + word_rows + grammar_rows, Path(temporary))
            span_config = SixLayerConfig(
                vocab_size=len(tokenizer) + maximum + 1, max_positions=2 * maximum + 2,
                depth='joint', word_replacements=replacements, prediction_stages=4,
                fuse_stage_outputs=True, span_edits=True,
                pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            usable = [row for row in candidates if span_error(row, tokenizer, span_config) is None]
            if len(usable) == len(candidates):
                for row in fixed_rows:
                    issue = span_error(row, tokenizer, span_config)
                    if issue:
                        raise ValueError(f"Frozen sample {row['id']} fails v8 encoding: {issue}")
                return usable
            candidates = usable
    raise ValueError(f'Only {base_real_count + len(candidates)} real pairs fit the v8 corpus, which needs {MIN_REAL_PAIRS}')


def build(root, version, review_paths=None, *, alignment_path=None):
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

    base_rows = {split: json_lines(raw) for split, raw in data.items()}
    config_path = root / 'configs/six_layer_span_edits.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    counts = dict(base_manifest['counts'])
    if alignment_path is None:
        review_files = [path.read_bytes() for path in review_paths]
        additions, review_hashes = approved_additions(review_files, references, partition)
        if not additions:
            raise ValueError('No training pairs passed both independent reviews. No expanded dataset was created.')
        source_details = {
            'review_files_sha256': review_hashes,
            'approved_training': len(additions),
            'approved_documents': len({row['doc_id'] for row in additions}),
        }
        description = 'approved pairs'
    else:
        alignment = alignment_path.read_bytes()
        alignment_hash = sha256(alignment)
        if alignment_hash != partition['file_sha256'][ALIGNMENT_FILE]['sha256']:
            raise ValueError('Expected the pinned upstream arXivEdits sentence alignment train file')
        screened = author_candidates(json.loads(alignment), base_rows, references, partition)
        additions = compatible_author_pairs(root, screened, base_rows, config, counts['real_training'])
        source_details = {
            'alignment_sha256': alignment_hash,
            'screening_code_sha256': sha256(Path(__file__).read_bytes()),
            'minimum_word_overlap': MIN_WORD_OVERLAP,
            'screened_training_candidates': len(screened),
            'author_revision_training': len(additions),
            'author_revision_documents': len({row['group_id'] for row in additions}),
            'edit_kinds': dict(Counter(row['edit_kind'] for row in additions)),
            'approved_training': 0,
        }
        description = 'automatically screened author revisions'

    check_isolation(additions, base_rows, references, partition)
    extra = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in additions).encode('utf8')
    separator = b'' if data['train'].endswith(b'\n') else b'\n'
    data['train'] += separator + extra
    counts['real_training'] += len(additions)
    counts['total_training'] += len(additions)

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
            **source_details,
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
    print(f"Added {len(additions)} {description}. {counts['real_training']} real training pairs in {dataset_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--reviews', nargs=2, type=Path, metavar='JSONL',
                        help='Training review exports from two independent human curators')
    source.add_argument('--author-revisions', nargs='?', const=Path('.cache/arxivedits/train_alignment.json'),
                        type=Path, metavar='ALIGNMENTS', help='Build the local training study from pinned author revisions')
    parser.add_argument('--version', type=int, help='Use a new version number (reviews default to 2, author revisions to 3)')
    args = parser.parse_args()
    version = args.version if args.version is not None else (3 if args.author_revisions else 2)
    if version < 2:
        parser.error('--version must be at least 2')
    try:
        build(ROOT, version, args.reviews, alignment_path=args.author_revisions)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
