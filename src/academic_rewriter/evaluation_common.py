"""Shared, dependency-light evaluation validation and paired statistics."""
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def words(text):
    return re.findall(r'\w+', text.lower())


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def load_aligned(project, config):
    data = read_jsonl(project/config['data'])
    ids = [r['id'] for r in data]
    if not data or len(ids) != len(set(ids)):
        raise ValueError('Evaluation data must have unique, nonempty case IDs')
    if any(not r.get('doc_id') or not r.get('group_id') for r in data):
        raise ValueError('doc_id and group_id are required for grouped evaluation')
    status=config.get('evaluation_status','inspected_test_diagnostic')
    if status not in {'inspected_test_diagnostic','fresh_confirmatory'}:
        raise ValueError('Declare inspected_test_diagnostic or fresh_confirmatory explicitly')
    if status=='fresh_confirmatory':
        old_docs=set();old_groups=set();old_texts=set()
        for split in ['train','validation','test']:
            for row in read_jsonl(project/'data/processed'/f'{split}.jsonl'):
                old_docs.add(row['doc_id']);old_groups.add(row['group_id'])
                old_texts.update(' '.join(row[k].lower().split()) for k in ['source','target'])
        if any(row['doc_id'] in old_docs or row['group_id'] in old_groups or any(' '.join(row[k].lower().split()) in old_texts for k in ['source','target']) for row in data):
            raise ValueError('Fresh confirmatory cases overlap training, validation or the inspected test set')
    candidates = {}
    labels = {}
    for entry in config['models']:
        key = entry['id']
        if key in candidates or key in {'copy_source', 'human_reference'}:
            raise ValueError('Duplicate/reserved model ID')
        rows = read_jsonl(project/entry['predictions'])
        if len(rows) != len(data) or {r['id'] for r in rows} != set(ids) or len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'{key}: incomplete or duplicate prediction alignment')
        by_id = {r['id']: r for r in rows}
        ordered = []
        for case in data:
            row = by_id[case['id']]
            if row['source'] != case['source'] or row['reference'] != case['target']:
                raise ValueError(f'{key}: source/reference mismatch for {case["id"]}')
            if not isinstance(row['prediction'], str):
                raise ValueError('Predictions must be strings')
            ordered.append(row['prediction'])
        candidates[key] = ordered
        labels[key] = entry['label']
    for control in config.get('controls', []):
        if control not in {'copy_source', 'human_reference'}:
            raise ValueError('Unknown control')
        candidates[control] = [r['source' if control=='copy_source' else 'target'] for r in data]
        labels[control] = 'Copy source (control)' if control=='copy_source' else 'Human reference (control)'
    return data, candidates, labels


def percentile(values, q):
    values = sorted(values)
    k = (len(values)-1)*q
    low = int(k)
    return values[low]*(1-(k-low))+values[min(low+1,len(values)-1)]*(k-low)


def clustered_interval(values, groups, samples=1000, seed=42):
    """Percentile CI by resampling document groups, keeping within-group pairs intact."""
    if samples<1: raise ValueError('At least one bootstrap sample required')
    if len(values) != len(groups) or not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError('Finite, complete aligned values/groups required')
    by_group = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    if len(by_group) < 2:
        return dict(mean=statistics.mean(values), ci95=None, groups=len(by_group), reason='Fewer than two groups')
    rng = random.Random(seed)
    keys = sorted(by_group)
    means = []
    for _ in range(samples):
        drawn = [v for key in rng.choices(keys,k=len(keys)) for v in by_group[key]]
        means.append(statistics.mean(drawn))
    return dict(mean=statistics.mean(values), ci95=[percentile(means,.025),percentile(means,.975)],
                groups=len(keys),bootstrap_samples=samples,seed=seed,unit='connected document group')


def paired_difference(a, b, groups, samples=1000, seed=42):
    if len(a) != len(b):
        raise ValueError('Paired comparison requires identical cases')
    return clustered_interval([x-y for x,y in zip(a,b)],groups,samples,seed)


def quadratic_kappa(a, b, low=1, high=5):
    if len(a) != len(b) or not a or any(v<low or v>high for v in a+b):
        raise ValueError('Invalid ordinal ratings')
    n = len(a)
    observed = sum((x-y)**2 for x,y in zip(a,b))/n
    expected = sum((x-y)**2 for x in a for y in b)/(n*n)
    return None if expected==0 else 1-observed/expected
