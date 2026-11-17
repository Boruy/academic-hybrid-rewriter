"""Generate training-only, hypothetical academic formalization pairs.

These are model-authored semantic-frame templates, not harvested paper text or
independent human examples. Both sides are rendered from the same slots. This
provides auditable weak supervision, not evidence of real-world writing quality.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from .prepare import numbers

GENERATOR = 'codex-semantic-frames-v1'
DOMAINS = [
    ('classification', 'classifier', 'dataset', 'accuracy'),
    ('segmentation', 'segmentation model', 'image dataset', 'Dice score'),
    ('retrieval', 'retrieval model', 'document collection', 'recall'),
    ('translation', 'translation model', 'parallel corpus', 'translation quality'),
    ('forecasting', 'forecasting model', 'time series dataset', 'forecast accuracy'),
    ('speech recognition', 'speech recognition model', 'speech corpus', 'recognition accuracy'),
    ('anomaly detection', 'anomaly detector', 'sensor dataset', 'detection accuracy'),
    ('question answering', 'question answering model', 'question dataset', 'answer accuracy'),
    ('object detection', 'object detector', 'image collection', 'average precision'),
    ('recommendation', 'recommendation model', 'interaction dataset', 'ranking quality'),
    ('summarization', 'summarization model', 'document dataset', 'summary quality'),
    ('clustering', 'clustering method', 'feature dataset', 'cluster quality'),
    ('optimization', 'optimization method', 'benchmark suite', 'solution quality'),
    ('registration', 'registration method', 'scan dataset', 'alignment accuracy'),
    ('denoising', 'denoising model', 'signal dataset', 'reconstruction quality'),
    ('imputation', 'imputation model', 'tabular dataset', 'imputation accuracy'),
    ('calibration', 'calibration method', 'validation dataset', 'calibration quality'),
    ('tracking', 'tracking model', 'video dataset', 'tracking accuracy'),
    ('parsing', 'parsing model', 'treebank', 'parsing accuracy'),
    ('entity recognition', 'entity recognition model', 'annotated corpus', 'entity recognition accuracy'),
    ('compression', 'compression method', 'file collection', 'compression ratio'),
    ('graph prediction', 'graph prediction model', 'graph dataset', 'prediction accuracy'),
    ('image reconstruction', 'reconstruction model', 'image dataset', 'reconstruction accuracy'),
    ('quality estimation', 'quality estimation model', 'evaluation dataset', 'estimation accuracy'),
]

# Each pair describes the same hypothetical observation or procedure. No result
# is represented as a claim from a real publication. Shorter words and active
# voice are retained where appropriate; elaborate vocabulary is not the goal.
FRAMES = [
    ('evaluate_samples', 'We tried out the {model} on {n} samples from the {data}.',
     'We evaluated the {model} on {n} samples from the {data}.'),
    ('inspect_errors', 'We looked at {n} errors to find out why the {model} failed.',
     'We examined {n} errors to identify why the {model} failed.'),
    ('compare_baseline', 'We checked how the {model} did against the baseline on {n} samples.',
     'We compared the {model} with the baseline on {n} samples.'),
    ('observe_accuracy', 'The {model} got an accuracy of {p}% on the {data}.',
     'The {model} achieved an accuracy of {p}% on the {data}.'),
    ('observe_runtime', 'The {model} took {t} seconds to process {n} samples.',
     'Processing {n} samples with the {model} required {t} seconds.'),
    ('evaluate_noise', 'We added {p}% noise to see how the {model} would do.',
     'We added {p}% noise to evaluate the performance of the {model}.'),
    ('repeat_runs', 'We ran the {task} experiment {k} times to check how much the results changed.',
     'We repeated the {task} experiment {k} times to assess variation in the results.'),
    ('possible_gain', 'The {model} may help make {metric} better on the {data}.',
     'The {model} may improve {metric} on the {data}.'),
    ('possible_limit', 'The {model} might not work well when only {n} samples are available.',
     'The {model} might not perform well when only {n} samples are available.'),
    ('retain_uncertainty', 'The results do not show that the {model} will work in every setting.',
     'The results do not establish that the {model} will perform well in every setting.'),
    ('sample_selection', 'We picked {n} samples at random from the {data}.',
     'We randomly selected {n} samples from the {data}.'),
    ('remove_duplicates', 'We took out {n} repeated entries from the {data}.',
     'We removed {n} duplicate entries from the {data}.'),
    ('missing_observations', 'We left out {n} samples because their labels were missing in the {data}.',
     'We excluded {n} samples with missing labels from the {data}.'),
    ('collect_measurements', 'We wrote down the {metric} of the {model} after each run.',
     'We recorded the {metric} of the {model} after each run.'),
    ('shared_settings', 'We kept the {task} settings the same for all {k} runs.',
     'We used identical {task} settings across all {k} runs.'),
    ('hyperparameter_choice', 'We picked the {model} settings using the validation results, not the test results.',
     'We selected the {model} settings based on validation results, not test results.'),
    ('report_mean', 'We worked out the mean {metric} over {k} runs of the {model}.',
     'We calculated the mean {metric} across {k} runs of the {model}.'),
    ('threshold', 'We set the threshold to {p}% before checking the {model} results.',
     'We fixed the threshold at {p}% before examining the {model} results.'),
    ('ablation', 'We took out the {part} to see what it did for the {model}.',
     'We removed the {part} to assess its contribution to the {model}.'),
    ('decrease', 'The {model} made {metric} drop by {delta} percentage points.',
     'The {model} reduced {metric} by {delta} percentage points.'),
    ('increase', 'The {model} made {metric} go up by {delta} percentage points.',
     'The {model} increased {metric} by {delta} percentage points.'),
    ('limited_evidence', 'These {n} samples are not enough to say the {model} is better than the baseline.',
     'These {n} samples are not sufficient to establish that the {model} outperforms the baseline.'),
    ('further_evaluation', 'We need to do more tests before using the {model} in other settings.',
     'Further evaluation is required before applying the {model} to other settings.'),
    ('runtime_comparison', 'The {model} was faster than the baseline by {t} seconds on the {data}.',
     'The runtime of the {model} on the {data} was {t} seconds shorter than that of the baseline.'),
    ('memory', 'The {model} used {memory} MB of memory during {task}.',
     'Memory consumption of the {model} during {task} was {memory} MB.'),
    ('batch_size', 'We put {k} samples in each batch when training the {model}.',
     'We trained the {model} with a batch size of {k} samples.'),
    ('training_budget', 'We stopped training the {model} after {n} updates.',
     'Training of the {model} was terminated after {n} updates.'),
    ('measurement_scope', 'We only checked the {model} on the {data}, so the results may not hold elsewhere.',
     'We evaluated the {model} only on the {data}, so the results may not generalize to other settings.'),
    ('error_categories', 'We split the {n} errors from the {model} into {k} groups.',
     'We categorized the {n} errors from the {model} into {k} groups.'),
    ('excluded_inference', 'The {model} results do not tell us whether the difference is statistically significant.',
     'The {model} results do not establish whether the difference is statistically significant.'),
    ('consistent_observation', 'We saw the same {metric} trend in all {k} runs of the {model}.',
     'The {metric} trend was consistent across all {k} runs of the {model}.'),
    ('dataset_scope', 'The {data} has {n} samples, and we used all of them for {task}.',
     'The {data} contains {n} samples, all of which were used for {task}.'),
    ('prior_work', 'Earlier work on {task} did not look at the {part}.',
     'Previous research on {task} did not examine the {part}.'),
    ('negative_result', 'Adding the {part} did not make the {model} better on the {data}.',
     'Adding the {part} did not improve the performance of the {model} on the {data}.'),
    ('relationship', 'The {model} got worse as the amount of noise went up.',
     'The performance of the {model} declined as the amount of noise increased.'),
    ('possibility', 'The {part} could help the {model} handle noisy inputs.',
     'The {part} could improve the ability of the {model} to handle noisy inputs.'),
    ('stability', 'The {model} gave almost the same {metric} in {k} runs.',
     'The {model} produced nearly identical {metric} across {k} runs.'),
    ('contrast', 'Even with {n} samples, the {model} did not beat the baseline.',
     'The {model} did not outperform the baseline despite using {n} samples.'),
    ('qualification', 'The {model} works better only when the {part} is present.',
     'The {model} performs better only in the presence of the {part}.'),
    ('reproducibility', 'We kept the {task} code and settings so others can run the experiment again.',
     'We retained the {task} code and settings to enable replication of the experiment.'),
    ('scope_of_claim', 'We are not saying that the {model} is the best choice for every {task} dataset.',
     'We do not claim that the {model} is the optimal choice for every {task} dataset.'),
    ('data_comparison', 'We checked the {model} on both the {data} and a separate dataset.',
     'We evaluated the {model} on both the {data} and a separate dataset.'),
    ('measurement_process', 'We measured how long the {model} took after {k} warm-up runs.',
     'We measured the runtime of the {model} after {k} warm-up runs.'),
    ('manual_inspection', 'We looked through {n} outputs to find problems with the {model}.',
     'We inspected {n} outputs to identify problems with the {model}.'),
    ('confidence', 'The {model} results suggest a possible improvement, but they do not prove it.',
     'The {model} results suggest a potential improvement, but they do not establish it conclusively.'),
    ('procedure', 'We first cleaned the {data} and then trained the {model}.',
     'We cleaned the {data} before training the {model}.'),
    ('inclusion', 'We used only samples with labels from the {data} to train the {model}.',
     'Training of the {model} used only labeled samples from the {data}.'),
    ('future_work', 'We plan to check whether the {part} helps the {model} on larger datasets.',
     'Future work will evaluate whether the {part} improves the {model} on larger datasets.'),
]
PARTS = ['attention module', 'normalization layer', 'input filter', 'feature extractor',
         'regularization term', 'sampling procedure', 'context module', 'gating mechanism']
GUARD_WORDS = ['not', 'no', 'may', 'might', 'could', 'only', 'every', 'all', 'both']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_key(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def validate_pair(row):
    if row.get('split') != 'train' or row.get('synthetic') is not True:
        raise ValueError('Synthetic pairs are training-only')
    if row.get('human_verified') is not False:
        raise ValueError('Generated pairs cannot claim independent human approval')
    if not row['source'].strip() or text_key(row['source']) == text_key(row['target']):
        raise ValueError('Pair must contain a meaningful text change')
    if numbers(row['source']) != numbers(row['target']):
        raise ValueError('Numeric quantities changed')
    # Guard propositions using explicit frame metadata and high-risk markers.
    # Synonymous rendering may use neither marker (e.g. almost / nearly).
    for word in GUARD_WORDS:
        counts = [len(re.findall(r'\b' + word + r'\b', row[k], re.I)) for k in ['source', 'target']]
        if counts[0] != counts[1]:
            raise ValueError('Protected scope/negation/uncertainty marker changed: ' + word)
    for term in row['protected_terms']:
        if row['source'].casefold().count(term.casefold()) != row['target'].casefold().count(term.casefold()):
            raise ValueError('Protected technical term changed: ' + term)


def generate_pairs(count=12000, seed=42, forbidden_texts=()):
    if count < 1:
        raise ValueError('Positive pair count required')
    rng = random.Random(seed)
    forbidden = {text_key(text) for text in forbidden_texts}
    seen, rows = set(), []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise ValueError('Unable to construct requested number of distinct guarded pairs')
        task, model, data, metric = rng.choice(DOMAINS)
        slots = dict(task=task, model=model, data=data, metric=metric, part=rng.choice(PARTS),
                     n=rng.randint(50, 5000), p=rng.randint(60, 99), k=rng.randint(2, 20),
                     t=round(rng.uniform(0.2, 50), 2), delta=round(rng.uniform(0.1, 8), 1),
                     memory=rng.choice([128, 256, 512, 768, 1024, 1536, 2048]))
        frames = rng.sample(FRAMES, 2 if rng.random() < 0.3 else 1)
        source = ' '.join(frame[1].format(**slots) for frame in frames)
        target = ' '.join(frame[2].format(**slots) for frame in frames)
        identity = hashlib.sha256((source + '\0' + target).encode()).hexdigest()
        if identity in seen or text_key(source) in forbidden or text_key(target) in forbidden:
            continue
        protected = [term for term in [model, data, metric, slots['part']]
                     if term.casefold() in source.casefold()]
        row = dict(id='syn-v1-' + identity[:20], doc_id='synthetic-v1/' + task.replace(' ', '_'),
                   split='train', source=source, target=target, intent='academic_formalization',
                   synthetic=True, human_verified=False, generator=GENERATOR,
                   semantic_frames=[frame[0] for frame in frames], semantic_slots=slots,
                   protected_terms=protected, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   target_sha256=hashlib.sha256(target.encode()).hexdigest(),
                   verification='shared_semantic_frame_and_automatic_guards_only')
        try:
            validate_pair(row)
        except ValueError:
            continue
        seen.add(identity)
        rows.append(row)
    return rows


def prepare(output, original, count=12000, seed=42):
    output, original = Path(output), Path(original)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new versioned output directory; existing data is preserved')
    real = {split: [json.loads(line) for line in (original / (split + '.jsonl')).read_text(
        encoding='utf8').splitlines() if line.strip()] for split in ['train', 'validation', 'test']}
    holdout = [row[key] for split in ['validation', 'test'] for row in real[split]
               for key in ['source', 'target']]
    synthetic = generate_pairs(count, seed, holdout)
    training = [dict(row, split='train', synthetic=False, origin='existing_iterater_train')
                for row in real['train']] + synthetic
    random.Random(seed).shuffle(training)
    output.mkdir(parents=True)
    for split, rows in [('train', training), ('validation', real['validation']), ('test', real['test'])]:
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        (output / (split + '.jsonl')).write_text(content, encoding='utf8', newline='\n')
    # Preserve actual held-out bytes, not just decoded/re-encoded equivalents.
    for split in ['validation', 'test']:
        (output / (split + '.jsonl')).write_bytes((original / (split + '.jsonl')).read_bytes())
    manifest = dict(version='academic-pairs-v1', generator=GENERATOR, seed=seed,
                    task='Informal scientific expression to equivalent academic expression',
                    counts=dict(synthetic_training=len(synthetic), real_training=len(real['train']),
                                total_training=len(training), validation=len(real['validation']), test=len(real['test'])),
                    template_families=len(FRAMES), domains=len(DOMAINS),
                    frame_usage=dict(Counter(frame for row in synthetic for frame in row['semantic_frames'])),
                    synthetic_human_verified=0, source='Model-authored hypothetical semantic frames; no new paper text',
                    data_sha256={split: digest(output / (split + '.jsonl')) for split in real},
                    original_data_sha256={split: digest(original / (split + '.jsonl')) for split in real},
                    generator_code_sha256=digest(__file__),
                    limitations=[
                        'Synthetic examples are training-only; no independent human verification is claimed.',
                        'Template variation is not equivalent to independent real examples or natural diversity.',
                        'All numeric findings are hypothetical, not claims about real methods or experiments.',
                        'Original validation/test bytes are unchanged; these inspected sets are diagnostics, not fresh confirmation.',
                        'Automatic guards and shared slots do not prove semantic equivalence.',
                        'The 491 existing revision pairs are mixed-intent supervision, not uniformly academic-style positives.'
                    ])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/academic-pairs-v1')
    parser.add_argument('--original', default='data/processed')
    parser.add_argument('--count', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.original, args.count, args.seed), indent=2))


if __name__ == '__main__':
    main()
