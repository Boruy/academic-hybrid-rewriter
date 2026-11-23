import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.expand_academic_pairs import approved_additions, author_candidates, build, normalized_hash, sha256


def encode(value):
    return (json.dumps(value) + '\r\n').encode('utf8')


class ReviewedExpansionTests(unittest.TestCase):
    def setUp(self):
        self.text = {'source': 'We made use of the observations.', 'target': 'We used the observations.'}
        self.reference = {'id': 'new-train-1', 'doc_id': 'new-paper', 'split': 'train', 'task_candidate': True}
        for field, text in self.text.items():
            self.reference[field + '_sha256'] = sha256(text.encode('utf8'))
            self.reference['normalized_' + field + '_sha256'] = normalized_hash(text)
        self.partition = {'revision': 'fixture', 'documents': {'new-paper': 'train'},
                          'previously_inspected_documents': []}

    def review(self, curator, **changes):
        return encode({**self.reference, **self.text, 'rater_id': curator, 'reviewer_type': 'human',
                       'category': 'Academic Style', 'meaning': 5, 'fluency': 4, 'source_fluency': 4,
                       'source_suitable': False, 'source_needs_revision': True, 'target_suitable': True,
                       'major_fact_error': False, **changes})

    def additions(self, first, second):
        return approved_additions([first, second], [self.reference], self.partition)[0]

    def test_blank_or_disagreeing_review_adds_nothing(self):
        first = self.review('curator_1')
        for second in (self.review('curator_2', meaning=None), self.review('curator_2', major_fact_error=True)):
            with self.subTest(review=json.loads(second)):
                self.assertEqual(self.additions(first, second), [])

    def test_reviews_must_match_training_text_and_distinct_curators(self):
        first = self.review('curator_1')
        with self.assertRaisesRegex(ValueError, 'distinct curator'):
            self.additions(first, first)
        with self.assertRaisesRegex(ValueError, 'human reviewer'):
            self.additions(first, self.review('curator_2', reviewer_type='ai_assisted'))
        with self.assertRaisesRegex(ValueError, 'review text differs'):
            self.additions(first, self.review('curator_2', target='A different sentence.'))
        self.partition['documents']['new-paper'] = 'test'
        with self.assertRaisesRegex(ValueError, 'frozen train partition'):
            self.additions(first, self.review('curator_2'))

    def test_build_preserves_held_out_bytes_and_rejects_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def put(relative, data):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return path

            corpus = {}
            for split in ('train', 'validation', 'test'):
                row = {'id': split, 'doc_id': f'old-{split}', 'source': f'Original {split} sentence.',
                       'target': f'Edited {split} sentence.'}
                corpus[split] = encode(row)
                put(f'data/processed/{split}.jsonl', corpus[split])
                put(f'data/academic-pairs-v1/{split}.jsonl', corpus[split])
            fingerprints = {split: sha256(raw) for split, raw in corpus.items()}
            generator = put('src/academic_rewriter/synthetic.py', b'# frozen generator fixture\n')
            put('data/academic-pairs-v1/manifest.json', encode({
                'generator_code_sha256': sha256(generator.read_bytes()), 'data_sha256': fingerprints,
                'counts': {'real_training': 1, 'synthetic_training': 0, 'total_training': 1,
                           'validation': 1, 'test': 1},
            }))
            partition = put('data/curation/arxivedits/frozen_partition.json',
                            encode({**self.partition, 'previous_data_sha256': fingerprints}))
            put('data/curation/arxivedits/references.jsonl', encode(self.reference))
            put('configs/six_layer_span_edits.json', encode({'settings': {'span_edits': True}}))
            reviews = [put(f'.cache/curator_{index}.jsonl', self.review(f'curator_{index}')) for index in (1, 2)]

            build(root, 2, reviews)

            result = root / '.cache/academic-pairs-v2'
            for split in ('validation', 'test'):
                self.assertEqual((result / f'{split}.jsonl').read_bytes(), corpus[split])
            training = (result / 'train.jsonl').read_bytes()
            self.assertTrue(training.startswith(corpus['train']))
            addition = json.loads(training.splitlines()[-1])
            self.assertEqual(addition['id'], self.reference['id'])
            self.assertTrue(addition['human_verified'])
            self.assertFalse(addition['synthetic'])
            config = json.loads((result / 'experiment.json').read_bytes())
            self.assertEqual(config['data'], '.cache/academic-pairs-v2')
            self.assertEqual(config['output'], 'runs/six_layer_joint_v8_academic_pairs_v2')
            with self.assertRaises(FileExistsError):
                build(root, 2, reviews)

            changed = json.loads(partition.read_bytes())
            changed['previously_inspected_documents'].append(self.reference['doc_id'])
            partition.write_bytes(encode(changed))
            with self.assertRaisesRegex(ValueError, 'protected document'):
                build(root, 3, reviews)
            self.assertFalse((root / '.cache/academic-pairs-v3').exists())


class AuthorExpansionTests(unittest.TestCase):
    def test_author_screen_preserves_protected_content_and_partitions(self):
        lexical = ('Our method makes use of sparse operations for GPU training.',
                   'Our method uses sparse operations for GPU training.')
        punctuation = ('In this experiment the measured errors remained stable.',
                       'In this experiment, the measured errors remained stable.')
        held_out = 'We developed a novel method for comparing the observed scores.'
        pairs = [
            lexical,
            punctuation,
            ('The implementation processed 200 samples in the second experiment.',
             'The implementation processed 201 samples in the second experiment.'),
            ('Our algorithm does not depend on the choice of basis.',
             'Our algorithm does depend on the choice of basis.'),
            ('The GPU experiment uses a deterministic sampling strategy for comparison.',
             'The CPU experiment uses a deterministic sampling strategy for comparison.'),
            ('The resulting estimate in [EQUATION] agrees with the measured values.',
             'The resulting estimate agrees with the measured values.'),
            (held_out, 'We developed a method for comparing the observed scores.'),
        ]

        def paper(revisions):
            return {
                '1': {f'old-{i}': s for i, (s, _) in enumerate(revisions)},
                '2': {f'new-{i}': t for i, (_, t) in enumerate(revisions)},
                'alignment': [[f'old-{i}', f'new-{i}'] for i in range(len(revisions))],
                'license': {'1': 'fixture-license', '2': 'fixture-license'},
            }

        corpus = {'new-paper': paper(pairs), 'hep-ph-1234567': paper([lexical])}
        partition = {'documents': {'hep-ph/1234567': 'test'}, 'revision': 'fixture',
                     'previously_inspected_documents': []}
        base = {'train': [], 'validation': [{'source': held_out, 'target': held_out}], 'test': []}
        candidates = author_candidates(corpus, base, [], partition)

        self.assertEqual([(row['source'], row['target']) for row in candidates], [lexical, punctuation])
        self.assertEqual([row['edit_kind'] for row in candidates], ['word_change', 'punctuation_or_case'])
        self.assertTrue(all(row['split'] == 'train' and row['training_only'] for row in candidates))
        self.assertTrue(all(not row['synthetic'] and not row['human_verified'] for row in candidates))
        self.assertTrue(all(row['doc_id'] == 'new-paper' for row in candidates))


if __name__ == '__main__':
    unittest.main()
