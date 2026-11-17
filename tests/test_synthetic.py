import json
from pathlib import Path
import tempfile
import unittest

from academic_rewriter.synthetic import DOMAINS, FRAMES, generate_pairs, prepare, validate_pair


class SyntheticTests(unittest.TestCase):
    def test_all_authored_frames_preserve_protected_markers(self):
        slots = dict(task='classification', model='classifier', data='dataset', metric='accuracy',
                     part='attention module', n=123, p=91, k=4, t=1.2, delta=2.1, memory=512)
        self.assertEqual(len(FRAMES), 48)
        for name, source, target in FRAMES:
            with self.subTest(frame=name):
                validate_pair(dict(source=source.format(**slots), target=target.format(**slots),
                                   split='train', synthetic=True, human_verified=False,
                                   protected_terms=['classifier'] if '{model}' in source else []))

    def test_reproducible_guarded_training_pairs_reject_corruption(self):
        rows = generate_pairs(250, seed=8)
        self.assertEqual(rows, generate_pairs(250, seed=8))
        self.assertEqual(len({row['id'] for row in rows}), 250)
        for row in rows:
            validate_pair(row)
        row = next(row for row in rows if any(character.isdigit() for character in row['source']))
        for invalid in [dict(row, split='test'), dict(row, human_verified=True),
                        dict(row, source=row['source']+' 999999'),
                        dict(row, source='not '+row['source'])]:
            with self.assertRaises(ValueError):
                validate_pair(invalid)
        forbidden = rows[0]['source']
        generated = generate_pairs(250, seed=8, forbidden_texts=[forbidden])
        self.assertNotIn(forbidden, [row['source'] for row in generated])

    def test_real_heldout_bytes_are_unchanged_and_never_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root/'original'
            original.mkdir()
            for split in ['train', 'validation', 'test']:
                row = dict(id=split, doc_id=split, source=split+' source sentence', target=split+' target sentence')
                (original/(split+'.jsonl')).write_bytes((json.dumps(row)+'\r\n').encode())
            output = root/'expanded'
            manifest = prepare(output, original, count=100)
            self.assertEqual(manifest['counts']['total_training'], 101)
            for split in ['validation', 'test']:
                self.assertEqual((original/(split+'.jsonl')).read_bytes(), (output/(split+'.jsonl')).read_bytes())
                self.assertFalse(json.loads((output/(split+'.jsonl')).read_text()).get('synthetic', False))
            with self.assertRaisesRegex(ValueError, 'versioned'):
                prepare(output, original, count=100)


if __name__ == '__main__':
    unittest.main()
