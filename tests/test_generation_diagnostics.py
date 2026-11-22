"""Regression checks for generation-budget diagnostics, independent of a trained model."""
import unittest
from pathlib import Path
from academic_rewriter.test import generation_diagnostics, parser, evaluation_output_path

class Tokenizer:
    pad_token_id = 0
    def __call__(self, text):
        return {'input_ids': [1] + text.split() + [2]}

class GenerationDiagnosticsTests(unittest.TestCase):
    def test_cap_hit_excludes_initial_bos_and_batch_padding(self):
        # The first BOS is an output marker; trailing zeroes are batch padding.
        d = generation_diagnostics([1, 1, 8, 9, 2, 0, 0], Tokenizer(), 'one two', 'one', 4, 4)
        self.assertEqual(d['generated_tokens'], 4)
        self.assertTrue(d['generation_limit_reached'])
        self.assertFalse(d['input_truncated'])
        self.assertFalse(d['target_truncated'])
        early = generation_diagnostics([1, 1, 8, 2, 0], Tokenizer(), 'one two three', 'one two three', 4, 8)
        self.assertEqual(early['generated_tokens'], 3)
        self.assertFalse(early['generation_limit_reached'])
        self.assertTrue(early['input_truncated'])
        self.assertTrue(early['target_truncated'])

    def test_default_budget_and_explicit_override(self):
        self.assertEqual(parser().parse_args(['--checkpoint', 'example']).max_new_tokens, 192)
        self.assertEqual(parser().parse_args(['--checkpoint', 'example', '--max-new-tokens', '48']).max_new_tokens, 48)

    def test_output_directory_records_budget_and_honors_override(self):
        args = parser().parse_args(['--checkpoint', 'example'])
        self.assertEqual(evaluation_output_path(args), Path('example/evaluation/test-192'))
        args.max_new_tokens = 48
        self.assertEqual(evaluation_output_path(args), Path('example/evaluation/test-48'))
        args.no_word_guidance = True
        self.assertEqual(evaluation_output_path(args), Path('example/evaluation/test-48-no-word-guidance'))
        args.output = 'custom/results'
        self.assertEqual(evaluation_output_path(args), Path('custom/results'))

if __name__ == '__main__':
    unittest.main()
