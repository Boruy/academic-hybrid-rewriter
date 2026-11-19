import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from academic_rewriter.joint_words import WordTask, word_replacements
from academic_rewriter.rewrite import rewrite_neural
from academic_rewriter.run import batch, build_tokenizer, parser, read_rows, run
from academic_rewriter.six_layer import NEURAL_DEPTHS, SixLayerConfig, SixLayerRewriter
from academic_rewriter.test import parser as evaluation_parser, run as evaluate_checkpoint


WORDS = [
    dict(source='We get access to the archive.', target='We gain access to the archive.'),
    dict(source='We get estimates from the model.', target='We obtain estimates from the model.'),
    dict(source='The threads get stuck.', target='The threads get stuck.', focus='get'),
]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')


class JointRewriterTests(unittest.TestCase):
    def test_sentence_supervision_reaches_the_word_guidance_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(8)
            tokenizer = build_tokenizer(WORDS, Path(temporary))
            model = SixLayerRewriter(SixLayerConfig(
                vocab_size=len(tokenizer), d_model=16, attention_heads=2, ffn_dim=32,
                dropout=0., max_positions=48, depth='joint', word_replacements=word_replacements(WORDS)))
            task = WordTask(model, tokenizer, 'cpu', 20)
            inputs = batch(WORDS[:2], tokenizer, 'cpu', 20, task)
            guidance = inputs['word_guidance']
            self.assertGreater(guidance.sum().item(), 0.)
            self.assertEqual(guidance[:, 0].count_nonzero().item(), 0)  # Ignore the BOS position.
            losses = model.sequence_losses(**inputs)
            self.assertEqual(set(losses), {'medium', 'deep'})
            gradients = torch.autograd.grad(sum(losses.values()),
                                            (model.word_head[-1].weight, model.word_fusion[-1].weight))
            for gradient in gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.)
            with torch.no_grad():
                model.word_head[-1].weight.zero_()
                model.word_head[-1].bias.copy_(torch.tensor([0., 3., 1.]))
            source = 'We get  10 records from the get-log and the GET endpoint.'
            revised, edits = task.rewrite(source)
            self.assertEqual(revised, source.replace('get  ', 'gain  '))
            self.assertEqual(len(edits), 1)

    def test_cached_decoding_uses_both_stage_guides(self):
        torch.manual_seed(8)
        model = SixLayerRewriter(SixLayerConfig(
            vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint',
            word_replacements=word_replacements(WORDS))).eval()
        tokens = torch.tensor([[1, 4, 5, 2, 1, 6, 2]])
        mask = torch.ones_like(tokens)
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
        guidance = torch.zeros(1, tokens.size(1), 2)
        guidance[0, 1:3] = torch.tensor([[.5, .1], [.2, .7]])
        late_only = guidance.clone()
        late_only[..., 0] = 0
        with torch.no_grad():
            for depth, count in NEURAL_DEPTHS.items():
                with self.subTest(depth=depth):
                    expected = model.sequence_logits(tokens, mask, segments, depth=depth, word_guidance=guidance)
                    plain = model.sequence_logits(tokens, mask, segments, depth=depth)
                    self.assertFalse(torch.allclose(expected, plain))
                    later = model.sequence_logits(tokens, mask, segments, depth=depth, word_guidance=late_only)
                    if depth == 'deep':
                        self.assertFalse(torch.allclose(later, plain))
                    else:
                        torch.testing.assert_close(later, plain)
                    states = [None] * count
                    for position in range(tokens.size(1)):
                        actual, states = model.step(tokens[:, position], torch.tensor([position]),
                                                    segments[:, position], mask[:, position].bool(), states,
                                                    depth=depth, word_guidance=guidance[:, position])
                        torch.testing.assert_close(actual, expected[:, position], atol=4e-6, rtol=4e-5)

    def test_one_training_run_exports_all_modes_and_optional_word_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            words = root/'words.jsonl'
            reviewed = root/'reviewed.jsonl'
            output = root/'checkpoint'
            word_test = [dict(source='We get access to the records.', target='We gain access to the records.'),
                         dict(source='The workers get stuck.', target='The workers get stuck.', focus='get')]
            write_rows(words, [dict(row, split='train', synthetic=True) for row in WORDS])
            write_rows(reviewed, word_test)
            for split, examples in [('train', WORDS), ('validation', word_test), ('test', word_test)]:
                write_rows(root/(split+'.jsonl'), [dict(row, id=f'{split}-{i}') for i, row in enumerate(examples)])
            training = parser().parse_args(['--tiny', '--device', 'cpu', '--data', str(root),
                                            '--word-data', str(words), '--output', str(output), '--steps', '1',
                                            '--batch-size', '2', '--max-length', '24', '--checkpoint-selection', 'final'])
            flags = ['--checkpoint', str(output), '--data', str(root), '--device', 'cpu',
                     '--split', 'test', '--max-new-tokens', '8', '--limit', '0']
            with contextlib.redirect_stdout(io.StringIO()):
                trained = run(training)
                metrics = evaluate_checkpoint(evaluation_parser().parse_args(flags + ['--word-test', str(reviewed)]))
                with self.assertRaisesRegex(ValueError, 'separate, nonsynthetic'):
                    evaluate_checkpoint(evaluation_parser().parse_args(flags + ['--word-test', str(words)]))
            self.assertFalse(trained['test_set_used'])
            self.assertEqual(trained['rewrite_modes'], ['shallow', 'medium', 'deep'])
            record, = read_rows(output/'training_progress.jsonl')
            expected = .5*(record['medium_cross_entropy'] + record['deep_cross_entropy'])
            expected += training.word_loss_weight*record['word_cross_entropy']
            self.assertAlmostEqual(record['train_loss'], expected, places=5)
            self.assertTrue(metrics['test_set_used'])
            self.assertEqual(set(metrics['modes']), {'copy_baseline', 'shallow', 'medium', 'deep'})
            self.assertEqual(metrics['modes']['copy_baseline']['copy_rate'], 1.)
            self.assertEqual(metrics['word_test']['examples'], 2)
            report = output/'evaluation'/'test-8'
            predictions = read_rows(report/'predictions.jsonl')
            with (report/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual([row['source'] for row in comparison], [row['source'] for row in word_test])
            self.assertEqual(len(predictions), 2)
            for depth in trained['rewrite_modes']:
                column = f'{depth}_prediction'
                self.assertEqual([row[column] for row in comparison], [row[column] for row in predictions])
            single = rewrite_neural(word_test[0]['source'], 'shallow', output, 'cpu')
            self.assertEqual(single['prediction'], predictions[0]['shallow_prediction'])
