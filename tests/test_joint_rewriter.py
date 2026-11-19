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
    def test_preexisting_joint_checkpoints_keep_their_unfused_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'six_layer_config.json'
            config = dict(vocab_size=24, d_model=16, depth='joint', word_replacements=word_replacements(WORDS))
            for saved, layers in [(config, (2, 4, 6)), (dict(config, prediction_stages=4), (1, 2, 4, 6))]:
                with self.subTest(layers=layers):
                    path.write_text(json.dumps(saved), encoding='utf8')
                    model = SixLayerRewriter(SixLayerConfig.load(path))
                    self.assertEqual(model.prediction_stage_layers, layers)
                    self.assertIsNone(model.final_fusion)

    def test_final_loss_reaches_every_stage_projection_without_word_guides(self):
        torch.manual_seed(13)
        model = SixLayerRewriter(SixLayerConfig(
            vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint',
            prediction_stages=4, fuse_stage_outputs=True, word_replacements=word_replacements(WORDS)))
        source = torch.tensor([[1, 4, 2], [1, 5, 2]])
        targets = torch.tensor([[1, 6, 2], [1, 7, 2]])
        loss = model(source, torch.ones_like(source), targets).loss
        parameters = [model.mid_predictions[str(layer)][-1].weight for layer in model.prediction_stage_layers]
        parameters.append(model.final_fusion.weight)
        gradients = torch.autograd.grad(loss, parameters)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_sentence_supervision_reaches_the_word_guidance_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            torch.manual_seed(8)
            tokenizer = build_tokenizer(WORDS, Path(temporary))
            model = SixLayerRewriter(SixLayerConfig(
                vocab_size=len(tokenizer), d_model=16, attention_heads=2, ffn_dim=32,
                dropout=0., max_positions=48, depth='joint', prediction_stages=4,
                word_replacements=word_replacements(WORDS)))
            task = WordTask(model, tokenizer, 'cpu', 20)
            inputs = batch(WORDS[:2], tokenizer, 'cpu', 20, task)
            guidance = inputs['word_guidance']
            self.assertEqual(guidance.size(-1), 3)
            self.assertGreater(guidance.sum().item(), 0.)
            self.assertEqual(guidance[:, 0].count_nonzero().item(), 0)  # Ignore the BOS position.
            different_targets = [dict(row, target='Unrelated reference.') for row in WORDS[:2]]
            torch.testing.assert_close(batch(different_targets, tokenizer, 'cpu', 20, task)['word_guidance'],
                                       guidance)
            earlier_stage_weights = tuple(model.mid_predictions[layer][0].weight for layer in ('1', '2'))
            stage_gradients = torch.autograd.grad(guidance[..., 2].sum(), earlier_stage_weights, retain_graph=True)
            for gradient in stage_gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.)
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

    def test_cached_decoding_uses_the_guides_for_its_stages(self):
        torch.manual_seed(8)
        model = SixLayerRewriter(SixLayerConfig(
            vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint', prediction_stages=4,
            fuse_stage_outputs=True, word_replacements=word_replacements(WORDS))).eval()
        tokens = torch.tensor([[1, 4, 5, 2, 1, 6, 2]])
        mask = torch.ones_like(tokens)
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
        guidance = torch.zeros(1, tokens.size(1), 3)
        guidance[0, 1:3] = torch.tensor([[.5, .1, .3], [.2, .7, .4]])
        with torch.no_grad():
            for depth, count in NEURAL_DEPTHS.items():
                with self.subTest(depth=depth):
                    expected = model.sequence_logits(tokens, mask, segments, depth=depth, word_guidance=guidance)
                    plain = model.sequence_logits(tokens, mask, segments, depth=depth)
                    self.assertFalse(torch.allclose(expected, plain))
                    for layer, channel in model.word_guide_channels.items():
                        only = torch.zeros_like(guidance)
                        only[..., channel] = guidance[..., channel]
                        isolated = model.sequence_logits(tokens, mask, segments, depth=depth, word_guidance=only)
                        if layer < count:
                            self.assertFalse(torch.allclose(isolated, plain))
                        else:
                            torch.testing.assert_close(isolated, plain)
                    states = [None] * count
                    for position in range(tokens.size(1)):
                        actual, states = model.step(tokens[:, position], torch.tensor([position]),
                                                    segments[:, position], mask[:, position].bool(), states,
                                                    depth=depth, word_guidance=guidance[:, position])
                        torch.testing.assert_close(actual, expected[:, position], atol=4e-6, rtol=4e-5)
        with self.assertRaisesRegex(ValueError, 'only the fused final rewrite'):
            model.generate(tokens, mask, depth='medium', max_new_tokens=1)

    def test_one_training_run_exports_only_the_final_rewrite_and_optional_word_scores(self):
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
            self.assertEqual(trained['rewrite_modes'], ['final'])
            self.assertTrue(trained['fuse_stage_outputs'])
            self.assertEqual(trained['validation_cross_entropy'], trained['validation_cross_entropy_by_depth']['deep'])
            self.assertEqual(trained['prediction_stage_layers'], [1, 2, 4, 6])
            self.assertEqual(metrics['prediction_stage_layers'], [1, 2, 4, 6])
            record, = read_rows(output/'training_progress.jsonl')
            word_losses = [record[f'p{index}_word_cross_entropy'] for index in (1, 2, 3)]
            self.assertAlmostEqual(record['word_cross_entropy'], sum(word_losses)/3, places=5)
            expected = .5*(record['medium_cross_entropy'] + record['deep_cross_entropy'])
            expected += training.word_loss_weight*record['word_cross_entropy']
            self.assertAlmostEqual(record['train_loss'], expected, places=5)
            self.assertTrue(metrics['test_set_used'])
            self.assertNotIn('modes', metrics)
            self.assertEqual(metrics['rewrite_depth'], 'final')
            self.assertEqual(metrics['copy_baseline']['copy_rate'], 1.)
            self.assertEqual(metrics['word_test']['examples'], 2)
            report = output/'evaluation'/'test-8'
            predictions = read_rows(report/'predictions.jsonl')
            with (report/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ['id', 'source', 'prediction', 'reference'])
                comparison = list(reader)
            self.assertEqual([row['source'] for row in comparison], [row['source'] for row in word_test])
            self.assertEqual(len(predictions), 2)
            self.assertEqual([row['prediction'] for row in comparison], [row['prediction'] for row in predictions])
            for row in predictions:
                self.assertEqual({key for key in row if 'prediction' in key}, {'prediction'})
            single = rewrite_neural(word_test[0]['source'], None, output, 'cpu', max_new_tokens=8)
            self.assertEqual(single['depth'], 'final')
            self.assertNotIn('candidate', single)
            self.assertEqual(single['prediction'], single['source'] if single['fallback'] else predictions[0]['prediction'].strip())
            with self.assertRaisesRegex(ValueError, 'only the fused final rewrite'):
                rewrite_neural(word_test[0]['source'], 'shallow', output, 'cpu')
