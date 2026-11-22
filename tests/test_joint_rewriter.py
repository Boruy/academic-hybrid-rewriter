import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from academic_rewriter.joint_words import WordTask, changed_source_spans, word_replacements
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
    def test_preexisting_joint_checkpoints_keep_their_saved_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'six_layer_config.json'
            config = dict(vocab_size=24, d_model=16, depth='joint', word_replacements=word_replacements(WORDS))
            saved_layouts = [
                (config, (2, 4, 6), False),
                (dict(config, prediction_stages=4), (1, 2, 4, 6), False),
                (dict(config, prediction_stages=4, fuse_stage_outputs=True), (1, 2, 4, 6), True),
            ]
            for saved, layers, fused in saved_layouts:
                with self.subTest(layers=layers, fused=fused):
                    path.write_text(json.dumps(saved), encoding='utf8')
                    model = SixLayerRewriter(SixLayerConfig.load(path))
                    self.assertEqual(model.prediction_stage_layers, layers)
                    self.assertEqual(model.final_fusion is not None, fused)
                    self.assertFalse(model.config.candidate_guidance)
                    self.assertFalse(model.config.contextual_edits)
                    self.assertNotIn('word_candidate_embeddings', model.state_dict())
                    self.assertIsNone(model.copy_head)

    def test_candidate_guide_weights_replacements_at_their_subword_positions(self):
        model = SixLayerRewriter(SixLayerConfig(
            vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, depth='joint',
            prediction_stages=4, candidate_guidance=True, word_replacements=word_replacements(WORDS)))
        probabilities = torch.tensor([[.25, .5, .25], [.25, .25, .5], [1., 0., 0.]])
        rows = torch.tensor([0, 0, 1])
        spans = torch.tensor([[0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        text_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            model.word_candidate_embeddings[0, 0] = 2.
            model.word_candidate_embeddings[1, 1] = 4.
            guide = model._make_word_guide(probabilities.log(), rows, spans, text_mask)

        self.assertEqual(guide.shape, (2, 7, 17))
        torch.testing.assert_close(guide[0, [1, 4], 0], torch.full((2,), .75))
        self.assertGreater(guide[0, 3, 0].item(), 0.)  # Boundary guidance can reach a neighbor.
        expected = torch.zeros(2, 7, 16)
        expected[0, 1:3, :2] = torch.tensor([1., 1.])
        expected[0, 4, :2] = torch.tensor([.5, 2.])
        torch.testing.assert_close(guide[..., 1:], expected)
        self.assertEqual(guide[1].count_nonzero().item(), 0)  # KEEP has no candidate contribution.

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
                fuse_stage_outputs=True, candidate_guidance=True,
                word_replacements=word_replacements(WORDS)))
            with torch.no_grad():
                model.word_candidate_embeddings[0, 0] = .1
                model.word_candidate_embeddings[1, 1] = .2
            task = WordTask(model, tokenizer, 'cpu', 20)
            inputs = batch(WORDS[:2], tokenizer, 'cpu', 20, task)
            guidance = inputs['word_guidance']
            self.assertEqual(guidance.shape, (*inputs['input_ids'].shape, 3, 1 + model.config.d_model))
            self.assertGreater(guidance[..., 0].sum().item(), 0.)
            self.assertGreater(guidance[..., 1:].count_nonzero().item(), 0)
            self.assertEqual(guidance[:, 0].count_nonzero().item(), 0)  # Ignore the BOS position.
            different_targets = [dict(row, target='Unrelated reference.') for row in WORDS[:2]]
            torch.testing.assert_close(batch(different_targets, tokenizer, 'cpu', 20, task)['word_guidance'],
                                       guidance)
            earlier_stage_weights = tuple(model.mid_predictions[layer][0].weight for layer in ('1', '2'))
            stage_gradients = torch.autograd.grad(guidance[..., 2, 0].sum(), earlier_stage_weights, retain_graph=True)
            for gradient in stage_gradients:
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0.)
            losses = model.sequence_losses(**inputs)
            self.assertEqual(set(losses), {'medium', 'deep'})
            gradients = torch.autograd.grad(losses['deep'], (model.word_head[-1].weight,
                                            model.word_fusion[-1].weight, model.word_candidate_embeddings))
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
        tokens = torch.tensor([[1, 4, 5, 2, 1, 6, 2]])
        mask = torch.ones_like(tokens)
        segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
        signals = torch.zeros(1, tokens.size(1), 3)
        signals[0, 1:3] = torch.tensor([[.5, .1, .3], [.2, .7, .4]])
        for candidate_guidance in (False, True):
            torch.manual_seed(8)
            model = SixLayerRewriter(SixLayerConfig(
                vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint',
                prediction_stages=4, fuse_stage_outputs=True, candidate_guidance=candidate_guidance,
                word_replacements=word_replacements(WORDS))).eval()
            guidance = signals
            if candidate_guidance:
                # Leave location scores at zero to isolate the candidate residual.
                guidance = signals.new_zeros((*signals.shape, 1 + model.config.d_model))
                guidance[..., 1] = signals
            with torch.no_grad():
                for depth, count in NEURAL_DEPTHS.items():
                    with self.subTest(depth=depth, candidate_guidance=candidate_guidance):
                        expected = model.sequence_logits(tokens, mask, segments, depth=depth, word_guidance=guidance)
                        plain = model.sequence_logits(tokens, mask, segments, depth=depth)
                        self.assertFalse(torch.allclose(expected, plain))
                        for layer, channel in model.word_guide_channels.items():
                            only = torch.zeros_like(guidance)
                            only[:, :, channel] = guidance[:, :, channel]
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

    def test_edit_supervision_covers_unlisted_words_without_leaking_references(self):
        real = dict(source='We feed features in LayoutLM.', target='We feed features into LayoutLM.')
        synthetic = dict(source='We call them as tails.', target='We call them tails.', synthetic=True)
        changed = changed_source_spans(real['source'], real['target'])
        self.assertEqual({real['source'][start:stop] for start, stop in changed}, {'in'})
        self.assertEqual(changed_source_spans('Each test is useful.', 'Each test is very useful.'), set())
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = build_tokenizer([real, synthetic], Path(temporary))
            model = SixLayerRewriter(SixLayerConfig(
                vocab_size=len(tokenizer), d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint',
                prediction_stages=4, fuse_stage_outputs=True, candidate_guidance=True, source_copy=True,
                contextual_edits=True, word_replacements=word_replacements(WORDS))).eval()
            task = WordTask(model, tokenizer, 'cpu', 32)
            sources = [real['source'], synthetic['source']]
            encoding = tokenizer(sources, return_offsets_mapping=True, return_tensors='pt', padding=True)
            guides, loss = task.supervised_guidance([real, synthetic], encoding)
            torch.testing.assert_close(guides, task.guidance(sources, encoding))
            in_position = real['source'].index(' in ') + 1
            (token_index,) = (encoding['offset_mapping'][0, :, 0] == in_position).nonzero().flatten()
            self.assertTrue((guides[0, token_index, :, -1] > 0).all())
            self.assertEqual(guides[..., 1:-1].count_nonzero().item(), 0)
            other_guides, other_loss = task.supervised_guidance([real, dict(synthetic, target='')], encoding)
            torch.testing.assert_close(other_guides, guides)
            torch.testing.assert_close(other_loss, loss)
            self.assertIsNone(task.supervised_guidance([dict(real, synthetic=True), synthetic], encoding)[1])
            gradient, = torch.autograd.grad(loss, (model.word_head[-1].weight,))
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient[task.edit_label].abs().sum().item(), 0.)

    def test_source_copy_is_causal_and_matches_cached_decoding(self):
        torch.manual_seed(8)
        model = SixLayerRewriter(SixLayerConfig(
            vocab_size=24, d_model=16, attention_heads=2, ffn_dim=32, dropout=0., depth='joint',
            prediction_stages=4, fuse_stage_outputs=True, source_copy=True, contextual_edits=True,
            word_replacements=word_replacements(WORDS))).eval()
        tokens = torch.tensor([[1, 4, 4, 7, 2, 1, 9, 2], [1, 2, 1, 8, 2, 0, 0, 0]])
        mask = tokens.ne(0)
        segments = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0, 0]])
        guidance = torch.zeros((*tokens.shape, 3, 2))
        guidance[0, 1, -1, :] = 1.
        guidance[0, 3, -1, 0] = 1.  # Edge enhancement alone leaves the copy bias unchanged.
        with torch.no_grad():
            expected = model.sequence_logits(tokens, mask, segments, word_guidance=guidance)
            states = [None] * 6
            for index in range(tokens.size(1)):
                valid = mask[:, index]
                actual, states = model.step(tokens[:, index], torch.full((2,), index),
                                            segments[:, index], valid, states, word_guidance=guidance[:, index])
                torch.testing.assert_close(actual[valid], expected[valid, index], atol=4e-6, rtol=4e-5)

            # Isolate E3 as the only difference between eligible copy positions.
            model.token_embedding.weight.zero_()
            model.copy_head.query.weight.zero_()
            model.copy_head.gate.weight.zero_()
            gate_bias = torch.tensor(12.)
            model.copy_head.gate.bias.fill_(gate_bias.item())
            probabilities = model.sequence_logits(tokens, mask, segments, word_guidance=guidance).exp()
        vocab_mass = (-gate_bias).sigmoid().item() / model.config.vocab_size
        copy_mass = gate_bias.sigmoid().item()
        # Only the first occurrence of token 4 has been seen at position 1.
        early = torch.full_like(probabilities[0, 1], vocab_mass)
        early[4] += copy_mass
        torch.testing.assert_close(probabilities[0, 1], early)
        # E3 downweights the first occurrence of token 4 during the rewrite.
        edit_weight = torch.exp(torch.tensor(-1.)).item()
        later = torch.full_like(probabilities[0, 6], vocab_mass)
        later[4] += copy_mass * (1 + edit_weight) / (2 + edit_weight)
        later[7] += copy_mass / (2 + edit_weight)
        torch.testing.assert_close(probabilities[0, 6], later)
        torch.testing.assert_close(probabilities[1, 3], torch.full_like(later, 1 / model.config.vocab_size))

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
                rows = [dict(row, id=f'{split}-{i}') for i, row in enumerate(examples)]
                if split == 'train':
                    rows[0]['synthetic'] = True
                write_rows(root/(split+'.jsonl'), rows)
            training = parser().parse_args(['--tiny', '--device', 'cpu', '--data', str(root),
                                            '--word-data', str(words), '--output', str(output), '--steps', '1',
                                            '--source-copy', '--contextual-edits', '--sampling', 'balanced',
                                            '--batch-size', '2', '--max-length', '24', '--checkpoint-selection', 'final'])
            flags = ['--checkpoint', str(output), '--data', str(root), '--device', 'cpu',
                     '--split', 'test', '--max-new-tokens', '8', '--limit', '0']
            with contextlib.redirect_stdout(io.StringIO()) as console:
                trained = run(training)
                metrics = evaluate_checkpoint(evaluation_parser().parse_args(flags + ['--word-test', str(reviewed)]))
                without_guidance = evaluate_checkpoint(evaluation_parser().parse_args(flags + ['--no-word-guidance']))
                with self.assertRaisesRegex(ValueError, 'separate, nonsynthetic'):
                    evaluate_checkpoint(evaluation_parser().parse_args(flags + ['--word-test', str(words)]))
            self.assertRegex(console.getvalue(), r'(?m)^\[train 1/1\] loss=\S+ ma20=\S+ '
                                                 r'p3_ce=\S+ p4_ce=\S+ word_ce=\S+ edit_ce=\S+ lr=\S+ elapsed=\S+$')
            self.assertFalse(trained['test_set_used'])
            self.assertEqual(trained['rewrite_modes'], ['final'])
            self.assertTrue(trained['fuse_stage_outputs'])
            self.assertTrue(trained['candidate_guidance'])
            self.assertTrue(metrics['candidate_guidance'])
            self.assertTrue(trained['source_copy'])
            self.assertTrue(metrics['source_copy'])
            self.assertTrue(trained['contextual_edits'])
            self.assertTrue(metrics['word_guidance_enabled'])
            self.assertFalse(without_guidance['word_guidance_enabled'])
            self.assertTrue((output/'evaluation'/'test-8-no-word-guidance'/'predictions.csv').is_file())
            self.assertEqual(trained['real_examples_seen'], 1)
            self.assertEqual(trained['synthetic_examples_seen'], 1)
            self.assertEqual(trained['validation_cross_entropy'], trained['validation_cross_entropy_by_depth']['deep'])
            self.assertEqual(trained['prediction_stage_layers'], [1, 2, 4, 6])
            self.assertEqual(metrics['prediction_stage_layers'], [1, 2, 4, 6])
            record, = read_rows(output/'training_progress.jsonl')
            word_losses = [record[f'p{index}_word_cross_entropy'] for index in (1, 2, 3)]
            self.assertAlmostEqual(record['word_cross_entropy'], sum(word_losses)/3, places=5)
            expected = .5*(record['medium_cross_entropy'] + record['deep_cross_entropy'])
            expected += training.word_loss_weight*(record['word_cross_entropy'] + record['source_edit_cross_entropy'])
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
