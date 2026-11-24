import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from academic_rewriter.joint_words import word_replacements
from academic_rewriter.rewrite import generate_predictions, load_checkpoint, rewrite_neural, span_fact_error
from academic_rewriter.run import build_subword_tokenizer, parser, read_rows, run
from academic_rewriter.six_layer import SixLayerConfig, SixLayerRewriter, SourceCopyHead
from academic_rewriter.span_edits import EditProgramDecoder, apply_edit_program, encode_edit_program, span_payload_weights
from academic_rewriter.test import parser as evaluation_parser, run as evaluate_checkpoint, sentence_metrics


WORDS = [dict(source='We get access.', target='We gain access.', synthetic=True, split='train')]


def config_for(tokenizer, *, span_edits=True):
    return SixLayerConfig(vocab_size=len(tokenizer) + (25 if span_edits else 0), d_model=16,
                          attention_heads=2, ffn_dim=32, dropout=0., max_positions=50,
                          depth='joint', prediction_stages=4, fuse_stage_outputs=True,
                          source_copy=True, contextual_edits=True, candidate_guidance=True,
                          span_edits=span_edits, word_replacements=word_replacements(WORDS))


class SpanEditTests(unittest.TestCase):
    def test_payload_weight_preserves_control_and_padding_weights(self):
        config = SixLayerConfig(vocab_size=13, d_model=16, max_positions=10, depth='joint',
                                prediction_stages=4, fuse_stage_outputs=True, span_edits=True,
                                word_replacements=word_replacements(WORDS))
        base = config.text_vocab_size
        labels = torch.tensor([[1, base, base + 1, 4, 2], [1, 2, -100, -100, -100]])
        expected = torch.tensor([[1., 1., 1., 2., 1.], [1., 1., 0., 0., 0.]])
        torch.testing.assert_close(span_payload_weights(labels, config, 2.), expected)

    def test_copy_scale_transfers_pointer_probability_to_vocabulary(self):
        config = SixLayerConfig(vocab_size=8, d_model=16)
        copy = SourceCopyHead(config)
        features = torch.zeros(1, 1, config.d_model)
        logits = torch.zeros(1, 1, config.vocab_size)
        tokens = torch.tensor([[4]])
        allowed = torch.ones(1, 1, 1, dtype=torch.bool)
        with torch.no_grad():
            copy.gate.bias.zero_()
            for scale in (1., .5, 0.):
                expected = torch.full_like(logits, (1. - .5 * scale) / config.vocab_size)
                expected[..., 4] += .5 * scale
                actual = copy(features, logits, features, tokens, allowed, copy_scale=scale).exp()
                torch.testing.assert_close(actual, expected)

    def test_reports_a_copied_edit_separately_from_retain(self):
        source = 'Keep this phrase.'
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = build_subword_tokenizer([dict(source=source, target=source)], Path(directory))
            config = config_for(tokenizer)
            base, bos, eos = config.text_vocab_size, config.bos_token_id, config.eos_token_id
            payload = tokenizer.encode('Keep', add_special_tokens=False)
            copied_edit = [bos, bos, base, base + 1, *payload, eos]
            retain = [bos, bos, eos] + [config.pad_token_id] * (len(copied_edit) - 3)
            model = SimpleNamespace(config=config, generate=lambda **kwargs: torch.tensor([retain, copied_edit]))
            inputs = tokenizer([source, source], return_tensors='pt')
            results = generate_predictions(model, tokenizer, [source, source], inputs, 24)
            self.assertEqual([row['prediction'] for row, _ in results], [source, source])
            self.assertEqual([row['edit_status'] for row, _ in results], ['retain', 'noop_edit'])

    def test_scores_final_edit_positions_against_the_reference(self):
        row = dict(source='We get access to those datasets.', reference='We gain access to the datasets.',
                   prediction='We gain access to those samples.', edit_status='edited',
                   edits=[dict(source='get', target='gain'), dict(source='datasets', target='samples')],
                   generation_limit_reached=False, input_truncated=False, target_truncated=False)
        scores = sentence_metrics([row], 'deep', 1.)['source_edit_positions']
        self.assertEqual(scores, dict(matched=1, missed=1, extra=1, precision=.5, recall=.5))

    def test_identifies_protected_changes_in_raw_edits(self):
        cases = [
            ('The method uses 12 samples.', 'The method uses 13 samples.', 'changed_numbers'),
            ('We evaluate seven architectures.', 'We evaluate diverse architectures.', 'changed_number_words'),
            ('This approach is upto 12 times faster than the baseline.',
             'This approach is up to 12 optimal for the baseline.', 'changed_numeric_comparison'),
            ('Bio-T5 extracts the annotations.', 'The model extracts the annotations.', 'changed_technical_identifier'),
            ('We use the image features.', 'We use the VQA image features.', 'changed_technical_identifier'),
            ('This approach is upto 12 times faster than the baseline.',
             'This method is up to 12 times faster than the baseline.', None),
            ('We use a corpus this kind.', 'We use a corpus of this kind.', None),
        ]
        for source, candidate, expected in cases:
            with self.subTest(source=source, candidate=candidate):
                self.assertEqual(span_fact_error(source, candidate), expected)

    def test_edit_targets_round_trip_with_literal_kept_text(self):
        source = '  We use a corpus this kind to describe  physical appearance.\n'
        target = '  We employ a corpus of this kind to describe  physical appearance.\n'
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = build_subword_tokenizer([dict(source=source, target=target)], Path(directory))
            config = config_for(tokenizer)
            tokens = encode_edit_program(source, target, tokenizer, config, max_length=24)
            prediction, edits = apply_edit_program(source, tokens, tokenizer, config)
            self.assertEqual(prediction, target)
            self.assertEqual([(edit.source, edit.target) for edit in edits], [(' use', ' employ'), ('', ' of')])
            self.assertEqual(edits[1].start, source.index(' this'))
            self.assertTrue(prediction.endswith('  physical appearance.\n'))
            deletion = target.replace(' this kind', '')
            tokens = encode_edit_program(target, deletion, tokenizer, config, max_length=24)
            self.assertEqual(apply_edit_program(target, tokens, tokenizer, config)[0], deletion)
            no_edits = [config.bos_token_id, config.eos_token_id, config.pad_token_id]
            self.assertEqual(apply_edit_program(source, no_edits, tokenizer, config), (source, []))

    def test_invalid_ranges_and_joined_words_fail_atomically(self):
        source = 'Keep this phrase.'
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = build_subword_tokenizer([dict(source=source, target=source)], Path(directory))
            config = config_for(tokenizer)
            base, bos, eos = config.text_vocab_size, config.bos_token_id, config.eos_token_id
            payload = tokenizer.encode('splice', add_special_tokens=False)
            invalid = [([bos, base + 1, base + 2], 'Incomplete edit program'),
                       ([bos, base + 1, eos], 'Incomplete edit range'),
                       ([bos, base + 1, base + 8, eos], 'exceed the source'),
                       ([bos, base, base + 2, base + 1, base + 2, eos], 'overlap'),
                       ([bos, base + 1, base + 1, base + 1, base + 2, eos], 'overlap'),
                       ([bos, base + 1, base + 2, *payload, eos], 'join text'),
                       ([bos, base, base, *payload, eos], 'join text')]
            for tokens, reason in invalid:
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    apply_edit_program(source, tokens, tokenizer, config)

    def test_decoder_masks_out_of_order_positions_and_retains_eos(self):
        config = SixLayerConfig(vocab_size=13, d_model=16, max_positions=10, depth='joint',
                                prediction_stages=4, fuse_stage_outputs=True, span_edits=True,
                                word_replacements=word_replacements(WORDS))
        decoder = EditProgramDecoder(config, [2, 1], 'cpu')
        base = config.text_vocab_size

        def choose(first_choices, second_choices, *, first=False, remaining=8):
            logits = torch.full((2, config.vocab_size), -1.)
            for row, preferences in enumerate((first_choices, second_choices)):
                for score, token in enumerate(reversed(preferences)):
                    logits[row, token] = score
            return decoder.choose(logits, first=first, remaining=remaining).tolist()

        self.assertEqual(choose([4], [4], first=True), [1, 1])
        self.assertEqual(choose([4, base + 1], [base + 3, base]), [base + 1, base])
        self.assertEqual(choose([base, 2, base + 2], [2, base]), [base + 2, base])
        self.assertEqual(choose([base + 1, 4], [base, 5]), [4, 5])
        self.assertEqual(choose([base + 2, 2], [base + 1, 2], remaining=2), [2, 2])

    def test_adapt_a_fused_checkpoint_and_export_applied_edits(self):
        row = dict(id='sample', source='We use a corpus this kind.', target='We use a corpus of this kind.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, output = root/'parent', root/'output'
            parent.mkdir()
            tokenizer = build_subword_tokenizer([row, *WORDS], parent/'tokenizer')
            original = SixLayerRewriter(config_for(tokenizer, span_edits=False))
            original.config.save(parent/'six_layer_config.json')
            torch.save(original.state_dict(), parent/'weights.pt')
            before = (parent/'weights.pt').read_bytes()
            (parent/'result.json').write_text('{"model":"six_layer"}', encoding='utf-8')
            words = root/'words.jsonl'
            words.write_text(json.dumps(WORDS[0])+'\n', encoding='utf-8')
            for split in ('train', 'validation'):
                (root/(split+'.jsonl')).write_text(json.dumps(row)+'\n', encoding='utf-8')
            args = parser().parse_args(['--data', str(root), '--output', str(output), '--word-data', str(words),
                                        '--init-checkpoint', str(parent), '--span-edits', '--device', 'cpu',
                                        '--steps', '1', '--batch-size', '1', '--max-length', '24',
                                        '--checkpoint-selection', 'final', '--span-payload-weight', '2'])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
                metrics = evaluate_checkpoint(evaluation_parser().parse_args([
                    '--data', str(root), '--checkpoint', str(output), '--split', 'validation',
                    '--device', 'cpu', '--max-new-tokens', '26', '--span-copy-scale', '.5']))
            saved, _, _ = load_checkpoint(output, 'cpu')
            self.assertEqual(saved.config.text_vocab_size, len(tokenizer))
            self.assertEqual(saved.config.edit_position_count, 25)
            self.assertEqual(result['added_edit_boundary_embeddings'], 25)
            self.assertEqual(result['span_payload_weight'], 2.)
            self.assertEqual(before, (parent/'weights.pt').read_bytes())
            evaluated, = read_rows(output/'evaluation'/'validation-26-copy-scale-0.5'/'predictions.jsonl')
            single = rewrite_neural(row['source'], None, output, 'cpu', max_new_tokens=26, span_copy_scale=.5)
            self.assertEqual(single['prediction'], row['source'] if single['fallback'] else evaluated['prediction'])
            self.assertEqual(single['edits'], [] if single['fallback'] else evaluated['edits'])
            self.assertIn('edit_error', evaluated)
            self.assertEqual(single['fact_error'], evaluated['fact_error'])
            self.assertEqual(single['edit_status'], evaluated['edit_status'])
            self.assertIn('fact_error_counts', metrics)
            self.assertEqual(metrics['span_copy_scale'], .5)
            self.assertEqual(sum(metrics['edit_status_counts'].values()), 1)
            self.assertNotIn('candidate', single)
