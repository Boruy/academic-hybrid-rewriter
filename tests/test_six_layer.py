import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from academic_rewriter.run import batch, parser, read_rows, run
from academic_rewriter.six_layer import SixLayerConfig, SixLayerRewriter
from academic_rewriter.test import parser as evaluation_parser, run as evaluate_checkpoint


class SixLayerTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(4)
        return SixLayerRewriter(SixLayerConfig(vocab_size=24, d_model=16, attention_heads=2,
                                               ffn_dim=32, dropout=0., max_positions=40)).eval()

    def test_six_blocks_are_causal_and_cached_logits_match(self):
        model = self.model()
        self.assertEqual([layer.kind for layer in model.layers],
                         ['lstm', 'lstm', 'transformer', 'transformer', 'mamba', 'mamba'])
        tokens = torch.tensor([[1, 4, 5, 2, 1, 7, 8, 2], [1, 6, 4, 2, 1, 9, 3, 2]])
        mask = torch.ones_like(tokens)
        segments = torch.zeros_like(tokens)
        segments[:, 4:] = 1
        expected = model.sequence_logits(tokens, mask, segments)
        changed = tokens.clone()
        changed[:, 5:] = 12
        altered = model.sequence_logits(changed, mask, segments)
        torch.testing.assert_close(expected[:, :5], altered[:, :5], atol=1e-6, rtol=1e-5)
        states = [None] * 6
        for index in range(tokens.size(1)):
            measured, states = model.step(tokens[:, index], torch.full((2,), index),
                                           segments[:, index], mask[:, index].bool(), states)
            torch.testing.assert_close(measured, expected[:, index], atol=2e-6, rtol=2e-5)

    def test_target_only_loss_and_all_layer_gradients(self):
        model = self.model()
        source = torch.tensor([[1, 4, 2], [1, 5, 2]])
        target = torch.tensor([[1, 6, 2], [1, 7, 2]])
        mask = torch.ones_like(source)
        output = model(source, mask, labels=target)
        manual = F.cross_entropy(output.logits[:, 2:5].reshape(-1, 24), target.reshape(-1))
        torch.testing.assert_close(output.loss, manual)
        output.loss.backward()
        for layer in model.layers:
            gradients = [parameter.grad for parameter in layer.parameters() if parameter.requires_grad]
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients))
            self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0)

    def test_right_padding_does_not_change_loss_or_cached_generation(self):
        model = self.model()
        source = torch.tensor([[1, 4, 2, 0, 0], [1, 6, 7, 8, 2]])
        mask = source.ne(0).long()
        target = torch.tensor([[1, 9, 2, -100], [1, 6, 9, 2]])
        output = model(source, mask, labels=target)
        self.assertTrue(torch.isfinite(output.loss))
        generated = model.generate(source, mask, max_new_tokens=5)
        for index, length in enumerate(mask.sum(1)):
            single = model.generate(source[index:index+1, :length], mask[index:index+1, :length],
                                     max_new_tokens=5)[0].tolist()
            joint = generated[index].tolist()
            self.assertEqual([value for value in joint if value != 0], [value for value in single if value != 0])
        with self.assertRaisesRegex(ValueError, 'right-padded'):
            model.generate(torch.tensor([[1, 0, 2]]), torch.tensor([[1, 0, 1]]))
        with self.assertRaisesRegex(ValueError, 'position capacity'):
            model.generate(source, mask, max_new_tokens=40)

    def test_train_reload_evaluate_and_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            for split in ['train', 'validation']:
                rows = [dict(id=split, source='we looked at ten samples', target='we examined ten samples')]
                if split == 'validation':
                    rows.append(dict(id='validation-quoted', source='we looked at "ten samples",\nthen compared them.',
                                     target='we examined "ten samples",\nthen compared them.'))
                (data/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
            output = root/'run'
            settings = ['--model', 'six_layer', '--depth', 'deep', '--tiny', '--device', 'cpu', '--data', str(data),
                        '--steps', '2', '--batch-size', '1', '--max-length', '16', '--eval-every', '1',
                        '--checkpoint-selection', 'best-validation', '--reject-truncation']
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(parser().parse_args(settings + ['--output', str(output)]))
                metrics = evaluate_checkpoint(evaluation_parser().parse_args([
                    '--checkpoint', str(output), '--data', str(data), '--split', 'validation',
                    '--device', 'cpu', '--batch-size', '1', '--max-new-tokens', '12']))
                continuation = run(parser().parse_args(settings + ['--init-checkpoint', str(output),
                                                        '--output', str(root/'continued')]))
            config = SixLayerConfig.load(output/'six_layer_config.json')
            model = SixLayerRewriter(config).eval()
            model.load_state_dict(torch.load(output/'weights.pt', weights_only=True))
            tokenizer = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True)
            with torch.no_grad():
                loss = model(**batch(read_rows(data/'validation.jsonl'), tokenizer, 'cpu', 16)).loss.item()
            self.assertAlmostEqual(result['validation_cross_entropy'], loss, places=5)
            self.assertEqual(result['sequence_layers'], 6)
            self.assertEqual(metrics['n'], 2)
            exported = output/'evaluation'/'validation-12'
            predictions = read_rows(exported/'predictions.jsonl')
            with (exported/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
                comparison = list(csv.DictReader(stream))
            self.assertEqual(comparison, [{key: row[key] for key in ['id', 'source', 'prediction', 'reference']}
                                          for row in predictions])
            self.assertEqual([row['source'] for row in comparison],
                             [row['source'] for row in read_rows(data/'validation.jsonl')])
            self.assertTrue((exported/'predictions.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(continuation['parent_checkpoint'], str(output))
            self.assertFalse((output/'backbone_config').exists())


if __name__ == '__main__':
    unittest.main()
