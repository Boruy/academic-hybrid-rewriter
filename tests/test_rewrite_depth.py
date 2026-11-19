import json
from pathlib import Path
import tempfile
import unittest

import torch

from academic_rewriter.six_layer import SixLayerConfig, SixLayerRewriter


class RewriteDepthTests(unittest.TestCase):
    def test_old_config_still_selects_all_six_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary)/'six_layer_config.json'
            config_path.write_text(json.dumps({'vocab_size': 24}), encoding='utf8')
            config = SixLayerConfig.load(config_path)
        self.assertEqual(config.depth, 'deep')
        self.assertEqual(SixLayerRewriter(config).active_layer_count, 6)

    def test_medium_uses_four_blocks_in_training_and_cached_generation(self):
        config = SixLayerConfig(vocab_size=24, d_model=16, attention_heads=2,
                                ffn_dim=32, dropout=0., max_positions=32, depth='medium')
        model = SixLayerRewriter(config).eval()
        source = torch.tensor([[1, 4, 2]])
        mask = torch.ones_like(source)
        target = torch.tensor([[1, 5, 2]])
        model(source, mask, labels=target).loss.backward()
        for layer in model.layers[:4]:
            self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        for layer in model.layers[4:]:
            self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

        with torch.no_grad():
            expected = model.sequence_logits(source, mask)
            states = [None] * 4
            for position in range(source.size(1)):
                actual, states = model.step(source[:, position], torch.tensor([position]),
                                            torch.tensor([0]), torch.tensor([True]), states)
                torch.testing.assert_close(actual, expected[:, position], atol=2e-6, rtol=2e-5)
            self.assertEqual(len(states), 4)
            self.assertEqual(model.generate(source, mask, max_new_tokens=2)[0, 0].item(),
                             config.bos_token_id)
