import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from academic_rewriter.run import parser, run


class TrainingControlsTests(unittest.TestCase):
    def test_interrupted_overwrite_clears_old_checkpoint_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'validation'):
                (root/(split+'.jsonl')).write_text(
                    json.dumps(dict(id=split, source='Original.', target='Revised.')) + '\n', encoding='utf8')
            output = root/'checkpoint'
            output.mkdir()
            weights = output/'weights.pt'
            weights.write_bytes(b'old weights')
            result = output/'result.json'
            result.write_text('{"model":"six_layer"}', encoding='utf8')
            flags = ['--tiny', '--depth', 'deep', '--device', 'cpu',
                     '--data', str(root), '--output', str(output)]

            with self.assertRaisesRegex(ValueError, 'Output already contains weights'):
                run(parser().parse_args(flags))
            self.assertTrue(result.is_file())

            with patch('academic_rewriter.run.build_tokenizer', side_effect=RuntimeError('interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run(parser().parse_args([*flags, '--overwrite']))
            self.assertFalse(result.exists())
            self.assertEqual(weights.read_bytes(), b'old weights')

    def test_validation_plateau_reduces_lr_and_persists_live_metrics(self):
        """Opposing validation targets expose overfitting and must lower the LR."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'
            data.mkdir()
            source = 'original original original original original'
            training = [dict(id='train', source=source, target='revised revised revised revised revised')]
            validation = [dict(id='validation', source=source, target=source)]
            for split, rows in [('train', training), ('validation', validation)]:
                (data/(split+'.jsonl')).write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf8')
            output = root/'run'
            args = parser().parse_args(['--tiny', '--model', 'six_layer', '--depth', 'deep', '--device', 'cpu',
                                       '--data', str(data), '--output', str(output), '--steps', '4',
                                       '--max-length', '16', '--checkpoint-selection', 'final',
                                       '--batch-size', '1', '--lr', '.02', '--eval-every', '1',
                                       '--lr-scheduler', 'plateau', '--lr-patience', '0', '--min-lr', '.001'])
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)
            history = json.loads((output/'validation_log.json').read_text(encoding='utf8'))
            live_validation = [json.loads(line) for line in (output/'validation_progress.jsonl').read_text(encoding='utf8').splitlines()]
            log = json.loads((output/'training_log.json').read_text(encoding='utf8'))
            live_log = [json.loads(line) for line in (output/'training_progress.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual([row['step'] for row in history], [0,1,2,3,4])
            self.assertEqual(history, live_validation)
            self.assertEqual(log, live_log)
            self.assertLess(result['final_learning_rate'], args.lr)
            self.assertLess(log[-1]['learning_rate'], args.lr)
            self.assertGreaterEqual(result['final_learning_rate'], args.min_lr)
            self.assertTrue(any(row['next_learning_rate']<row['learning_rate'] for row in history))
            self.assertAlmostEqual(result['validation_cross_entropy'], history[-1]['validation_cross_entropy'])
            for index, row in enumerate(log):
                expected = sum(item['train_loss'] for item in log[max(0,index-19):index+1])/min(20,index+1)
                self.assertAlmostEqual(row['train_loss_ma20'], expected)
                self.assertGreaterEqual(row['gradient_norm_before_clip'], 0)
            self.assertFalse(result['test_set_used'])


if __name__=='__main__':
    unittest.main()
