import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

from academic_rewriter.run import batch, parser, read_rows, run


class RefinementTrainingTests(unittest.TestCase):
    def test_four_arms_and_selected_weights_are_reproducible(self):
        """Exercise direct/continued BART and hybrid without using research data."""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/'data'; data.mkdir()
            for split,source,target in [('train','the result may work','the result may improve'),('validation','the result may improve','the result may work')]:
                (data/(split+'.jsonl')).write_text(json.dumps(dict(id=split,source=source,target=target))+'\n',encoding='utf8')
            common=['--tiny','--device','cpu','--data',str(data),'--steps','2','--batch-size','1','--max-length','16','--eval-every','1','--lr','.01','--sampling','shuffle-epochs','--checkpoint-selection','best-validation','--reject-truncation']
            for architecture in ['transformer','hybrid']:
                direct=root/architecture; continued=root/(architecture+'_continued')
                with contextlib.redirect_stdout(io.StringIO()):
                    initial=run(parser().parse_args(common+['--model',architecture,'--output',str(direct)]))
                    followup=run(parser().parse_args(common+['--model',architecture,'--output',str(continued),'--init-checkpoint',str(direct),'--early-stopping-patience','1']))
                self.assertEqual(followup['parent_weights_sha256'],__import__('hashlib').sha256((direct/'weights.pt').read_bytes()).hexdigest())
                self.assertTrue(followup['optimizer_reinitialized'])
                self.assertFalse(followup['test_set_used'])
                self.assertLessEqual(followup['validation_cross_entropy'],followup['final_step_validation_cross_entropy']+1e-5)
                self.assertEqual(followup['examples_seen'],followup['steps'])
                self.assertEqual(set(followup['data_sha256']),{'train','validation'})
                self.assertIn(followup['selected_step'],[0,1,2])
                self.assertEqual(initial['checkpoint_selection'],'best-validation')
                if architecture=='transformer':
                    tokenizer=AutoTokenizer.from_pretrained(continued/'tokenizer')
                    restored=BartForConditionalGeneration(BartConfig.from_pretrained(continued/'backbone_config')).eval()
                    restored.load_state_dict(torch.load(continued/'weights.pt',weights_only=True))
                    with torch.no_grad(): measured=restored(**batch(read_rows(data/'validation.jsonl'),tokenizer,'cpu',16)).loss.item()
                    self.assertAlmostEqual(measured,followup['validation_cross_entropy'],places=5)
                with self.assertRaisesRegex(ValueError,'already contains weights'):
                    run(parser().parse_args(common+['--model',architecture,'--output',str(direct)]))
            with self.assertRaisesRegex(ValueError,'must match'):
                run(parser().parse_args(common+['--model','hybrid','--output',str(root/'wrong'),'--init-checkpoint',str(root/'transformer')]))


if __name__=='__main__': unittest.main()
