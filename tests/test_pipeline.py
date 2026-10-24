import tempfile,unittest,json
from pathlib import Path
import torch
from transformers import BartConfig,BartForConditionalGeneration
from academic_rewriter.models import ReferenceMamba,HybridRewriter
from academic_rewriter.prepare import prepare,normalize,numbers
from academic_rewriter.run import build_tokenizer,batch

class PipelineTests(unittest.TestCase):
    def test_normalization_and_numbers(self):
        self.assertEqual(normalize(' A  study\n reports 50% '),'A study reports 50%')
        self.assertNotEqual(numbers('50 samples'),numbers('5 samples'))
    def test_document_and_text_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);raw=td/'raw';raw.mkdir()
            rows=[dict(doc_id=f'2001.{i:05d}',labels='style',before_sent=f'We looked at the novel sample named word{i}.',after_sent=f'We examined the novel sample named word{i}.') for i in range(20)]
            rows.append(dict(rows[0],doc_id='2001.99999',after_sent='We investigated the novel sample named word0.'))
            for s,rs in [('train',rows),('dev',[]),('test',[])]: (raw/(s+'.json')).write_text('\n'.join(json.dumps(r) for r in rs))
            stats=prepare(raw,td/'out'); self.assertEqual(stats['kept_total'],21)
            self.assertTrue(all(x['shared_docs']==x['shared_exact_texts']==0 for x in stats['leakage'].values()))
    def test_reference_mask_and_gradients(self):
        torch.manual_seed(0);m=ReferenceMamba(16);x=torch.randn(2,6,16,requires_grad=True);mask=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1]])
        y=m(x,mask);self.assertEqual(tuple(y.shape),(2,6,16));self.assertTrue(torch.allclose(y[0,:3],m(x[:1,:3],torch.ones(1,3))[0],atol=1e-6))
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertIsNotNone(m.A_log.grad)
    def test_hybrid_loss_generation_and_padding(self):
        with tempfile.TemporaryDirectory() as td:
            rows=[dict(source='we looked at a sample',target='we examined a sample'),dict(source='we examined many cases today',target='we examined many cases today carefully')]
            tok=build_tokenizer(rows,td);b=batch(rows,tok,'cpu',32)
            self.assertTrue((b['labels']==-100).any())
            conf=BartConfig(vocab_size=len(tok),d_model=16,encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=2,dropout=0.)
            for fusion in ['gated','residual','replace']:
                model=HybridRewriter(BartForConditionalGeneration(conf),fusion);model.eval();loss=model(**b).loss;self.assertTrue(torch.isfinite(loss));loss.backward()
                generated=model.generate(b['input_ids'],b['attention_mask'],max_new_tokens=3);self.assertEqual(generated.shape[0],2)

if __name__=='__main__':unittest.main()
