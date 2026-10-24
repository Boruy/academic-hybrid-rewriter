"""Reproducible tiny smoke experiment or pretrained BART fine-tuning."""
import argparse, copy, json, random, time
from pathlib import Path
import torch
from transformers import BartConfig,BartForConditionalGeneration,AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,processors
from .models import HybridRewriter

def read_rows(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]

def build_tokenizer(rows,out):
    tok=Tokenizer(models.WordLevel(unk_token='[UNK]')); tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tok.train_from_iterator((r[k] for r in rows for k in ['source','target']),trainers.WordLevelTrainer(vocab_size=4000,special_tokens=['[PAD]','[BOS]','[EOS]','[UNK]']))
    tok.post_processor=processors.TemplateProcessing(single='[BOS] $A [EOS]',special_tokens=[('[BOS]',1),('[EOS]',2)])
    result=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    result.save_pretrained(out); return result

def batch(rows,tokenizer,device,max_length):
    x=tokenizer([r['source'] for r in rows],padding=True,truncation=True,max_length=max_length,return_tensors='pt')
    y=tokenizer([r['target'] for r in rows],padding=True,truncation=True,max_length=max_length,return_tensors='pt')['input_ids']
    y[y==tokenizer.pad_token_id]=-100
    return {'input_ids':x['input_ids'].to(device),'attention_mask':x['attention_mask'].to(device),'labels':y.to(device)}

def run(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    all_train=read_rows(Path(args.data)/'train.jsonl'); validation=read_rows(Path(args.data)/'validation.jsonl')
    if not all_train or not validation: raise ValueError('Empty training or validation data')
    train=all_train[:args.train_limit] if args.train_limit else all_train
    val=validation[:args.val_limit] if args.val_limit else validation
    if args.tiny:
        tokenizer=build_tokenizer(all_train,out/'tokenizer')
        config=BartConfig(vocab_size=len(tokenizer),d_model=64,encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,max_position_embeddings=args.max_length+8,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,decoder_start_token_id=tokenizer.eos_token_id,dropout=0.)
        backbone=BartForConditionalGeneration(config)
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.pretrained); tokenizer.padding_side='right'
        backbone=BartForConditionalGeneration.from_pretrained(args.pretrained)
    backbone.config.save_pretrained(out/'backbone_config'); tokenizer.save_pretrained(out/'tokenizer')
    # Adapter initialization happens after identical seeded backbone creation in both experiments.
    model=backbone if args.model=='transformer' else HybridRewriter(backbone,args.fusion,args.backend)
    if args.model=='hybrid' and args.freeze_backbone: model.freeze_backbone()
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    fixed=batch(train[:args.batch_size],tokenizer,device,args.max_length)
    @torch.no_grad()
    def fixed_loss():
        model.eval(); return model(**fixed).loss.item()
    initial=fixed_loss(); log=[]; start=time.perf_counter()
    for step in range(args.steps):
        model.train(); chosen=random.sample(train,min(args.batch_size,len(train)))
        b=batch(chosen,tokenizer,device,args.max_length)
        optimizer.zero_grad(set_to_none=True); loss=model(**b).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        log.append({'step':step+1,'train_loss':loss.item()})
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start; final=fixed_loss(); model.eval()
    @torch.no_grad()
    def validation_loss():
        numerator=denominator=0
        for i in range(0,len(val),args.batch_size):
            b=batch(val[i:i+args.batch_size],tokenizer,device,args.max_length)
            tokens=(b['labels']!=-100).sum().item(); numerator+=model(**b).loss.item()*tokens; denominator+=tokens
        return numerator/denominator
    vl=validation_loss()
    inp=batch(val[:2],tokenizer,device,args.max_length)
    gen=model.generate(input_ids=inp['input_ids'],attention_mask=inp['attention_mask'],max_new_tokens=48,num_beams=1)
    predictions=[dict(id=r['id'],source=r['source'],reference=r['target'],prediction=p) for r,p in zip(val[:2],tokenizer.batch_decode(gen,skip_special_tokens=True))]
    max_tokens=lambda k:max(len(tokenizer(r[k])['input_ids']) for r in train)
    result={'model':args.model,'initialization':'random tiny BART' if args.tiny else args.pretrained,'backend':args.backend if args.model=='hybrid' else None,'fusion':args.fusion if args.model=='hybrid' else None,'seed':args.seed,'steps':args.steps,'train_pairs':len(train),'validation_pairs':len(val),'max_length':args.max_length,'batch_size':args.batch_size,'learning_rate':args.lr,'fixed_train_batch_loss_before':initial,'fixed_train_batch_loss_after':final,'validation_cross_entropy':vl,'seconds_training':elapsed,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'train_truncated_sources':sum(len(tokenizer(r['source'])['input_ids'])>args.max_length for r in train),'train_truncated_targets':sum(len(tokenizer(r['target'])['input_ids'])>args.max_length for r in train),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'torch':torch.__version__,'experiment_type':'pipeline smoke test, not a quality comparison' if args.tiny else 'fine-tuning','test_set_used':False,'args':vars(args)}
    torch.save(model.state_dict(),out/'weights.pt')
    for name,data in [('result.json',result),('training_log.json',log),('validation_examples.json',predictions)]: (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2)); return result

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['transformer','hybrid'],default='transformer'); p.add_argument('--tiny',action='store_true'); p.add_argument('--pretrained',default='facebook/bart-base'); p.add_argument('--backend',choices=['reference','official'],default='reference'); p.add_argument('--fusion',choices=['gated','residual','replace'],default='gated'); p.add_argument('--freeze-backbone',action='store_true'); p.add_argument('--data',default='data/processed'); p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=30); p.add_argument('--train-limit',type=int,default=64); p.add_argument('--val-limit',type=int,default=16); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-length',type=int,default=128); p.add_argument('--lr',type=float,default=.001); p.add_argument('--seed',type=int,default=42); p.add_argument('--device')
    return p
if __name__=='__main__': run(parser().parse_args())
