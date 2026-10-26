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
    if args.lr is None:
        args.lr = .001 if args.tiny else (.0001 if args.model=='hybrid' and args.freeze_backbone else .00002)
    if args.steps < 1 or args.log_every < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('steps, log-every, batch-size and lr must be positive')
    if args.eval_every < 0 or args.lr_patience < 0 or not 0 < args.lr_drop_factor < 1:
        raise ValueError('eval-every and lr-patience must be nonnegative; lr-drop-factor must be between 0 and 1')
    if args.lr_scheduler=='plateau' and (args.eval_every < 1 or not 0 < args.min_lr <= args.lr):
        raise ValueError('plateau requires eval-every > 0 and 0 < min-lr <= lr')
    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'run_args.json').write_text(json.dumps(vars(args),indent=2),encoding='utf8')
    all_train=read_rows(Path(args.data)/'train.jsonl'); validation=read_rows(Path(args.data)/'validation.jsonl')
    if not all_train or not validation: raise ValueError('Empty training or validation data')
    train=all_train[:args.train_limit] if args.train_limit else all_train
    val=validation[:args.val_limit] if args.val_limit else validation
    if args.tiny:
        tokenizer=build_tokenizer(all_train,out/'tokenizer')
        config=BartConfig(vocab_size=len(tokenizer),d_model=64,encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=128,decoder_ffn_dim=128,max_position_embeddings=args.max_length+8,pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,decoder_start_token_id=tokenizer.eos_token_id,dropout=0.)
        backbone=BartForConditionalGeneration(config)
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.pretrained,local_files_only=args.local_files_only); tokenizer.padding_side='right'
        backbone=BartForConditionalGeneration.from_pretrained(args.pretrained,local_files_only=args.local_files_only)
    backbone.config.save_pretrained(out/'backbone_config'); tokenizer.save_pretrained(out/'tokenizer')
    # Adapter initialization happens after identical seeded backbone creation in both experiments.
    model=backbone if args.model=='transformer' else HybridRewriter(backbone,args.fusion,args.backend)
    if args.model=='hybrid' and args.freeze_backbone: model.freeze_backbone()
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=args.lr_drop_factor,patience=args.lr_patience,threshold=.001,threshold_mode='rel',min_lr=args.min_lr) if args.lr_scheduler=='plateau' else None
    fixed=batch(train[:args.batch_size],tokenizer,device,args.max_length)
    @torch.no_grad()
    def fixed_loss():
        model.eval(); return model(**fixed).loss.item()
    @torch.no_grad()
    def validation_loss():
        model.eval(); numerator=denominator=0
        for i in range(0,len(val),args.batch_size):
            b=batch(val[i:i+args.batch_size],tokenizer,device,args.max_length)
            tokens=(b['labels']!=-100).sum().item(); numerator+=model(**b).loss.item()*tokens; denominator+=tokens
        return numerator/denominator
    validation_history=[]; validation_seconds=0.
    validation_path=out/'validation_progress.jsonl'
    validation_path.write_text('',encoding='utf8')
    def record_validation(step):
        nonlocal validation_seconds
        validation_start=time.perf_counter()
        value=validation_loss(); fixed_value=fixed_loss()
        before=optimizer.param_groups[0]['lr']
        if scheduler is not None: scheduler.step(value)
        after=optimizer.param_groups[0]['lr']
        entry={'step':step,'validation_cross_entropy':value,'fixed_train_batch_loss':fixed_value,'learning_rate':before,'next_learning_rate':after}
        validation_history.append(entry)
        with validation_path.open('a',encoding='utf8') as stream: stream.write(json.dumps(entry)+'\n')
        print(f'[validation {step}/{args.steps}] loss={value:.4f} fixed_train_loss={fixed_value:.4f} lr={before:.2e} next_lr={after:.2e}',flush=True)
        validation_seconds+=time.perf_counter()-validation_start
        return value
    initial=fixed_loss(); log=[]; start=time.perf_counter()
    print(f'Training {args.model} on {device}: {len(train)} training pairs, {len(val)} validation pairs, {args.steps} steps',flush=True)
    print(f'Initial fixed-batch loss: {initial:.4f}',flush=True)
    progress_path=out/'training_progress.jsonl'
    progress_path.write_text('',encoding='utf8')
    if args.eval_every: record_validation(0)
    for step in range(args.steps):
        model.train(); chosen=random.sample(train,min(args.batch_size,len(train)))
        b=batch(chosen,tokenizer,device,args.max_length)
        optimizer.zero_grad(set_to_none=True); loss=model(**b).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        learning_rate=optimizer.param_groups[0]['lr']; optimizer.step()
        loss_value=loss.item()
        recent=[entry['train_loss'] for entry in log[-19:]]+[loss_value]
        entry={'step':step+1,'train_loss':loss_value,'train_loss_ma20':sum(recent)/len(recent),'learning_rate':learning_rate,'gradient_norm_before_clip':grad_norm.item()}
        log.append(entry)
        with progress_path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(entry)+'\n')
            stream.flush()
        if step==0 or (step+1)%args.log_every==0 or step+1==args.steps:
            print(f"[train {step+1}/{args.steps}] loss={entry['train_loss']:.4f} ma20={entry['train_loss_ma20']:.4f} lr={learning_rate:.2e} elapsed={time.perf_counter()-start:.1f}s",flush=True)
        if args.eval_every and (step+1)%args.eval_every==0: record_validation(step+1)
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start-validation_seconds; final=fixed_loss(); model.eval()
    print('Computing validation loss...',flush=True)
    vl=validation_history[-1]['validation_cross_entropy'] if validation_history and validation_history[-1]['step']==args.steps else record_validation(args.steps)
    print(f'Validation cross-entropy: {vl:.4f}',flush=True)
    inp=batch(val[:2],tokenizer,device,args.max_length)
    gen=model.generate(input_ids=inp['input_ids'],attention_mask=inp['attention_mask'],max_new_tokens=48,num_beams=1)
    predictions=[dict(id=r['id'],source=r['source'],reference=r['target'],prediction=p) for r,p in zip(val[:2],tokenizer.batch_decode(gen,skip_special_tokens=True))]
    max_tokens=lambda k:max(len(tokenizer(r[k])['input_ids']) for r in train)
    result={'model':args.model,'initialization':'random tiny BART' if args.tiny else args.pretrained,'backend':args.backend if args.model=='hybrid' else None,'fusion':args.fusion if args.model=='hybrid' else None,'seed':args.seed,'steps':args.steps,'train_pairs':len(train),'validation_pairs':len(val),'max_length':args.max_length,'batch_size':args.batch_size,'learning_rate':args.lr,'fixed_train_batch_loss_before':initial,'fixed_train_batch_loss_after':final,'validation_cross_entropy':vl,'seconds_training':elapsed,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'train_truncated_sources':sum(len(tokenizer(r['source'])['input_ids'])>args.max_length for r in train),'train_truncated_targets':sum(len(tokenizer(r['target'])['input_ids'])>args.max_length for r in train),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'torch':torch.__version__,'experiment_type':'pipeline smoke test, not a quality comparison' if args.tiny else 'fine-tuning','test_set_used':False,'args':vars(args)}
    result.update(final_learning_rate=optimizer.param_groups[0]['lr'],lr_scheduler=args.lr_scheduler,best_recorded_validation_cross_entropy=min(row['validation_cross_entropy'] for row in validation_history),seconds_validation=validation_seconds)
    torch.save(model.state_dict(),out/'weights.pt')
    for name,data in [('result.json',result),('training_log.json',log),('validation_log.json',validation_history),('validation_examples.json',predictions)]: (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    print(f'Training complete. Results: {out.resolve()}',flush=True)
    return result

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['transformer','hybrid'],default='transformer'); p.add_argument('--tiny',action='store_true'); p.add_argument('--pretrained',default='facebook/bart-base'); p.add_argument('--backend',choices=['reference','official'],default='reference'); p.add_argument('--fusion',choices=['gated','residual','replace'],default='gated'); p.add_argument('--freeze-backbone',action='store_true'); p.add_argument('--data',default='data/processed'); p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=30); p.add_argument('--train-limit',type=int,default=64); p.add_argument('--val-limit',type=int,default=16); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--max-length',type=int,default=128); p.add_argument('--lr',type=float,help='Default: tiny=1e-3, frozen hybrid=1e-4, full fine-tuning=2e-5'); p.add_argument('--seed',type=int,default=42); p.add_argument('--device'); p.add_argument('--log-every',type=int,default=1)
    p.add_argument('--eval-every',type=int,default=50,help='Validation interval in optimizer steps; 0 disables intermediate validation')
    p.add_argument('--lr-scheduler',choices=['constant','plateau'],default='constant')
    p.add_argument('--lr-drop-factor',type=float,default=.5)
    p.add_argument('--lr-patience',type=int,default=1,help='Number of unimproved validation checks tolerated before reducing LR')
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--local-files-only',action='store_true',help='Load pretrained assets from the local cache without network checks')
    return p
if __name__=='__main__': run(parser().parse_args())
