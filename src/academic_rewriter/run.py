"""Train the six-layer academic rewriter or continue a local checkpoint."""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,processors,decoders
from .joint_words import WordTask, word_replacements
from .lexical import read_lexical_training
from .six_layer import CHECKPOINT_DEPTHS, SixLayerConfig, SixLayerRewriter

def read_rows(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def build_tokenizer(rows,out):
    tok=Tokenizer(models.WordLevel(unk_token='[UNK]')); tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tok.train_from_iterator((r[k] for r in rows for k in ['source','target']),trainers.WordLevelTrainer(vocab_size=4000,special_tokens=['[PAD]','[BOS]','[EOS]','[UNK]']))
    tok.post_processor=processors.TemplateProcessing(single='[BOS] $A [EOS]',special_tokens=[('[BOS]',1),('[EOS]',2)])
    result=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    result.save_pretrained(out); return result

def build_subword_tokenizer(rows,out):
    """Byte-level BPE trained on training text only, including a full byte alphabet."""
    tok=Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator((r[k] for r in rows for k in ['source','target']),
        trainers.BpeTrainer(vocab_size=4096,min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=['[PAD]','[BOS]','[EOS]','[UNK]']))
    tok.post_processor=processors.TemplateProcessing(single='[BOS] $A [EOS]',special_tokens=[('[BOS]',1),('[EOS]',2)])
    result=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    result.save_pretrained(out); return result

def _prepare_batch(rows, tokenizer, device, max_length, word_task=None, *, supervise_edits=False):
    sources = [r['source'] for r in rows]
    x=tokenizer(sources,padding=True,truncation=True,max_length=max_length,return_tensors='pt',
                return_offsets_mapping=word_task is not None)
    y=tokenizer([r['target'] for r in rows],padding=True,truncation=True,max_length=max_length,return_tensors='pt')['input_ids']
    y[y==tokenizer.pad_token_id]=-100
    inputs = {'input_ids':x['input_ids'].to(device),'attention_mask':x['attention_mask'].to(device),
              'labels':y.to(device)}
    edit_loss = None
    if word_task and supervise_edits and word_task.edit_label is not None:
        inputs['word_guidance'], edit_loss = word_task.supervised_guidance(rows, x)
    elif word_task:
        inputs['word_guidance'] = word_task.guidance(sources, x)
    return inputs, edit_loss


def batch(rows, tokenizer, device, max_length, word_task=None):
    return _prepare_batch(rows, tokenizer, device, max_length, word_task)[0]

def run(args):
    if args.lr is None:
        args.lr = .001 if args.tiny else .0003
    if args.steps < 1 or args.log_every < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('steps, log-every, batch-size and lr must be positive')
    if args.eval_every < 0 or args.lr_patience < 0 or not 0 < args.lr_drop_factor < 1:
        raise ValueError('eval-every and lr-patience must be nonnegative; lr-drop-factor must be between 0 and 1')
    if args.lr_scheduler=='plateau' and (args.eval_every < 1 or not 0 < args.min_lr <= args.lr):
        raise ValueError('plateau requires eval-every > 0 and 0 < min-lr <= lr')
    if args.early_stopping_patience < 0 or (args.early_stopping_patience and not args.eval_every):
        raise ValueError('early stopping requires a positive validation interval and nonnegative patience')
    if not math.isfinite(args.word_loss_weight) or args.word_loss_weight <= 0:
        raise ValueError('word-loss-weight must be finite and positive')
    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    out=Path(args.output)
    if args.init_checkpoint and Path(args.init_checkpoint).resolve()==out.resolve():
        raise ValueError('Continuation output must differ from its parent')
    if (out/'weights.pt').exists() and not args.overwrite:
        raise ValueError('Output already contains weights; use a new directory or explicit --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    result_path = out/'result.json'
    all_train=read_rows(Path(args.data)/'train.jsonl'); validation=read_rows(Path(args.data)/'validation.jsonl')
    if not all_train or not validation: raise ValueError('Empty training or validation data')
    if any(row.get('synthetic') or row.get('training_only') or row.get('split') == 'train' for row in validation):
        raise ValueError('Sentence validation requires a nonsynthetic held-out split')
    train=all_train[:args.train_limit] if args.train_limit else all_train
    val=validation[:args.val_limit] if args.val_limit else validation
    fixed_rows = train[:args.batch_size]
    if args.sampling == 'balanced':
        real_train = [row for row in train if not row.get('synthetic')]
        synthetic_train = [row for row in train if row.get('synthetic')]
        real_per_batch = (args.batch_size + 1) // 2
        synthetic_per_batch = args.batch_size // 2
        if not synthetic_per_batch:
            raise ValueError('Balanced sampling requires a batch size of at least 2')
        if len(real_train) < real_per_batch or len(synthetic_train) < synthetic_per_batch:
            raise ValueError(f'Balanced sampling needs {real_per_batch} real and '
                             f'{synthetic_per_batch} synthetic training pairs per batch')
        fixed_rows = real_train[:real_per_batch] + synthetic_train[:synthetic_per_batch]
    parent=None
    if args.init_checkpoint:
        parent=Path(args.init_checkpoint)
        if parent.resolve()==out.resolve(): raise ValueError('Continuation output must differ from its parent')
        parent_meta=json.loads((parent/'result.json').read_text(encoding='utf8'))
        if parent_meta['model']!='six_layer':
            raise ValueError('Continuation requires a six-layer checkpoint')
        tokenizer=AutoTokenizer.from_pretrained(parent/'tokenizer',local_files_only=True)
        six_config=SixLayerConfig.load(parent/'six_layer_config.json')
        if args.depth is not None and args.depth != six_config.depth:
            raise ValueError(f'Continuation requires the saved depth ({six_config.depth})')
        if args.word_guidance is not None and (args.word_guidance == 'candidates') != six_config.candidate_guidance:
            raise ValueError('Continuation requires the saved word guidance')
        if args.source_copy is not None and args.source_copy != six_config.source_copy:
            raise ValueError('Continuation requires the saved source-copy setting')
        if args.contextual_edits is not None and args.contextual_edits != six_config.contextual_edits:
            raise ValueError('Continuation requires the saved contextual-edit setting')
    depth = six_config.depth if parent else args.depth or 'joint'
    if args.source_copy and depth != 'joint':
        raise ValueError('Source copying requires joint training')
    if not parent and args.contextual_edits and not args.source_copy:
        raise ValueError('Contextual edit guidance requires --source-copy')
    word_rows = []
    word_hash = None
    replacements = {}
    if depth == 'joint':
        word_rows, word_hash = read_lexical_training(args.word_data)
        replacements = word_replacements(word_rows)
        if parent and replacements != six_config.word_replacements:
            raise ValueError('Continuation requires the saved word replacement vocabulary')
    if parent and 2 * args.max_length > six_config.max_positions:
        raise ValueError('Continuation max-length exceeds saved position capacity')
    if args.overwrite:
        # A reader uses result.json to identify a completed checkpoint.
        result_path.unlink(missing_ok=True)
    (out/'run_args.json').write_text(json.dumps(vars(args),indent=2),encoding='utf8')
    if not parent:
        tokenizer=(build_tokenizer if args.tiny else build_subword_tokenizer)(all_train + word_rows,out/'tokenizer')
        six_config=SixLayerConfig(vocab_size=len(tokenizer),d_model=32 if args.tiny else args.d_model,
            attention_heads=args.attention_heads,ffn_dim=64 if args.tiny else args.ffn_dim,
            dropout=0. if args.tiny else args.dropout,max_positions=2*args.max_length+2,
            pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,depth=depth,word_replacements=replacements,
            prediction_stages=4 if depth == 'joint' else 3, fuse_stage_outputs=depth == 'joint',
            candidate_guidance=depth == 'joint' and args.word_guidance != 'position',
            source_copy=bool(args.source_copy), contextual_edits=bool(args.contextual_edits))
    tokenizer.padding_side='right'; tokenizer.save_pretrained(out/'tokenizer')
    six_config.save(out/'six_layer_config.json');model=SixLayerRewriter(six_config)
    if parent:
        model.load_state_dict(torch.load(parent/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    if args.reject_truncation:
        for split,rows in [('train',train),('validation',val)]:
            for row in rows:
                for field in ['source','target']:
                    if len(tokenizer(row[field])['input_ids'])>args.max_length:
                        raise ValueError(f'{split} {row["id"]} {field} exceeds max-length; refusing silent truncation')
    model.to(device)
    word_task = WordTask(model, tokenizer, device, args.max_length) if word_rows else None
    word_examples = word_task.training_examples(word_rows) if word_task else []
    fixed_words = word_examples[:args.batch_size]
    word_rng = random.Random(args.seed + 1)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=args.lr_drop_factor,patience=args.lr_patience,threshold=.001,threshold_mode='rel',min_lr=args.min_lr) if args.lr_scheduler=='plateau' else None

    def train_objective(rows, words):
        inputs, edit_loss = _prepare_batch(rows, tokenizer, device, args.max_length, word_task,
                                          supervise_edits=True)
        sentence = model.sequence_losses(**inputs)
        loss = sum(sentence.values()) / len(sentence)
        terms = {f'{name}_cross_entropy': value for name, value in sentence.items()}
        if word_task:
            word_stages = word_task.losses(words)
            word_loss = torch.stack(word_stages).mean()
            loss = loss + args.word_loss_weight * word_loss
            terms['word_cross_entropy'] = word_loss
            terms.update({f'p{index}_word_cross_entropy': value
                          for index, value in enumerate(word_stages, start=1)})
        if edit_loss is not None:
            loss = loss + args.word_loss_weight * edit_loss
            terms['source_edit_cross_entropy'] = edit_loss
        return loss, terms

    @torch.no_grad()
    def fixed_loss():
        model.eval()
        return train_objective(fixed_rows, fixed_words)[0].item()
    @torch.no_grad()
    def validation_losses():
        model.eval(); denominator=0
        numerators = dict.fromkeys(model.sentence_depths, 0.)
        for i in range(0,len(val),args.batch_size):
            b=batch(val[i:i+args.batch_size],tokenizer,device,args.max_length,word_task)
            tokens=(b['labels']!=-100).sum().item(); denominator+=tokens
            for name, value in model.sequence_losses(**b).items():
                numerators[name] += value.item()*tokens
        return {name: value/denominator for name, value in numerators.items()}

    def validation_score(by_depth):
        if six_config.fuse_stage_outputs:
            return by_depth['deep']
        return sum(by_depth.values()) / len(by_depth)

    validation_history=[]; validation_seconds=0.; best_value=float('inf'); best_step=None; unimproved=0
    validation_path=out/'validation_progress.jsonl'
    validation_path.write_text('',encoding='utf8')
    def record_validation(step):
        nonlocal validation_seconds,best_value,best_step,unimproved
        validation_start=time.perf_counter()
        print(f'[validation {step}/{args.steps}] Evaluating {len(val)} pairs', flush=True)
        by_depth = validation_losses()
        value = validation_score(by_depth)
        fixed_value=fixed_loss()
        if not math.isfinite(value): raise RuntimeError('Non-finite validation loss')
        before=optimizer.param_groups[0]['lr']
        if scheduler is not None: scheduler.step(value)
        after=optimizer.param_groups[0]['lr']
        entry={'step':step,'validation_cross_entropy':value,'fixed_train_batch_loss':fixed_value,'learning_rate':before,'next_learning_rate':after}
        entry['validation_cross_entropy_by_depth'] = by_depth
        validation_history.append(entry)
        if value<best_value:
            best_value=value; best_step=step; unimproved=0
            if args.checkpoint_selection=='best-validation': torch.save(model.state_dict(),out/'weights.pt')
        else: unimproved+=1
        with validation_path.open('a',encoding='utf8') as stream: stream.write(json.dumps(entry)+'\n')
        print(f'[validation {step}/{args.steps}] loss={value:.4f} fixed_train_loss={fixed_value:.4f} lr={before:.2e} next_lr={after:.2e}',flush=True)
        validation_seconds+=time.perf_counter()-validation_start
        return entry
    print(f'Training {args.model} on {device}: {len(train)} training pairs, {len(val)} validation pairs, {args.steps} steps',flush=True)
    initial=fixed_loss(); log=[]; start=time.perf_counter()
    print(f'Initial fixed-batch loss: {initial:.4f}',flush=True)
    progress_path=out/'training_progress.jsonl'
    progress_path.write_text('',encoding='utf8')
    if args.eval_every or args.checkpoint_selection=='best-validation': record_validation(0)
    order=[]; completed_epochs=0; examples_seen=0
    real_examples_seen = 0
    for step in range(args.steps):
        model.train()
        if args.sampling == 'balanced':
            chosen = random.sample(real_train, real_per_batch)
            chosen.extend(random.sample(synthetic_train, synthetic_per_batch))
            random.shuffle(chosen)
        elif args.sampling=='shuffle-epochs':
            if not order:
                order=list(range(len(train))); random.shuffle(order); completed_epochs+=1
            chosen=[train[index] for index in order[:args.batch_size]]; del order[:args.batch_size]
        else: chosen=random.sample(train,min(args.batch_size,len(train)))
        examples_seen+=len(chosen)
        real_examples_seen += sum(not row.get('synthetic') for row in chosen)
        words = word_rng.sample(word_examples, min(args.batch_size, len(word_examples)))
        optimizer.zero_grad(set_to_none=True)
        loss, terms = train_objective(chosen, words)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        learning_rate=optimizer.param_groups[0]['lr']; optimizer.step()
        loss_value=loss.item()
        recent=[entry['train_loss'] for entry in log[-19:]]+[loss_value]
        entry={'step':step+1,'train_loss':loss_value,'train_loss_ma20':sum(recent)/len(recent),'learning_rate':learning_rate,'gradient_norm_before_clip':grad_norm.item()}
        entry.update({name: value.item() for name, value in terms.items()})
        log.append(entry)
        with progress_path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(entry)+'\n')
            stream.flush()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            progress = (f"[train {step + 1}/{args.steps}] loss={loss_value:.4f} "
                        f"ma20={entry['train_loss_ma20']:.4f}")
            if six_config.fuse_stage_outputs:
                progress += (f" p3_ce={entry['medium_cross_entropy']:.4f}"
                             f" p4_ce={entry['deep_cross_entropy']:.4f}"
                             f" word_ce={entry['word_cross_entropy']:.4f}")
            if 'source_edit_cross_entropy' in entry:
                progress += f" edit_ce={entry['source_edit_cross_entropy']:.4f}"
            print(f'{progress} lr={learning_rate:.2e} elapsed={time.perf_counter() - start:.1f}s', flush=True)
        if args.eval_every and (step+1)%args.eval_every==0: record_validation(step+1)
        if args.early_stopping_patience and unimproved>=args.early_stopping_patience:
            print(f'Early stopping after {step+1} updates; best validation step={best_step}',flush=True)
            break
    actual_steps=step+1
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start-validation_seconds; final=fixed_loss(); model.eval()
    print('Computing validation loss...',flush=True)
    final_validation = (validation_history[-1] if validation_history and validation_history[-1]['step']==actual_steps
                        else record_validation(actual_steps))
    final_vl = final_validation['validation_cross_entropy']
    if args.checkpoint_selection=='best-validation':
        model.load_state_dict(torch.load(out/'weights.pt',map_location=device,weights_only=True)); final=fixed_loss()
        selected_validation = validation_losses()
    else: selected_validation = final_validation['validation_cross_entropy_by_depth']
    vl = validation_score(selected_validation)
    print(f'Validation cross-entropy: {vl:.4f}',flush=True)
    with torch.no_grad():
        inp=batch(val[:2],tokenizer,device,args.max_length,word_task)
        gen=model.generate(input_ids=inp['input_ids'],attention_mask=inp['attention_mask'],
                           max_new_tokens=args.max_length,num_beams=1,word_guidance=inp.get('word_guidance'))
    predictions=[dict(id=r['id'],source=r['source'],reference=r['target'],prediction=p) for r,p in zip(val[:2],tokenizer.batch_decode(gen,skip_special_tokens=True))]
    result={'model':'six_layer','seed':args.seed,'steps':args.steps,'train_pairs':len(train),'validation_pairs':len(val),'max_length':args.max_length,'batch_size':args.batch_size,'learning_rate':args.lr,'fixed_train_batch_loss_before':initial,'fixed_train_batch_loss_after':final,'validation_cross_entropy':vl,'seconds_training':elapsed,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'train_truncated_sources':sum(len(tokenizer(r['source'])['input_ids'])>args.max_length for r in train),'train_truncated_targets':sum(len(tokenizer(r['target'])['input_ids'])>args.max_length for r in train),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'torch':torch.__version__,'test_set_used':False,'args':vars(args)}
    result.update(steps=actual_steps,requested_steps=args.steps,final_step_validation_cross_entropy=final_vl,checkpoint_selection=args.checkpoint_selection,selected_step=best_step if args.checkpoint_selection=='best-validation' else actual_steps,best_recorded_validation_cross_entropy=best_value,final_learning_rate=optimizer.param_groups[0]['lr'],lr_scheduler=args.lr_scheduler,seconds_validation=validation_seconds,sampling=args.sampling,examples_seen=examples_seen,dataset_equivalent_passes=examples_seen/len(train),optimizer_reinitialized=True,parent_checkpoint=str(parent) if parent else None,parent_weights_sha256=file_hash(parent/'weights.pt') if parent else None,data_sha256={split:file_hash(Path(args.data)/(split+'.jsonl')) for split in ['train','validation']})
    result.update(initialization=('local checkpoint: '+str(parent)) if parent else 'random six-layer model; training-only tokenizer',
        backend='reference',layer_layout=list(model.layer_layout),sequence_layers=6,
        rewrite_depth=six_config.depth,active_sequence_layers=model.active_layer_count,
        active_layer_layout=list(model.layer_layout[:model.active_layer_count]),
        conditional_generation='causal source prefix with source copying and target-only token loss'
        if six_config.source_copy else 'causal source prefix; target-only token loss',
        source_copy=six_config.source_copy, contextual_edits=six_config.contextual_edits,
        real_examples_seen=real_examples_seen,
        synthetic_examples_seen=examples_seen-real_examples_seen,
        experiment_type='pipeline smoke test, not a quality comparison' if args.tiny else 'six-layer synthetic-augmented exploratory training',
        cross_entropy_comparability='Compare CE only when tokenization and evaluation protocol match.')
    if not parent:
        result['fresh_input_embedding_std'] = model.input_embedding_init_std
    result['validation_cross_entropy_by_depth'] = selected_validation
    if word_task:
        fused = six_config.fuse_stage_outputs
        result.update(word_training_pairs=len(word_rows), word_training_data_sha256=word_hash,
                      rewrite_modes=['final'] if fused else ['shallow', *model.sentence_depths],
                      fuse_stage_outputs=fused,
                      candidate_guidance=six_config.candidate_guidance,
                      prediction_stage_layers=list(model.prediction_stage_layers),
                      objective_weights=({'word': args.word_loss_weight, 'aux_sentence': .5, 'final_sentence': .5}
                                         if fused else {'shallow': args.word_loss_weight, 'medium': .5, 'deep': .5}),
                      checkpoint_selection_metric=('Fused final validation cross-entropy' if fused else
                                                   'Mean of medium and deep validation cross-entropies'))
        if six_config.contextual_edits:
            result['objective_weights']['source_edit'] = args.word_loss_weight
    manifest=Path(args.data)/'manifest.json'
    if manifest.exists():result['training_data_manifest_sha256']=file_hash(manifest)
    result['validation_examples_max_new_tokens']=args.max_length
    torch.save(model.state_dict(),out/'weights.pt')
    for name, data in [('training_log.json', log), ('validation_log.json', validation_history),
                       ('validation_examples.json', predictions)]:
        (out/name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    pending_result = result_path.with_suffix('.json.tmp')
    pending_result.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    pending_result.replace(result_path)
    print(json.dumps(result,indent=2),flush=True)
    print(f'Training complete. Results: {out.resolve()}',flush=True)
    return result

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['six_layer'],default='six_layer')
    p.add_argument('--tiny',action='store_true',help='Use a small model and word-level tokenizer for pipeline checks')
    p.add_argument('--data',default='data/academic-pairs-v1')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--train-limit',type=int,default=0,help='0 uses the entire training split')
    p.add_argument('--val-limit',type=int,default=0,help='0 uses the entire validation split')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--lr',type=float,help='Default: tiny=1e-3, otherwise 3e-4')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device')
    p.add_argument('--log-every', type=int, default=1, help='Print progress every N updates (default: 1)')
    p.add_argument('--depth', choices=CHECKPOINT_DEPTHS,
                   help='New runs default to joint; continuations use the saved depth')
    p.add_argument('--word-data', default='data/word-formality-v1/train.jsonl', help='Word training pairs for the joint model')
    p.add_argument('--word-loss-weight', type=float, default=.25, help='Weight on the joint single-word cross-entropy')
    p.add_argument('--word-guidance', choices=['position', 'candidates'],
                   help='Fresh joint runs default to candidates; continuations use the saved setting')
    p.add_argument('--source-copy', action=argparse.BooleanOptionalAction,
                   help='Mix source copying into P4. Continuations use the saved setting.')
    p.add_argument('--contextual-edits', action=argparse.BooleanOptionalAction,
                   help='Learn edit positions from real sentence pairs and guide source copying.')
    p.add_argument('--d-model',type=int,default=192,help='Six-layer hidden width')
    p.add_argument('--attention-heads',type=int,default=4,help='Six-layer attention head count')
    p.add_argument('--ffn-dim',type=int,default=768,help='Six-layer Transformer feed-forward width')
    p.add_argument('--dropout',type=float,default=.1,help='Six-layer residual dropout')
    p.add_argument('--eval-every',type=int,default=250,help='Validation interval in optimizer steps; 0 disables intermediate validation')
    p.add_argument('--lr-scheduler',choices=['constant','plateau'],default='constant')
    p.add_argument('--lr-drop-factor',type=float,default=.5)
    p.add_argument('--lr-patience',type=int,default=1,help='Number of unimproved validation checks tolerated before reducing LR')
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--init-checkpoint',help='Initialize matching model, tokenizer and config from local weights; start a new optimizer')
    p.add_argument('--checkpoint-selection',choices=['final','best-validation'],default='best-validation',help='Best validation includes step 0, so a harmful continuation can select unchanged parent weights')
    p.add_argument('--early-stopping-patience',type=int,default=0,help='Stop after this many unimproved validation checks; 0 disables')
    p.add_argument('--sampling', choices=['random-batches', 'shuffle-epochs', 'balanced'],
                   default='shuffle-epochs', help='Balanced batches use half real pairs, rounded up')
    p.add_argument('--reject-truncation',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='Explicitly allow replacing an existing output checkpoint')
    return p
if __name__=='__main__': run(parser().parse_args())
