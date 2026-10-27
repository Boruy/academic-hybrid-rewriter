"""Build the stage-two visual report from recorded experiments, without training.

Run: .venv/Scripts/python.exe scripts/report_stage_two.py
Requires the optional analysis dependencies (matplotlib).
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from academic_rewriter.evaluate import evaluate, lcs
from academic_rewriter.prepare import numbers

RUNS = [('bart_baseline', 'BART baseline', '#4473c5'),
        ('bart_adapter_visible', 'Frozen hybrid', '#0c8b81'),
        ('bart_hybrid_unfrozen', 'Full hybrid', '#d16c50')]
BG, INK, MUTED, GRID = '#faf8f3', '#152c3c', '#637580', '#e1e5e4'


def read(path):
    return json.loads((PROJECT / path).read_text(encoding='utf8'))


def jsonl(path):
    return [json.loads(s) for s in (PROJECT / path).read_text(encoding='utf8').splitlines() if s.strip()]


def score(row):
    a, b = row['prediction'].lower().split(), row['reference'].lower().split()
    return 2 * lcs(a, b) / (len(a) + len(b)) if a or b else 0.


def build_data():
    sources = ['data/processed/stats.json', 'runs/lr_tuning_offline/comparison.json']
    models = []
    profiles = {}
    keys = ['baseline', 'hybrid', 'full']
    for run, label, color in RUNS:
        base = f'runs/{run}'
        result, history = read(base + '/result.json'), read(base + '/training_log.json')
        sources += [base + '/result.json', base + '/training_log.json']
        assert len(history) == result['steps'] == 300
        assert [x['step'] for x in history] == list(range(1, 301))
        models.append(dict(run=run, label=label, color=color, result=result, history=history))
    for profile, budget, folder in [('complete128',128,'test-128'), ('historical48',48,'test')]:
        predictions, metrics = [], []
        for model in models:
            base = f"runs/{model['run']}/evaluation/{folder}"
            metric, rows = read(base + '/metrics.json'), jsonl(base + '/predictions.jsonl')
            computed = evaluate(rows)
            for key in ['n', 'whitespace_rouge_l_f1', 'number_multiset_preservation_rate', 'copy_rate']:
                assert math.isclose(computed[key], metric[key], abs_tol=1e-10), (model['run'], profile, key)
            assert metric['split']=='test' and metric['n']==87 and metric['max_new_tokens']==budget
            assert metric['max_length']==128 and metric['num_beams']==1 and not metric['do_sample']
            sources += [base + '/metrics.json', base + '/predictions.jsonl']
            predictions.append(rows)
            metrics.append(metric)
        assert all([r['id'] for r in rows]==[r['id'] for r in predictions[0]] for rows in predictions)
        cases=[]
        for i, aligned in enumerate(zip(*predictions)):
            a=aligned[0]
            assert all((r['source'],r['reference'])==(a['source'],a['reference']) for r in aligned)
            case=dict(index=i+1,id=a['id'],source=a['source'],reference=a['reference'],source_copy_score=score(dict(prediction=a['source'],reference=a['reference'])))
            for name,row in zip(keys,aligned):
                case[name]=dict(text=row['prediction'],score=score(row),
                    copied=row['source'].strip()==row['prediction'].strip(),
                    same_word_sequence=re.findall(r'\w+',row['source'].lower())==re.findall(r'\w+',row['prediction'].lower()),
                    numbers_preserved=numbers(row['source'])==numbers(row['prediction']),
                    source_numbers=numbers(row['source']),output_numbers=numbers(row['prediction']),
                    generated_tokens=row.get('generated_tokens'),
                    generation_limit_reached=row.get('generation_limit_reached'))
            case['delta']=case['full']['score']-case['baseline']['score']
            cases.append(case)
        profiles[profile]=dict(budget=budget,metrics=metrics,cases=cases,source_copy_rouge_l=statistics.mean(c['source_copy_score'] for c in cases))
    assert [c['id'] for c in profiles['complete128']['cases']]==[c['id'] for c in profiles['historical48']['cases']]
    for model,metric in zip(models,profiles['complete128']['metrics']):
        model['test']=metric
    lr=read('runs/lr_tuning_offline/comparison.json')
    trials=lr['runs']+[lr['adaptive_trial']]
    for trial in trials:
        parts=trial['checkpoint'].replace('\\','/').split('/')
        trial['path']='/'.join(parts[parts.index('runs'):])
        trial['history']=read(trial['path']+'/training_log.json')
        sources.append(trial['path']+'/training_log.json')
        vp=trial['path']+'/validation_log.json'
        trial['validation']=read(vp) if (PROJECT/vp).exists() else []
        if trial['validation']: sources.append(vp)
    stats=read('data/processed/stats.json')
    assert sum(s['pairs'] for s in stats['splits'].values())==stats['kept_total']
    assert sum(stats['rejections'].values())+stats['kept_total']==stats['raw_total']
    assessment_path='output/multidimensional/summary.json'
    assessment=read(assessment_path) if (PROJECT/assessment_path).exists() else None
    assessment_cases={}
    if assessment:
        for path,expected in assessment['sources'].items():
            assert hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==expected, 'Stale assessment input: '+path
        sources += [assessment_path,'output/multidimensional/cases.jsonl']
        for row in jsonl('output/multidimensional/cases.jsonl'):
            assessment_cases.setdefault(row['id'],{})[row['model']]=row['metrics']
    human_path='output/human-evaluation/summary.json'
    human_review=read(human_path) if (PROJECT/human_path).exists() else None
    if human_review: sources.append(human_path)
    return dict(title='Academic Text Style Enhancement',stage='Second submission',stats=stats,human_review=human_review,
        models=models,cases=profiles['complete128']['cases'],profiles=profiles,default_profile='complete128',assessment=assessment,assessment_cases=assessment_cases,
        model_keys=keys,learning_rate_trials=trials,
        reference_same_word_sequence_count=sum(re.findall(r'\w+',c['source'].lower())==re.findall(r'\w+',c['reference'].lower()) for c in profiles['complete128']['cases']),
        protocol=dict(seed=42,steps=300,batch_size=2,input_tokens=128,output_tokens=128,
            interpretation='Single-seed pilot; learning rates and trainable capacities differ. No architectural or academic-tone advantage established.',
            evaluation='All three checkpoints evaluated on the same 87 inspected pairs at both 48- and 128-token output budgets. This is a post-inspection decoding diagnostic, not confirmatory model selection.',
            efficiency='Time and CUDA allocation are process observations, not controlled efficiency benchmarks.'),
        sources=[dict(path=p,sha256=hashlib.sha256((PROJECT/p).read_bytes()).hexdigest()) for p in sorted(set(sources))])


def smooth(values, window=20):
    return [statistics.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def figures(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], 'font.size': 10, 'text.color': INK,
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.titleweight': 'bold', 'axes.titlesize': 12, 'axes.titlepad': 18,
        'axes.edgecolor': GRID, 'axes.facecolor': BG, 'figure.facecolor': BG,
        'grid.color': GRID, 'svg.fonttype': 'none', 'svg.hashsalt': 'stage-two',
        'savefig.facecolor': BG, 'axes.spines.top': False, 'axes.spines.right': False})
    models = data['models']
    labels = [m['label'] for m in models]
    colors = [m['color'] for m in models]

    def finish(fig, name, note):
        fig.text(.045, .025, note, fontsize=9, color=MUTED, va='bottom')
        fig.savefig(out / (name + '.png'), dpi=190)
        svg_path = out / (name + '.svg')
        fig.savefig(svg_path, metadata={'Date': None})
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf8').splitlines())+'\n', encoding='utf8', newline='\n')
        plt.close(fig)

    def bars(ax, vals, title, unit='', fmt='.4f', labs=None, cols=None):
        y = range(len(vals))
        ax.barh(y, vals, color=cols or colors[:len(vals)], height=.48)
        ax.set_yticks(list(y), labs or labels[:len(vals)])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals)*1.32)
        ax.set_title(title, loc='left')
        ax.set_xlabel(unit)
        ax.grid(axis='x', alpha=.65)
        ax.set_axisbelow(True)
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.tick_params(axis='both', length=0)
        for i, v in enumerate(vals):
            ax.text(v + max(vals)*.025, i, format(v, fmt), va='center', fontsize=11, weight='bold')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
    fig.subplots_adjust(left=.12, right=.96, top=.78, bottom=.13, hspace=.85, wspace=.65)
    fig.text(.045, .94, 'STAGE TWO  /  RECORDED EXPERIMENTS', color='#0c8b81', fontsize=11, weight='bold')
    fig.text(.045, .875, 'Three models. Little substantive revision.', fontsize=25, weight='bold')
    fig.text(.045, .825, 'BART + Mamba  |  676 revision pairs  |  300 updates  |  seed 42', color=MUTED, fontsize=12)
    bars(axes[0,0], [m['result']['validation_cross_entropy'] for m in models], '01  Validation loss', 'Token cross-entropy / lower is better')
    bars(axes[0,1], [m['test']['cross_entropy'] for m in models], '02  Test loss', 'Token cross-entropy / 87 pairs')
    bars(axes[0,2], [m['test']['whitespace_rouge_l_f1'] for m in models]+[data['profiles']['complete128']['source_copy_rouge_l']], '03  Reference overlap', 'Whitespace ROUGE-L F1 / higher is better', labs=labels+['Copy source'], cols=colors+[MUTED])
    bars(axes[1,0], [m['result']['trainable_parameters']/1e6 for m in models], '04  Trainable capacity', 'Million parameters', '.2f')
    bars(axes[1,1], [m['result']['peak_cuda_allocated_mib'] for m in models], '05  Observed CUDA allocation', 'Peak allocated MiB / process observation', ',.0f')
    ax=axes[1,2]
    for m in models:
        ax.plot([r['step'] for r in m['history']], smooth([r['train_loss'] for r in m['history']]), color=m['color'], lw=1.8)
    ax.set(title='06  Training trajectory', xlabel='Optimizer step', ylabel='Train CE / trailing mean, 20 updates', xlim=(1,300), ylim=(0,2.1))
    ax.grid(axis='y', alpha=.7)
    finish(fig, 'overview', 'Different learning rates and capacities; no architectural advantage established. All three checkpoints tested at a 128-token output budget.\nNumber preservation and overlap do not certify meaning or academic tone. Test examples have been inspected.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharey=True)
    fig.subplots_adjust(left=.065, right=.98, top=.73, bottom=.23, wspace=.12)
    fig.text(.045,.92,'TRAINING DYNAMICS',color='#0c8b81',fontsize=11,weight='bold')
    fig.text(.045,.83,'Mini-batch noise, with the trend in view.',fontsize=23,weight='bold')
    for ax, m in zip(axes, models):
        steps=[r['step'] for r in m['history']]
        values=[r['train_loss'] for r in m['history']]
        ax.plot(steps,values,color=m['color'],alpha=.22,lw=.7,label='Raw mini-batch CE')
        ax.plot(steps,smooth(values),color=m['color'],lw=2.2,label='Trailing mean (20)')
        ax.set(title=m['label'],xlabel='Optimizer step',xlim=(1,300),ylim=(0,4.0))
        ax.grid(axis='y',alpha=.6)
        ax.text(.03,.94,f"LR {m['result']['learning_rate']:.0e}", transform=ax.transAxes,color=MUTED)
        ax.text(.03,.86,f"First 50: {statistics.mean(values[:50]):.3f}  /  Last 50: {statistics.mean(values[-50:]):.3f}", transform=ax.transAxes,fontsize=9)
    axes[0].set_ylabel('Training token cross-entropy')
    axes[0].legend(loc='upper right',bbox_to_anchor=(1,-.2),frameon=False,ncol=2,fontsize=9)
    finish(fig,'training','Raw losses use different two-pair batches at each update. A shared y-axis prevents visual exaggeration.\nThe curve is training loss; it is not a validation history. All three main runs have final-only validation scores.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    fig.subplots_adjust(left=.09,right=.96,top=.72,bottom=.26,wspace=.36)
    fig.text(.045,.92,'VALIDATION-ONLY LEARNING-RATE STUDY',color='#0c8b81',fontsize=11,weight='bold')
    fig.text(.045,.83,'Choose on validation. Keep the budget fixed.',fontsize=23,weight='bold')
    trial_colors=['#d16c50','#4473c5','#0c8b81','#9273b7']
    names=[]
    for trial,c in zip(data['learning_rate_trials'],trial_colors):
        name=f"{trial['learning_rate']:.0e}" + (' plateau' if trial.get('lr_scheduler')=='plateau' else ' constant')
        names.append(name)
        hist=trial['validation']
        if hist:
            axes[0].plot([r['step'] for r in hist],[r['validation_cross_entropy'] for r in hist],'-o',color=c,ms=4,label=name,lw=1.7)
        else:
            axes[0].scatter([300],[trial['validation_cross_entropy']],marker='X',s=90,color=c,label=name+' (final only)',zorder=5)
    axes[0].set(title='Recorded validation evaluations',xlabel='Optimizer step',ylabel='Validation token CE / lower is better',xlim=(-8,315))
    axes[0].grid(axis='y',alpha=.6)
    axes[0].legend(fontsize=8,frameon=False,loc='lower left')
    bars(axes[1],[r['validation_cross_entropy'] for r in data['learning_rate_trials']], 'Final validation loss', 'Token CE / bars begin at zero', labs=names, cols=trial_colors)
    finish(fig,'learning-rate','Same initialization, seed, data, batches and 300 updates. 2e-5 constant has the lowest final validation CE among these trials.\nA single-seed pilot; final-step weights are saved. An observed best score is not a saved best checkpoint.')

    cases=data['cases']
    fig, axes=plt.subplots(1,3,figsize=(15,5.8))
    fig.subplots_adjust(left=.06,right=.97,top=.72,bottom=.27,wspace=.34)
    fig.text(.045,.92,'TEST DIAGNOSTICS  /  87 ALIGNED PAIRS',color='#0c8b81',fontsize=11,weight='bold')
    fig.text(.045,.83,'Three models, one decoding protocol.',fontsize=23,weight='bold')
    for name,m in zip(data['model_keys'],models):
        axes[0].hist([c[name]['score'] for c in cases],bins=[i/10 for i in range(11)],histtype='step',color=m['color'],lw=2,label=m['label'])
    axes[0].set(title='Reference-overlap distribution',xlabel='Whitespace ROUGE-L F1',ylabel='Test examples',xlim=(0,1))
    axes[0].legend(frameon=False,fontsize=9)
    for name,model in zip(['hybrid','full'],models[1:]):
        axes[1].scatter([c['baseline']['score'] for c in cases],[c[name]['score'] for c in cases],s=26,alpha=.7,color=model['color'],edgecolor=BG,lw=.4,label=model['label'])
    axes[1].legend(frameon=False,fontsize=8)
    axes[1].plot([0,1],[0,1],ls='--',color=MUTED,lw=1)
    axes[1].set(title='Paired example scores',xlabel='BART ROUGE-L',ylabel='Hybrid ROUGE-L',xlim=(0,1.03),ylim=(0,1.03))
    ax=axes[2]
    categories=['Numbers preserved','Exact source copies','Same word sequence']
    for offset,name,m in zip([-.25,0,.25],data['model_keys'],models):
        vals=[sum(c[name]['numbers_preserved'] for c in cases),sum(c[name]['copied'] for c in cases),sum(c[name]['same_word_sequence'] for c in cases)]
        ax.barh([i+offset for i in range(3)],vals,height=.21,color=m['color'])
        for i,v in enumerate(vals): ax.text(v+1.8,i+offset,f'{v}/87',va='center',fontsize=10)
    ax.set_yticks([0,1,2],categories,fontsize=9)
    ax.set(title='Diagnostic counts',xlabel='Test examples',xlim=(0,108))
    ax.invert_yaxis()
    for ax in axes: ax.grid(axis='y',alpha=.5); ax.set_axisbelow(True)
    finish(fig,'test-diagnostics','Same word sequence ignores case/punctuation; only 4/87 human references retain the source sequence.\nNumber counts include examples without numbers. Copying can earn high overlap without performing the requested revision.\nThese checks do not establish meaning preservation or better academic tone. Input: 128 tokens; generation: at most 128 new tokens.')

    fig,ax=plt.subplots(figsize=(14,5.2))
    fig.subplots_adjust(left=.03,right=.97,top=.72,bottom=.2)
    fig.text(.045,.92,'MODEL PATH',color='#0c8b81',fontsize=11,weight='bold')
    fig.text(.045,.81,'Pretrained context, with a selective-state adapter.',fontsize=23,weight='bold')
    ax.set(xlim=(0,14),ylim=(0,3));ax.axis('off')
    boxes=[(.05,1.1,1.55,1.0,'Source','Revision input','#e6ecee'),(2.05,1.1,2.15,1.0,'BART encoder','Context states H','#e1e9f6'),(4.75,1.1,2.7,1.0,'Mamba adapter','M = Mamba(LN(H))','#d7eeea'),(8.0,1.1,2.2,1.0,'Gated fusion','Z = H + G * M','#f5e4dc'),(10.75,1.1,2.6,1.0,'BART decoder','Suggested revision','#e1e9f6')]
    for x,y,w,h,title,subtitle,c in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',facecolor=c,edgecolor='none'))
        ax.text(x+w/2,y+.64,title,ha='center',weight='bold',fontsize=11)
        ax.text(x+w/2,y+.29,subtitle,ha='center',fontsize=9,color=MUTED)
    for a,b in zip(boxes,boxes[1:]):
        ax.add_patch(FancyArrowPatch((a[0]+a[2]+.06,1.6),(b[0]-.09,1.6),arrowstyle='-|>',mutation_scale=14,color=MUTED,lw=1.3))
    ax.plot([3.1,3.1,9.1],[2.18,2.65,2.65],color='#d16c50',lw=1.3)
    ax.annotate('',xy=(9.1,2.18),xytext=(9.1,2.65),arrowprops=dict(arrowstyle='->',color='#d16c50',lw=1.3))
    ax.text(6.2,2.83,'Encoder skip path H',ha='center',fontsize=9,color='#d16c50')
    ax.text(7,.48,'G = sigmoid(W[H; M] + b)     |     Author reviews the suggestion',ha='center',fontsize=11,color=MUTED)
    finish(fig,'architecture','The adapter uses a sequential PyTorch reference scan. Official-kernel parity and optimized speedups remain unverified.\nTransformer attention remains in the hybrid; the complete model is not linear-time.')

    if data.get('assessment') and 'bertscore' in data['assessment']['dimensions']['semantic']:
        summary=data['assessment']
        keys=list(summary['models'])
        names=[summary['models'][k]['label'].replace(' (control)','') for k in keys]
        palette=colors+['#637580','#9273b7']
        fig,axes=plt.subplots(1,3,figsize=(16,6.1))
        fig.subplots_adjust(left=.14,right=.97,top=.73,bottom=.28,wspace=.65)
        fig.text(.045,.93,'MULTIDIMENSIONAL EVALUATION / 87 INSPECTED PAIRS',color='#0c8b81',fontsize=11,weight='bold')
        fig.text(.045,.84,'High preservation scores can hide a copying failure.',fontsize=23,weight='bold')
        fields=[('bertscore_source_f1','Source BERTScore F1',1.),('bidirectional_entailment_min','Bidirectional NLI / minimum',1.),('word_edit_rate','Word edit rate (%)',100.)]
        for ax,(field,title,scale) in zip(axes,fields):
            rows=[summary['models'][k]['metrics'][field] for k in keys]
            values=[r['mean']*scale for r in rows]
            errors=[[max(0,(r['mean']-r['ci95'][0])*scale) for r in rows],[max(0,(r['ci95'][1]-r['mean'])*scale) for r in rows]]
            ax.barh(range(len(keys)),values,color=palette,height=.5,xerr=errors,error_kw=dict(ecolor=INK,capsize=2,lw=.8))
            ax.set_yticks(range(len(keys)),names,fontsize=9);ax.invert_yaxis()
            ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.6);ax.set_axisbelow(True)
            ax.set_xlim(0,1.24 if scale==1 else max(values)*1.45)
            if scale==1: ax.set_xticks([0,.25,.5,.75,1.])
            for i,value in enumerate(values):
                right=rows[i]['ci95'][1]*scale
                ax.text(right+(.025 if scale==1 else .7),i,format(value,'.4f' if scale==1 else '.2f'),va='center',fontsize=10)
            ax.spines[['left','bottom']].set_visible(False);ax.tick_params(length=0)
        finish(fig,'multidimensional','Raw, unrescaled DistilRoBERTa BERTScore (layer 5). Generic DeBERTa NLI; not a calibrated equivalence test or published MIS.\nIntervals resample 10 connected document groups (1,000 draws). Human references are not adjudicated semantic gold.\nAcademic-style labels, independent fluency ratings and successful-rewrite judgments remain pending.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='output/stage-two')
    args=parser.parse_args()
    out=PROJECT / args.output
    out.mkdir(parents=True,exist_ok=True)
    data=build_data()
    (out/'results-data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
    figures(data,out)
    template=(PROJECT/'scripts/templates/stage_two.html').read_text(encoding='utf8')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    html=template.replace('__RESULTS_DATA__',payload)
    names=['overview','training','learning-rate','test-diagnostics','architecture']
    if data.get('assessment') and 'bertscore' in data['assessment']['dimensions']['semantic']: names.append('multidimensional')
    else: html=html.replace('__FIGURE_MULTIDIMENSIONAL__','<p>Multidimensional evaluation has not been run.</p>')
    for name in names:
        svg=(out/(name+'.svg')).read_text(encoding='utf8')
        svg=svg[svg.index('<svg'):]
        # Inline figures share a document: namespace their IDs and references.
        identifiers = re.findall(r'id="([^"]+)"', svg)
        for identifier in identifiers:
            svg=svg.replace('id="'+identifier+'"', 'id="'+name+'-'+identifier+'"')
            svg=svg.replace('url(#'+identifier+')', 'url(#'+name+'-'+identifier+')')
            svg=svg.replace('href="#'+identifier+'"', 'href="#'+name+'-'+identifier+'"')
        html=html.replace('__FIGURE_'+name.upper().replace('-','_')+'__',svg)
    assert '__FIGURE_' not in html and '__RESULTS_DATA__' not in html
    (out/'index.html').write_text(html,encoding='utf8',newline='\n')
    print(json.dumps(dict(output=str(out),models=len(data['models']),cases=len(data['cases']),source_files=len(data['sources'])),indent=2))


if __name__=='__main__':
    main()
