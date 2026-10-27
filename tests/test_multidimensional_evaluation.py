"""Checks for scientific failure modes in multidimensional evaluation."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from academic_rewriter.assess import lexical,nli_probabilities,require_untruncated
from academic_rewriter.evaluation_common import (load_aligned,paired_difference,quadratic_kappa,write_json)
from academic_rewriter.human_eval import successful_rewrite,validate_rating,summarize
from academic_rewriter.train_style import validate_annotations

THRESHOLDS=dict(required_raters=2,minimum_meaning=4,minimum_academic=4,minimum_fluency=4,minimum_improvement=1,thresholds_calibrated=False)

class EvaluationTests(unittest.TestCase):
    def test_identity_control_cannot_be_success_despite_maximum_ratings(self):
        rating=dict(meaning=5,academic=5,fluency=5,academic_improvement=2,major_fact_error=False)
        self.assertFalse(successful_rewrite([rating,rating],'Same text.','Same text.',THRESHOLDS))
        self.assertFalse(successful_rewrite([],'Source.','Changed.',THRESHOLDS))
        self.assertTrue(successful_rewrite([rating,rating],'We used a lot of data.','We used a large dataset.',THRESHOLDS))
        self.assertFalse(successful_rewrite([rating,dict(rating,major_fact_error=True)],'Source.','Changed.',THRESHOLDS))
        self.assertFalse(successful_rewrite([rating,dict(rating,academic_improvement=0)],'Source.','Changed.',THRESHOLDS))

    def test_blank_and_boolean_ratings_are_not_accepted_as_scores(self):
        self.assertFalse(validate_rating(dict(meaning=None)))
        with self.assertRaises(ValueError):validate_rating(dict(meaning=True,academic=5,fluency=5,academic_improvement=2,major_fact_error=False))
        with self.assertRaises(ValueError):validate_rating(dict(meaning=5,academic=5,fluency=5,academic_improvement=2,major_fact_error='false'))

    def test_nli_label_order_and_softmax_are_explicit(self):
        p=nli_probabilities([1000,998,999],dict(contradiction=0,entailment=1,neutral=2))
        self.assertGreater(p['contradiction'],p['neutral'])
        self.assertGreater(p['neutral'],p['entailment'])
        self.assertAlmostEqual(sum(p.values()),1.)

    def test_edit_rate_and_protected_flags_do_not_claim_semantic_equivalence(self):
        identity=lexical('The model may improve 5 cases.','The model may improve 5 cases.','The model could improve five cases.')
        self.assertEqual(identity['word_edit_rate'],0.)
        changed=lexical('The model may not improve 5 cases.','The model improves 50 cases.','Reference.')
        self.assertEqual(changed['numbers_preserved'],0.)
        self.assertEqual(changed['negation_changed'],1.)
        self.assertEqual(changed['hedging_changed'],1.)

    def test_paired_group_bootstrap_and_undefined_agreement(self):
        a=[2.,3.,5.];b=[1.,2.,4.];groups=['one','one','two']
        result=paired_difference(a,b,groups,200,17)
        self.assertEqual(result['mean'],1.)
        self.assertEqual(result['ci95'],[1.,1.])
        self.assertEqual(result,paired_difference(a,b,groups,200,17))
        self.assertAlmostEqual(quadratic_kappa([1,2,3,4,5],[1,2,3,4,5]),1.)
        self.assertAlmostEqual(quadratic_kappa([1,2,3,4,5],[5,4,3,2,1]),-1.)
        self.assertIsNone(quadratic_kappa([5,5],[5,5]))

    def test_long_metric_inputs_fail_instead_of_silent_truncation(self):
        class Tokenizer:
            def __call__(self,a,b=None,truncation=False):return {'input_ids':list(range(len(a.split())+(len(b.split()) if b else 0)+2))}
        with self.assertRaises(ValueError):require_untruncated(Tokenizer(),['one two three'],4)
        with self.assertRaises(ValueError):require_untruncated(Tokenizer(),['one two'],5,pairs=['three four'])

    def test_alignment_uses_ids_and_rejects_duplicate_or_wrong_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);case=dict(id='a',doc_id='d',group_id='d',source='Source',target='Target')
            (root/'data.jsonl').write_text(json.dumps(case)+'\n')
            prediction=dict(id='a',source='Source',reference='Target',prediction='Output')
            (root/'p.jsonl').write_text(json.dumps(prediction)+'\n')
            config=dict(data='data.jsonl',models=[dict(id='model',label='Model',predictions='p.jsonl')],controls=['copy_source'])
            _,c,_=load_aligned(root,config);self.assertEqual(c['copy_source'],['Source'])
            (root/'p.jsonl').write_text(json.dumps(dict(prediction,reference='wrong'))+'\n')
            with self.assertRaises(ValueError):load_aligned(root,config)
            (root/'p.jsonl').write_text((json.dumps(prediction)+'\n')*2)
            with self.assertRaises(ValueError):load_aligned(root,config)

    def test_seen_cases_cannot_be_called_fresh_confirmatory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data/processed').mkdir(parents=True)
            case=dict(id='case',doc_id='known',group_id='known',source='Source',target='Target')
            for split in ['train','validation','test']:(root/'data/processed'/f'{split}.jsonl').write_text(json.dumps(case)+'\n')
            (root/'fresh.jsonl').write_text(json.dumps(dict(case,id='new-id'))+'\n')
            config=dict(data='fresh.jsonl',models=[],evaluation_status='fresh_confirmatory')
            with self.assertRaises(ValueError):load_aligned(root,config)

    def test_style_intents_and_generator_overlap_are_rejected(self):
        rows=[dict(text=f'Unique {split} {label}',academic=label,doc_id=f'{split}-{label}',split=split,label_source='human',guideline_version='test-fixture') for split in ['train','validation','test'] for label in [0,1]]
        self.assertEqual(len(validate_annotations(rows,set(),set())['train']),2)
        with self.assertRaises(ValueError):validate_annotations(rows,{'train-0'},set())
        with self.assertRaises(ValueError):validate_annotations([dict(r,academic='style') for r in rows],set(),set())
        leaked=[dict(r) for r in rows];leaked[-1]['doc_id']='train-0'
        with self.assertRaises(ValueError):validate_annotations(leaked,set(),set())

    def test_unfilled_human_packets_have_no_success_rate(self):
        from academic_rewriter.human_eval import text_digest
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);uid='blind';source='Source.';candidate='Changed.'
            key=dict(items={uid:dict(case_id='case',model='model',group_id='group',source=source,candidate=candidate,digest=text_digest(source,candidate))},raters=['rater_1','rater_2'],labels={'model':'Model'},thresholds=THRESHOLDS,evaluation_status='test_fixture',case_ids=['case'])
            write_json(root/'key.json',key)
            files=[]
            for name in key['raters']:
                path=root/(name+'.jsonl');files.append(str(path));path.write_text(json.dumps(dict(item_id=uid,rater_id=name,source=source,candidate=candidate,meaning=None,academic=None,fluency=None,academic_improvement=None,major_fact_error=None))+'\n')
            summarize(argparse.Namespace(key=str(root/'key.json'),ratings=files,summary=str(root/'summary.json')))
            result=json.loads((root/'summary.json').read_text())
            self.assertEqual(result['status'],'pending_ratings')
            self.assertIsNone(result['models']['model']['successful_rewrite_rate'])

if __name__=='__main__':unittest.main()
