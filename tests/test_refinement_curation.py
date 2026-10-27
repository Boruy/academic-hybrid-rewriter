import importlib.util
from pathlib import Path
import unittest

from academic_rewriter.refinement import approve_pair, approve_no_edit, validate_augmentation
from academic_rewriter.refinement_metrics import summarize
from academic_rewriter.human_eval import text_digest


class RefinementCurationTests(unittest.TestCase):
    def review(self,who,**extra):
        return dict(rater_id=who,reviewer_type='human',category='Academic Style',meaning=5,fluency=5,source_fluency=4,source_suitable=False,source_needs_revision=True,target_suitable=True,major_fact_error=False,**extra)

    def test_ai_or_single_reviewer_cannot_approve(self):
        a=self.review('a');b=self.review('b')
        self.assertTrue(approve_pair([a,b]));self.assertFalse(approve_pair([a]));self.assertFalse(approve_pair([a,a]))
        with self.assertRaisesRegex(ValueError,'human'):approve_pair([a,dict(b,reviewer_type='ai_assisted')])
        self.assertFalse(approve_pair([a,dict(b,major_fact_error=True)]))
        self.assertTrue(approve_no_edit([dict(a,source_needs_revision=False,source_suitable=True),dict(b,source_needs_revision=False,source_suitable=True)]))

    def test_augmentation_never_enters_heldout_and_preserves_terms(self):
        parent=dict(split='train',doc_id='paper',target='The method may improve accuracy by 2 percent.')
        row=dict(split='train',doc_id='paper',source='The method may make accuracy better by 2 percent.',target=parent['target'],protected_terms=['method','accuracy'])
        validate_augmentation(row,parent)
        with self.assertRaisesRegex(ValueError,'training'):validate_augmentation(dict(row,split='test'),parent)
        with self.assertRaisesRegex(ValueError,'guards'):validate_augmentation(dict(row,source=row['source'].replace('2','3')),parent)
        with self.assertRaisesRegex(ValueError,'guards'):validate_augmentation(dict(row,source=row['source'].replace('may ','')),parent)
        with self.assertRaisesRegex(ValueError,'technical'):validate_augmentation(dict(row,source=row['source'].replace('method','model')),parent)

    def test_pending_metrics_and_separate_denominators(self):
        cases=[dict(id='needs',model='bart',doc_id='p1',source='we used data',prediction='We used a dataset.',needs_revision=True,source_judgment_human_verified=True),dict(id='good',model='bart',doc_id='p2',source='The method may improve accuracy.',prediction='The method improves accuracy.',needs_revision=False,source_judgment_human_verified=True)]
        pending=summarize(cases,[])
        self.assertIsNone(pending['models']['bart']['effective_revision_rate'])
        ratings=[]
        for case in cases:
            for who in ['a','b']:
                ratings.append(dict(id=case['id'],model='bart',rater_id=who,reviewer_type='human',text_digest=text_digest(case['source'],case['prediction']),meaning=5,academic=4,fluency=5,academic_improvement=1,major_fact_error=False,unnecessary_edit=case['id']=='good'))
        result=summarize(cases,ratings)['models']['bart']
        self.assertEqual(result['provisional_effective_revision_rate']['mean'],1)
        self.assertEqual(result['provisional_unnecessary_edit_rate']['mean'],1)
        self.assertIsNone(result['effective_revision_rate'])
        copied=[dict(cases[0],prediction=cases[0]['source'])]
        for r in ratings[:2]:r['text_digest']=text_digest(copied[0]['source'],copied[0]['prediction'])
        self.assertEqual(summarize(copied,ratings[:2])['models']['bart']['provisional_effective_revision_rate']['mean'],0)

if __name__=='__main__':unittest.main()
