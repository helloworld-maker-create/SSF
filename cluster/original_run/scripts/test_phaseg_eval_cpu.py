"""Identity-blind exports and the previously corrected native metric units."""
import unittest
from phaseg_health_pack_cpu import anonymize, blank_row
from phaseg_eval_inputs import selected_rows, video_dimension_raw


class EvaluationContractTests(unittest.TestCase):
    def test_frozen_counts(self):
        for arm in ['baseline','ssf']:
            for outer in [200,400,600]:
                for weight in ['raw','EMA']:
                    for seed in [2026,2027]:
                        self.assertEqual(len(selected_rows(arm,outer,weight,seed)),48 if outer==600 else 8)

    def test_blind_pack_keeps_all_cases_and_no_prefilled_judgement(self):
        items=[dict(source_video_id=f'model{i%4}_p{i//8}_s{i%2}',
                    prompt=f'prompt{i//8}',seed=2026+i%2,
                    action_applicable=i//8<4) for i in range(64)]
        # Unique IDs, unlike the grouped mock formula above.
        for i,x in enumerate(items):x['source_video_id']+=f'_{i}'
        coded=anonymize(items,b'frozen-test-salt')
        self.assertEqual({x['source_video_id'] for x in coded},{x['source_video_id'] for x in items})
        self.assertEqual(coded,anonymize(items,b'frozen-test-salt'))
        self.assertEqual(len({x['clip_id'] for x in coded}),64)
        for item in coded:
            row=blank_row(item)
            self.assertEqual([row[x] for x in ['O','T','severe_deformation','severe_identity_or_object_failure']],['']*4)
            self.assertEqual(row['A'],'' if item['action_applicable'] else 'NA')
            self.assertNotIn('source_video_id',row)
            self.assertNotIn('model_identity',row)

    def test_musiq_video_unit_differs_from_dimension_aggregate(self):
        self.assertEqual(video_dimension_raw('imaging_quality',73.5),.735)
        self.assertEqual(video_dimension_raw('subject_consistency',.9735),.9735)
        self.assertEqual(video_dimension_raw('dynamic_degree',0),0)
        self.assertEqual(video_dimension_raw('dynamic_degree',1),1)


if __name__=='__main__':unittest.main()
