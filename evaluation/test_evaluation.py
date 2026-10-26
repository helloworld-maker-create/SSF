"""CPU-only regression tests; no metric/model imports, GPU work or downloads."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HERE, digest, fingerprint, generation_record, gpu_preflight,
                    inspect_video, load_config, load_panel, make_action_aliases, official_entries,
                    paired_summary, read, score_points, stat_identity, write_once)
from run import plan, prepare, summarize
from score_worker import canonical_result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        # Keep tiny generated fixtures for inspection; no deletion of old data.
        self.root = Path(tempfile.mkdtemp(prefix='ssf_eval_cpu_'))

    def config(self):
        cfg = read(HERE/'config.example.json')
        cfg.update(video_root=str(self.root/'videos'), output=str(self.root/'results'),
                   metric_python='/unused/python', vbench_cache='/unused/cache',
                   require_generation_records=False, record_pattern=None, seeds=[2026,2027],
                   dimensions=['human_action'])
        panel = copy.deepcopy(load_panel(HERE/'data/phaseG48.json'))
        panel['prompts'] = [p for p in panel['prompts'] if 'human_action' in p['dims']][:1]
        cfg['panel'] = str(self.root/'panel.json'); write_once(cfg['panel'], panel)
        cfg_path = self.root/'config.json'; write_once(cfg_path, cfg)
        return load_config(cfg_path), panel['prompts'][0]

    def videos(self, cfg, row):
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                p = Path(cfg['video_root'])/arm['name']/str(seed)/(row['prompt_id']+'.mp4')
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'unit metadata fixture only')

    def test_frozen_panel_counts_and_memberships(self):
        for name, count in [('phaseG48',48),('phaseCD96',96),('official944',944)]:
            panel = load_panel(HERE/'data'/(name+'.json'))
            self.assertEqual(len(panel['prompts']), count)
            paths = {r['prompt_id']:self.root/(r['prompt_id']+'.mp4') for r in panel['prompts']}
            for dim in read(HERE/'metric_assets.json')['submodules']:
                entries = official_entries(panel['prompts'], dim, paths)
                self.assertEqual(len(entries), sum(dim in r['dims'] for r in panel['prompts']))
                if dim == 'object_class':
                    self.assertTrue(all('object_class' in e.get('auxiliary_info',{}) for e in entries))

    def test_source_manifests_match(self):
        for item in read(HERE/'data/provenance.json')['panels']:
            self.assertEqual(digest(HERE/'data'/item['file']),item['sha256'])
            self.assertEqual(digest(HERE/'data'/item['text_file']),item['text_sha256'])
            self.assertEqual((HERE/'data'/item['text_file']).read_text().splitlines(),
                             [p['prompt'] for p in read(HERE/'data'/item['file'])['prompts']])

    def test_human_action_alias_keeps_exact_video_inode(self):
        p = self.root/'prompt-id.mp4'; p.write_bytes(b'unique-video-bytes')
        e = dict(prompt_en='A person is playing flute', dimension=['human_action'], video_list=[str(p)])
        result = make_action_aliases(self.root/'aliases',[e])
        alias = Path(result[0]['video_list'][0])
        self.assertEqual(alias.name,'A person is playing flute-0.mp4')
        self.assertEqual(stat_identity(alias)['inode'],stat_identity(p)['inode'])
        self.assertEqual(digest(alias),digest(p))

    def test_write_once_preserves_evidence(self):
        p = self.root/'evidence.json'; write_once(p,{'x':1}); write_once(p,{'x':1})
        with self.assertRaises(ValueError): write_once(p,{'x':2})
        self.assertEqual(read(p),{'x':1})

    def test_no_color_roundoff_win(self):
        r = paired_summary([1e-12,-1e-12,0.0],1e-8)
        self.assertEqual((r['wins'],r['ties'],r['losses']),(0,3,0))
        self.assertEqual(r['sign_test_two_sided'],1.0)

    def test_score_units_musiq_and_other_metrics(self):
        self.assertEqual(score_points('imaging_quality',67),67)
        self.assertEqual(score_points('subject_consistency',.67),67)
        with self.assertRaises(ValueError): score_points('dynamic_degree',float('nan'))

    def test_missing_is_not_zero(self):
        self.assertEqual(paired_summary([]),dict(status='NO_PAIRED_DATA',n_prompts=0))

    def test_config_rejects_raw_ema_mismatch(self):
        cfg,row = self.config(); cfg['arms'][1]['weight_kind']='raw'
        p=self.root/'bad.json'; write_once(p,cfg)
        with self.assertRaises(ValueError): load_config(p)

    def test_missing_inputs_stop_prepare(self):
        cfg,row=self.config()
        with self.assertRaisesRegex(ValueError,'MISSING_INPUTS'): prepare(cfg)
        self.assertFalse((Path(cfg['output'])/'manifest.json').exists())

    def test_plan_never_checks_videos_or_gpu(self):
        cfg,row=self.config()
        with patch('run.inspect_video',side_effect=AssertionError), patch('run.gpu_preflight',side_effect=AssertionError):
            r=plan(cfg)
        self.assertFalse(r['GPU_execution']); self.assertEqual(r['expected_existing_videos'],4)

    def test_partial_reports_never_claim_complete(self):
        cfg,row=self.config(); self.videos(cfg,row)
        with patch('run.inspect_video',return_value={'fixture':True}): manifest=prepare(cfg)
        report,rows=summarize(manifest)
        self.assertEqual(report['status'],'PARTIAL_NOT_A_COMPLETE_EVALUATION')
        self.assertFalse(report['official_Total_computed']); self.assertEqual(rows,[])
        self.assertEqual(report['comparisons'][0]['all_seed_paired_prompts'],0)

    def test_prompt_is_unit_seeds_averaged_first(self):
        cfg,row=self.config(); self.videos(cfg,row)
        with patch('run.inspect_video',return_value={'fixture':True}): manifest=prepare(cfg)
        for job in manifest['jobs']:
            folder=Path(job['folder']); inp=read(folder/'input.json')
            score=1. if inp['arm']['name']=='baseline' else (3. if inp['seed']==2026 else 5.)
            write_once(folder/'result.json',dict(aggregate_native=score/100.,eligible=1,missing_prompt_ids=[],
                per_video=[dict(prompt_id=row['prompt_id'],native_score=score/100.,score_points=score)]))
            write_once(folder/'COMPLETED.json',dict(input_hash=fingerprint(inp),result_sha256=digest(folder/'result.json')))
        report,rows=summarize(manifest); stats=report['comparisons'][0]['statistics']
        self.assertEqual(stats['n_prompts'],1); self.assertEqual(stats['mean'],3.)
        self.assertEqual(report['status'],'COMPLETE'); self.assertEqual(len(rows),4)

    def test_noise_mismatch_is_rejected(self):
        cfg,row=self.config(); self.videos(cfg,row)
        cfg.update(require_generation_records=True,record_pattern='{arm}/{seed}/{prompt_id}.json')
        for i,arm in enumerate(cfg['arms']):
            arm['checkpoint_sha256']=str(i)*64
            for seed in cfg['seeds']:
                p=Path(cfg['video_root'])/arm['name']/str(seed)/(row['prompt_id']+'.mp4')
                record=dict(prompt=row['prompt'],seed=seed,weight='EMA',checkpoint_SHA256=arm['checkpoint_sha256'],
                    video=dict(lossless=True,SHA256=digest(p),RGB_SHA256='f'*64),latent_finite=True,
                    initial_noise_SHA256=str(i)*64,renoise_trace=['n']*21)
                write_once(p.with_suffix('.json'),record)
        with patch('run.inspect_video',return_value={'fixture':True}):
            with self.assertRaisesRegex(ValueError,'PAIRED_NOISE_MISMATCH'): prepare(cfg)

    def test_detector_missing_retained_explicitly(self):
        inp=dict(dimension='object_class',inputs=[dict(prompt='a dog',prompt_id='p1'),dict(prompt='a cat',prompt_id='p2')])
        entries=[dict(prompt_en='a dog',video_list=[str(self.root/'a.mp4')]),dict(prompt_en='a cat',video_list=[str(self.root/'b.mp4')])]
        result=canonical_result(1.,[dict(video_path=str(self.root/'a.mp4'),video_results=1.)],inp,entries)
        self.assertEqual(result['missing_prompt_ids'],['p2']); self.assertEqual(len(result['per_video']),1)
        inp['dimension']='subject_consistency'
        with self.assertRaises(ValueError): canonical_result(1.,[],inp,entries)

    def test_gpu_preflight_rejects_other_process(self):
        gpu='0, GPU-unit, Test GPU, 0, 140000, 144000\n'
        apps='GPU-unit, 999999, another_users_job, 100\n'
        with patch('common.subprocess.check_output',side_effect=[gpu,apps]):
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_RESOURCE'): gpu_preflight('0')

    def test_stat_identity_detects_input_changes(self):
        p=self.root/'input.mp4'; p.write_bytes(b'one'); before=stat_identity(p)
        p.write_bytes(b'changed'); self.assertNotEqual(before,stat_identity(p))

    def test_video_contract_rejects_preview_and_wrong_frame_count(self):
        info=dict(codec_name='h264',pix_fmt='gbrp',width=832,height=480,nb_read_frames='81',avg_frame_rate='16/1')
        with patch('common.subprocess.check_output',return_value=json.dumps({'streams':[info]})):
            self.assertEqual(inspect_video('/fixture.mp4')['pix_fmt'],'gbrp')
        for change in [dict(pix_fmt='yuv420p'),dict(nb_read_frames='80')]:
            with patch('common.subprocess.check_output',return_value=json.dumps({'streams':[{**info,**change}]})):
                with self.assertRaises(ValueError): inspect_video('/fixture.mp4')

    def test_all_metric_asset_templates_registered(self):
        spec=read(HERE/'metric_assets.json')
        def visit(v):
            if isinstance(v,str) and ('${CACHE}' in v or '${VBENCH}' in v):
                self.assertTrue(any(k==v or k.startswith(v.rstrip('/')+'/') for k in spec['verified_assets']),v)
            elif isinstance(v,dict):
                for item in v.values():visit(item)
            elif isinstance(v,list):
                for item in v:visit(item)
        visit(spec['submodules'])


if __name__=='__main__':
    unittest.main(verbosity=2)
