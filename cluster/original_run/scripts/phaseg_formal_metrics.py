"""Frozen official-member scoring of Phase G training outputs, no generation.

Uses the same metric environment, weights and filename helper qualified in A.
Must be explicitly launched on one freshly preflighted idle GPU. No controller.
"""
from g_common import *
import argparse, faulthandler, gc, importlib, random, sys, traceback
from phaseg_eval_inputs import selected_rows, load_generation_records
from phaseg_formal_control import authorized
from reference_metrics import verify_scoring_inputs
faulthandler.enable(all_threads=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['baseline','ssf'],required=True)
    ap.add_argument('--outer',type=int,choices=[200,400,600],required=True)
    ap.add_argument('--weight',choices=['raw','EMA'],required=True)
    ap.add_argument('--seed',type=int,choices=[2026,2027],required=True)
    ap.add_argument('--dimensions',default='all')
    a=ap.parse_args();authorized(a.arm);check_source()
    c=read(ROOT/'contracts/evaluation_manifest.json')
    dims=c['metrics'] if a.dimensions=='all' else a.dimensions.split(',')
    assert dims and len(set(dims))==len(dims) and set(dims)<=set(c['metrics'])
    # Validate only this checkpoint/seed's registered scoring inputs, not old
    # libraries of videos or all model weights on every metric invocation.
    rows=selected_rows(a.arm,a.outer,a.weight,a.seed)
    records=load_generation_records(rows)
    expected_checkpoint=read(ROOT/f'training/{a.arm}/checkpoints/outer_{a.outer:06d}.COMMITTED.json')
    assert expected_checkpoint['status']=='COMMITTED'
    assert {r['checkpoint_SHA256'] for r in records.values()}=={expected_checkpoint['SHA256']}
    original=read(c['reuse_contract_source'])
    assert sha(c['reuse_contract_source'])==c['reuse_contract_SHA256']
    source=original['prompt_source'];assert sha(source['path'])==source['sha256']
    info=read(source['path'])
    entries=[{**entry,'video_list':[records[entry['prompt_en']]['video']['path']]}
             for entry in info if entry['prompt_en'] in records]
    assert len(entries)==len(rows)
    for row in rows:
        entry=next(e for e in entries if e['prompt_en']==row['prompt'])
        assert set(entry['dimension'])==set(json.loads(row['official_memberships']))
    folder=ROOT/f'scoring/formal/{a.arm}/outer{a.outer}/{a.weight}/{a.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    metadata_path=folder/'eligible_full_info.json'
    if metadata_path.exists():assert read(metadata_path)==entries
    else:dump(metadata_path,entries)
    category='stage_B_generation_scoring_GPUh' if a.arm=='baseline' else 'stage_C_generation_scoring_GPUh'
    budget_guard(category,60)
    lease=acquire_gpu();ok=False;started=False
    tag=f'formal_metrics_{a.arm}_{a.outer}_{a.weight}_{a.seed}_{time.time_ns()}'
    try:
        pre=resources(True)
        dump(ROOT/'state'/(tag+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category=category,resource=pre))
        started=True
        import torch, numpy as np, vbench
        from vbench.utils import init_submodules
        from vbench.distributed import dist_init
        from importlib import metadata
        expected=c['metric_environment']
        assert sys.executable==expected['executable']
        assert str(torch.__version__)==expected['torch_runtime_version'] and np.__version__==expected['numpy_runtime_version']
        for name,version in expected['packages'].items():assert metadata.version(name)==version
        for path,digest in c['metric_module_SHA256'].items():assert sha(path)==digest
        assert sha(c['metric_normalization']['path'])==c['metric_normalization']['sha256']
        old_scripts=ROOT.parent/'goalB_20261014_095243/scripts';sys.path.insert(0,str(old_scripts))
        from b_common import block_downloads
        import metric_filename_contract as action
        assert sha(c['human_action']['helper'])==c['human_action']['helper_SHA256']
        block_downloads();resources(True);torch.cuda.set_device(0);dist_init();torch.set_grad_enabled(False)
        for dim in dims:
            eligible=[e for e in entries if dim in e['dimension']]
            if not eligible:continue
            out=folder/(dim+'_eval_results.json');prov=folder/(dim+'_provenance.json')
            if out.exists():
                previous=read(prov)
                assert previous['result_SHA256']==sha(out)
                assert previous['checkpoint_SHA256']==expected_checkpoint['SHA256']
                assert previous['eligible']==len(eligible)
                continue
            assert not prov.exists(),'PRESERVE_PARTIAL_SCORE_FOR_REVIEW'
            budget_guard(category,60);resources(False)
            path=metadata_path;metric_entries=entries
            if dim=='human_action':
                path,alias_manifest,metric_entries=action.prepare_action_aliases(folder/'human_action_aliases',entries,records)
            random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
            sub=init_submodules([dim],local=True,read_frame=False)[dim]
            assets=verify_scoring_inputs(sub);module=importlib.import_module('vbench.'+dim)
            result=getattr(module,'compute_'+dim)(str(path),torch.device('cuda:0'),sub,imaging_quality_preprocessing_mode='longer')
            torch.cuda.synchronize()
            expected_paths={e['video_list'][0] for e in metric_entries if dim in e['dimension']}
            seen=[r['video_path'] for r in result[1]]
            assert len(seen)==len(set(seen)) and set(seen)<=expected_paths
            if dim not in ['object_class','multiple_objects']:assert set(seen)==expected_paths
            assert np.isfinite(result[0]) and all(np.isfinite(float(r['video_results'])) for r in result[1])
            dump(out,{dim:result})
            dump(prov,dict(arm=a.arm,outer=a.outer,weight=a.weight,seed=a.seed,dimension=dim,
                checkpoint_SHA256=expected_checkpoint['SHA256'],result_SHA256=sha(out),metadata_SHA256=sha(path),
                source_SHA256=sha(module.__file__),eligible=len(expected_paths),retained=len(seen),missing=sorted(expected_paths-set(seen)),
                verified_metric_assets=assets,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                scoring_source='lossless RGB originals; official membership only',official_Total_computed=False,
                units='unmodified native output; MUSIQ per-video /100 before common raw-score x100 analysis',resource=pre))
            print('PHASEG_METRIC_PASS',a.arm,a.outer,a.weight,a.seed,dim,result[0],flush=True)
            gc.collect();torch.cuda.empty_cache()
        ok=True
    finally:
        cleaned=False
        try:
            if 'torch' in sys.modules:
                import torch.distributed as dist
                if dist.is_initialized():dist.destroy_process_group()
            cleaned=True
        finally:
            if started:dump(ROOT/'state'/(tag+'_finished.json'),dict(pid=os.getpid(),ended_at_epoch=time.time(),success=ok and cleaned))
            lease.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/formal_scoring_failures.jsonl',dict(at=time.time(),traceback=traceback.format_exc()))
        raise
