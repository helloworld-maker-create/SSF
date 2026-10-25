"""Stage A official-member scoring only, existing metric code and assets."""
from g_common import *
import argparse,faulthandler,gc,importlib,random,sys,traceback
faulthandler.enable(all_threads=True)

def verify_scoring_inputs(submodule):
    prior=ROOT.parent/'phaseF_reader_credit_20261023_143148/source_manifest/input_freeze_end.json'
    expected={str(Path(x['path']).resolve()):x for x in read(prior)['files']}
    candidates=[]
    def visit(x):
        if isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif isinstance(x,str) and Path(x).is_file():candidates.append(Path(x))
    visit(submodule);verified=[]
    for p in dict.fromkeys(candidates):
        st=p.stat();key=str(p.resolve());actual=sha(p)
        if key in expected:
            item=expected[key];assert st.st_size==item['bytes'] and actual==item['sha256'],'METRIC_ASSET_IDENTITY_CHANGED '+str(p)
        else:assert st.st_size<1*1024**2,'UNREGISTERED_LARGE_METRIC_ASSET '+str(p)
        verified.append(dict(path=str(p),resolved=key,SHA256=actual,bytes=st.st_size,matched_prior_audit=key in expected))
    return verified

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['released_DMD','ODE'],required=True);ap.add_argument('--seed',choices=[2026,2027],type=int,required=True);ap.add_argument('--dimensions',default='all');a=ap.parse_args()
    approved_stage_A();check_source();lease=acquire_gpu();pre=resources(True)
    suffix=a.dimensions.replace(',','_');runid='reference_metrics_'+a.arm+'_'+str(a.seed)+'_'+suffix;ok=False
    dump(ROOT/'state'/(runid+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category='stage_A_reference_GPUh',resource=pre))
    try:
        import torch,numpy as np,vbench
        from vbench.utils import init_submodules
        from vbench.distributed import dist_init
        from importlib import metadata
        c=read(ROOT/'contracts/evaluation_manifest.json')
        requested=c['metrics'] if a.dimensions=='all' else a.dimensions.split(',')
        assert requested and set(requested)<=set(c['metrics'])
        expected=c['metric_environment'];assert sys.executable==expected['executable'] and str(torch.__version__)=='2.5.1+cu124' and np.__version__=='1.26.3'
        for name,version in expected['packages'].items():assert metadata.version(name)==version
        for path,digest in c['metric_module_SHA256'].items():assert sha(path)==digest
        n=c['metric_normalization'];assert sha(n['path'])==n['sha256']
        original=read(c['reuse_contract_source']);assert sha(c['reuse_contract_source'])==c['reuse_contract_SHA256']
        source=original['prompt_source'];assert sha(source['path'])==source['sha256']
        info=read(source['path'])
        purpose='ODE_original_panel' if a.arm=='ODE' else 'reference_health_original'
        selected=[r for r in matrix() if r['purpose']==purpose and int(r['seed'])==a.seed and set(requested).intersection(json.loads(r['official_memberships']))]
        records={}
        for row in selected:
            p=ROOT/'reference/records'/(row['video_id']+'.json');assert p.exists(),'GENERATION_NOT_COMPLETE'
            r=read(p);assert r['matrix_row']==row and r['finite'] and r['video']['lossless']
            assert sha(r['video']['path'])==r['video']['SHA256']
            records[row['prompt']]={**r,'video':{**r['video'],'sha256':r['video']['SHA256']}}
        entries=[{**v,'video_list':[records[v['prompt_en']]['video']['path']]} for v in info if v['prompt_en'] in records]
        folder=ROOT/'reference/scoring'/a.arm/str(a.seed);folder.mkdir(parents=True,exist_ok=True)
        metadata_path=folder/('eligible_full_info_'+suffix+'.json');dump(metadata_path,entries)
        # Reuse the validated filename helper and no-download guard unchanged.
        old_scripts=ROOT.parent/'goalB_20261014_095243/scripts';sys.path.insert(0,str(old_scripts))
        from b_common import block_downloads
        import metric_filename_contract as action
        assert sha(c['human_action']['helper'])==c['human_action']['helper_SHA256']
        block_downloads();resources(True);torch.cuda.set_device(0);dist_init();torch.set_grad_enabled(False)
        dims=[d for d in requested if any(d in e['dimension'] for e in entries)]
        for dim in dims:
            completed=folder/(dim+'_eval_results.json');prov=folder/(dim+'_provenance.json')
            if completed.exists():
                assert prov.exists()
                previous=read(prov)
                assert previous['result_SHA256']==sha(completed)
                assert previous['checkpoint_SHA256']==next(iter(records.values()))['checkpoint']['SHA256']
                assert previous['eligible']==len({e['video_list'][0] for e in entries if dim in e['dimension']})
                continue
            budget_guard('stage_A_reference_GPUh',8);resources(False)
            path=metadata_path;metric_entries=entries
            if dim=='human_action':path,alias_manifest,metric_entries=action.prepare_action_aliases(folder/'human_action_aliases',entries,records)
            random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
            sub=init_submodules([dim],local=True,read_frame=False)[dim];verified_assets=verify_scoring_inputs(sub);module=importlib.import_module('vbench.'+dim)
            result=getattr(module,'compute_'+dim)(str(path),torch.device('cuda:0'),sub,imaging_quality_preprocessing_mode='longer');torch.cuda.synchronize()
            expected_paths={e['video_list'][0] for e in metric_entries if dim in e['dimension']};seen=[v['video_path'] for v in result[1]]
            assert len(seen)==len(set(seen)) and set(seen)<=expected_paths
            if dim not in ['object_class','multiple_objects']:assert set(seen)==expected_paths
            assert np.isfinite(result[0]) and all(np.isfinite(float(v['video_results'])) for v in result[1])
            out=folder/(dim+'_eval_results.json');dump(out,{dim:result})
            dump(folder/(dim+'_provenance.json'),dict(arm=a.arm,seed=a.seed,metric=dim,raw_score_times100_for_points=True,
                result_SHA256=sha(out),metadata_SHA256=sha(path),source_SHA256=sha(module.__file__),eligible=len(expected_paths),retained=len(seen),
                missing=sorted(expected_paths-set(seen)),seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                verified_metric_assets=verified_assets,
                role='capability reference, NOT matched-budget competitor; no subset Total',resource=pre,checkpoint_SHA256=next(iter(records.values()))['checkpoint']['SHA256']))
            print('METRIC_PASS',a.arm,a.seed,dim,result[0],len(seen),flush=True);gc.collect();torch.cuda.empty_cache()
        dump(folder/('complete_'+suffix+'.json'),dict(status='PASS_EXISTING_OFFICIAL_MEMBER_SCORING',dimensions=dims,prompt_count=len(records),official_Total_computed=False));ok=True
    finally:
        if 'torch' in sys.modules:
            import torch.distributed as dist
            if dist.is_initialized():dist.destroy_process_group()
        dump(ROOT/'state'/(runid+'_finished.json'),dict(pid=os.getpid(),ended_at_epoch=time.time(),success=ok));lease.close()

if __name__=='__main__':
    try:main()
    except BaseException:append(ROOT/'logs/reference_scoring_failures.jsonl',dict(at=time.time(),traceback=traceback.format_exc()));raise
