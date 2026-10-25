"""G0 historical capability reference: small audited manifests/scores only.

No weights or old video bytes are opened. No new inference or scoring. The
42 exact prompt overlap is metadata-selected, not selected by prior outcomes.
"""
from g_common import *
import collections,math,statistics

def main():
    CD=ROOT.parent/'phaseCD_75_20261016_204659'
    source=ROOT.parent/'phaseE_mechanism_20261022_135620/checkpoint_manifest/checkpoints.json'
    assert sha(source)=='bf61a1a8a1b0eafe4df971856e9c0d3e3b61a6e4d7dd376759b51f1884624c79'
    registry={(x['replicate'],x['arm'],x['horizon']):x for x in read(source)['checkpoints']}
    contract=read(ROOT/'contracts/evaluation_manifest.json');panel=contract['panel']['prompts']
    overlap={p['prompt_id']:p for p in panel if p['reused_CD']};assert len(overlap)==42
    cohort={dim:{pid for pid,p in overlap.items() if dim in p['dims']} for dim in contract['metrics']}
    cohort['frame_difference']=set(overlap)
    groups=[(rep,arm,h) for rep in ['R1','R2'] for arm in ['baseline','normal'] for h in [20,50,75]]
    values={};sources=[];video_refs=[];paired_noise_checks=0;new_pending=[]
    new_ODE={(x['prompt_id'],int(x['seed'])):x for x in matrix() if x['purpose']=='ODE_original_panel'}
    with (ROOT/'contracts/matched_mvp_matrix.csv').open() as f:pass
    for rep,arm,h in groups:
        ck=registry[rep,'baseline' if arm=='baseline' else 'ssf',h]
        for seed in [2026,2027]:
            evalfolder=CD/'panel'/str(h)/'evaluation'/rep/arm/str(seed)
            genfolder=CD/'panel'/str(h)/'generation'/rep/arm/str(seed)
            for dim in contract['metrics']:
                folder=evalfolder/('human_action_filename_contract_v1' if dim=='human_action' else '')
                path=folder/(dim+'_eval_results.json');prov=read(folder/(dim+'_provenance.json'))
                assert prov['checkpoint_SHA256']==ck['expected_SHA256'] and prov['result_SHA256']==sha(path)
                metric_path=next(p for p in contract['metric_module_SHA256'] if Path(p).name==dim+'.py')
                assert prov['source_SHA256']==contract['metric_module_SHA256'][metric_path],'SCORER_VERSION_DIFF '+str(path)
                aliases={}
                if dim=='human_action':aliases={x['alias']:x['prompt_id'] for x in read(folder/'alias_manifest.json')}
                result=read(path)[dim];mapped={}
                for row in result[1]:
                    pid=aliases[row['video_path']] if dim=='human_action' else Path(row['video_path']).stem
                    if pid in cohort[dim]:
                        score=float(row['video_results'])*(.01 if dim=='imaging_quality' else 1.0);assert math.isfinite(score)
                        assert pid not in mapped;mapped[pid]=score
                values[rep,arm,h,seed,dim]=mapped
                sources.append(dict(replicate=rep,arm=arm,horizon=h,seed=seed,dimension=dim,
                    path=str(path),SHA256=sha(path),checkpoint_SHA256=ck['expected_SHA256'],retained_overlap=len(mapped)))
            frame_values={}
            for pid,item in overlap.items():
                path=genfolder/'records'/(pid+'.json');rec=read(path)
                assert rec['prompt']==item['prompt'] and rec['seed']==seed and rec['checkpoint_sha256']==ck['expected_SHA256']
                assert rec['finite'] and rec['video']['lossless']
                frame_values[pid]=float(rec['frame_difference'])
                video_refs.append(dict(replicate=rep,arm=arm,horizon=h,seed=seed,prompt_id=pid,
                    video_path=rec['video']['path'],video_SHA256=rec['video']['sha256'],record_path=str(path),record_SHA256=sha(path)))
                npath=ROOT/'reference/records'/(new_ODE[pid,seed]['video_id']+'.json')
                if npath.exists():
                    new=read(npath)
                    assert rec['initial_noise_SHA256']==new['initial_noise_SHA256']
                    assert [v['sha256'] for v in rec['renoise']]==new['renoise_SHA256'],'OLD_NEW_NOISE_POLICY_DIFF'
                    paired_noise_checks+=1
                else:new_pending.append(str(npath))
            values[rep,arm,h,seed,'frame_difference']=frame_values
    # Same retained cohort across BOTH training seeds, all horizons and arms.
    # Missing detections are disclosed; they are never silently filled with0.
    common={d:set(ids) for d,ids in cohort.items()}
    for (rep,arm,h,seed,d),mapped in values.items():common[d]&=set(mapped)
    levels=[]
    for rep,arm,h in groups:
        for dim,ids in common.items():
            points=[statistics.mean(values[rep,arm,h,s,dim][pid] for s in [2026,2027]) for pid in sorted(ids)]
            levels.append(dict(replicate=rep,training_seed=2027 if rep=='R1' else 2028,arm=arm,generator_updates=h,
                critic_updates={20:96,50:246,75:371}[h],dimension=dim,
                nominal_overlap_prompts=len(cohort[dim]),retained_common_prompts=len(ids),
                mean=None if not points else statistics.mean(points)*(1 if dim=='frame_difference' else 100),
                units='0..255 sampled RGB-mean grayscale MAE proxy' if dim=='frame_difference' else 'official per-dimension raw score x100',
                raw_generator=True,EMA=False,effective_training_batch=1,inference_seeds='2026,2027'))
    dest=ROOT/'reference/historical_overlap_v2_units';dest.mkdir(exist_ok=True)
    with (dest/'absolute_levels.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(levels[0]));writer.writeheader();writer.writerows(levels)
    dump(dest/'audit.json',dict(status='PASS_METADATA_ONLY_HISTORICAL_CAPABILITY_REUSE',overlap=42,
        inference_seeds=[2026,2027],training_seeds=[2027,2028],horizons=[20,50,75],arms=['baseline','normal'],
        checkpoint_manifest=str(source),checkpoint_manifest_SHA256=sha(source),
        eligible_cohorts={d:sorted(x) for d,x in cohort.items()},retained_common_cohorts={d:sorted(x) for d,x in common.items()},
        missing={d:sorted(cohort[d]-common[d]) for d in cohort},metric_sources=sources,existing_video_references=video_refs,
        old_vs_new_reference_initial_and_renoise_checks=paired_noise_checks,pending_new_ODE_record_paths=sorted(set(new_pending)),
        old_weight_or_video_bytes_read=False,new_videos=0,new_GPU_scoring=0,official_Total_computed=False,
        comparison_boundary='legacy raw/B1 references differ from PhaseG fresh B64/EMA/600outer recipe in several ways; no single-factor attribution or collapse claim'))
    print('HISTORICAL_REFERENCE_REUSE_PASS',len(levels),'levels; cohort counts',{d:len(x) for d,x in common.items()},flush=True)

if __name__=='__main__':main()
