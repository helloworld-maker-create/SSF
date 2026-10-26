"""New-path G1: true native DMD, B64 reference vs accumulation, full resume.

Two qualified G/C optimizer steps TOTAL, on disposable qualification models;
never a formal trajectory, never a change of loss or State/KV semantics.
"""
from g_common import *
import argparse,faulthandler,gc,sys,traceback
faulthandler.enable(all_threads=True)
from g_runtime import *
from g_accumulation import SharedExitPlan
from g_accumulation_sync_candidate import effective_backward

def gradients(root):
    return {canonical(n):p.grad for n,p in root.named_parameters()}

def reset_graph(model):
    model.inference_pipeline=None;gc.collect()

def role_group(n):
    return 'Reader' if 'state_reader.' in n else 'Writer' if 'state_writer.' in n else 'backbone'

def qualify_component(model,opt,samples,uncond,cfg,kind,folder):
    from pipeline import SelfForcingTrainingPipeline
    root=model.generator if kind=='G' else model.fake_score
    loss_method=model.generator_loss if kind=='G' else model.critic_loss
    rank=dist.get_rank();W=dist.get_world_size();A=64//W;assert len(samples)==A
    start_rng=rng_state();manual={};reference_traces=[];reference_losses=[];reference_logs=[]
    before_params={canonical(n):(p.detach().clone() if p.numel() else None) for n,p in root.named_parameters()}
    torch.cuda.reset_peak_memory_stats();started=time.time()
    def loss_at(i):
        conditional={k:v.to(device='cuda',dtype=torch.bfloat16) for k,v in samples[i]['conditional'].items()}
        unconditional={k:v.to(device='cuda',dtype=torch.bfloat16) for k,v in uncond.items()}
        with ForwardTrace(model) as trace,DrawTrace() as draws:
            loss,logs=loss_method(image_or_video_shape=list(cfg.image_or_video_shape),conditional_dict=conditional,
                unconditional_dict=unconditional,clean_latent=None,initial_latent=None)
        assert bool(torch.isfinite(loss))
        expected=7 if cfg.persistent_state.enabled else 0
        assert trace.writes==expected,'ONE_WRITER_PER_CHUNK'
        assert trace.projectK==trace.projectV==30,'TEXT_PROJECTIONS_NOT_REUSED'
        assert trace.initial_state_zero
        assert all(flag==(kind=='G') for flag in trace.writer_output_grad_flags),'WRITER_GRAD_MODE_BOUNDARY'
        pipeline=model.inference_pipeline
        assert pipeline.crossattn_cache is trace.original_text_cache,'PIPELINE_CALLER_CACHE_IDENTITY_LOST'
        assert all(x['is_init'] for x in pipeline.crossattn_cache),'CALLER_TEXT_CACHE_PERSISTENCE'
        assert all(int(x['global_end_index'])==32760 and int(x['local_end_index'])==32760 for x in pipeline.kv_cache1)
        for c in pipeline.kv_cache1:
            assert torch.isfinite(c['k']).all() and torch.isfinite(c['v']).all()
        evidence=dict(sample_index=samples[i]['index'],prompt_SHA256=hashlib.sha256(samples[i]['prompt'].encode()).hexdigest(),
            loss_SHA256=thash(loss),loss=float(loss.detach()),forwards=trace.records,writer_calls=trace.writes,actual_stochastic_tensor_trace=draws.draws,
            text_K_projections=trace.projectK,text_V_projections=trace.projectV,fresh_zero_initial_state=trace.initial_state_zero,
            caller_cache_identity_preserved=True,writer_output_grad_flags=trace.writer_output_grad_flags,
            score_timestep=None if 'timestep' not in logs else thash(logs['timestep']))
        return loss,logs,evidence
    # Reference: synchronized independent sample gradients, stream their means.
    # Parameters/optimizers stay fixed throughout ALL64 samples.
    with SharedExitPlan(SelfForcingTrainingPipeline) as plan:
        for i in range(A):
            budget_guard('stage_A_qualification_GPUh',48);opt.zero_grad(set_to_none=True);reset_graph(model)
            loss,logs,trace=loss_at(i);loss.backward();torch.cuda.synchronize()
            for name,g in gradients(root).items():
                if g is None:continue
                assert torch.isfinite(g).all(),'NONFINITE_REFERENCE_GRADIENT '+name
                if name not in manual:manual[name]=torch.zeros_like(g,device='cpu',dtype=torch.float32)
                manual[name].add_(g.detach().float().cpu(),alpha=1/A)
            reference_traces.append(trace);reference_losses.append(float(loss.detach()))
            reference_logs.append({k:float(v.float().mean()) for k,v in logs.items() if isinstance(v,torch.Tensor)})
            del loss,logs;reset_graph(model)
            print('REFERENCE_MICRO_PASS',kind,rank,i+1,A,flush=True)
        shared_indices=plan.indices
        assert plan.calls==A
    reference_rng=rng_state();reference_memory=memory();reference_seconds=time.time()-started
    assert not opt.state,'REFERENCE_ADVANCED_OPTIMIZER'
    for name,p in [(canonical(n),p) for n,p in root.named_parameters()]:
        if before_params[name] is not None:assert torch.equal(p.detach(),before_params[name]),'PARAMETERS_CHANGED_WITHIN_REFERENCE_BATCH'
    # Exact same stochastic realization; avoid any synthetic replacement loss.
    opt.zero_grad(set_to_none=True);rng_restore(start_rng);reset_graph(model)
    actual_traces=[];comparison={};clip_before={}
    def actual_loss(i):
        reset_graph(model);loss,logs,trace=loss_at(i)
        assert trace==reference_traces[i],'NATIVE_FORWARD_OR_STOCHASTIC_REPLAY_MISMATCH'
        actual_traces.append(trace);print('FORWARD_REPLAY_PASS',kind,rank,i+1,A,flush=True);return loss
    def compare(module):
        sums=torch.zeros(3,dtype=torch.float64,device='cuda');groups={k:0.0 for k in ['Reader','Writer','backbone']};presence=[]
        actual=gradients(module)
        assert set(actual)==set(before_params)
        for name,g in actual.items():
            ref=manual.get(name)
            if g is None:
                if ref is not None and ref.numel():
                    presence.append(dict(name=name,reference='tensor',accumulation='None'))
                    sq=float(ref.double().square().sum());sums[0]+=sq;sums[1]+=sq
                continue
            assert torch.isfinite(g).all(),'NONFINITE_ACCUMULATED_GRADIENT '+name
            if ref is None and g.numel():presence.append(dict(name=name,reference='None',accumulation='tensor'))
            cpu=g.detach().float().cpu();ref=torch.zeros_like(cpu) if ref is None else ref
            assert cpu.shape==ref.shape,name
            diff=cpu-ref;sums[0]+=float(diff.double().square().sum());sums[1]+=float(ref.double().square().sum());sums[2]+=float(cpu.double().square().sum())
            groups[role_group(name)]+=float(cpu.double().square().sum())
        dist.all_reduce(sums)
        all_presence=[None]*W;dist.all_gather_object(all_presence,presence)
        comparison['gradient_presence_mismatches_by_rank']=all_presence
        err=float(sums[0].sqrt());norm=float(sums[1].sqrt());relative=err/max(norm,1e-30)
        comparison.update(relative_L2_error=relative,absolute_error_norm=err,reference_global_norm=norm,actual_global_norm=float(sums[2].sqrt()),threshold_relative=.01)
        if any(all_presence) or relative>.01:
            dump(folder/(kind+'_comparison_failed_rank'+str(rank)+'.json'),dict(status='FAIL_NEW_ACCUMULATION_QUALIFICATION',comparison=comparison,optimizer_step_not_executed=True))
        assert not any(all_presence),'GRADIENT_NONE_MASK_MISMATCH; zero and None have different AdamW semantics'
        assert relative<=.01,'FAIL64_MEAN_GRADIENT_RELATIVE_ERROR '+str(relative)
        gs=torch.tensor([groups[k] for k in groups],device='cuda',dtype=torch.float64);dist.all_reduce(gs)
        comparison['preclip_global_group_norms']={k:float(v.sqrt()) for k,v in zip(groups,gs)}
        assert comparison['actual_global_norm']>0
        # Retain compact per-local-shard fingerprints for postclip comparison.
        clip_before.update({name:None if g is None else thash(g) for name,g in actual.items()})
    torch.cuda.reset_peak_memory_stats();start=time.time()
    with SharedExitPlan(SelfForcingTrainingPipeline) as plan:
        report=effective_backward(root,opt,actual_loss,A,max_norm=10.,step=True,before_clip=compare)
        assert plan.indices==shared_indices and plan.calls==A
    torch.cuda.synchronize();actual_seconds=time.time()-start
    assert tree_hash(rng_state())==tree_hash(reference_rng),'ACCEPTED_RNG_ADVANCE_MISMATCH'
    assert len(actual_traces)==A
    # Inspection of actual AdamW moments: each live qualification optimizer
    # advanced once, never once per microbatch.
    assert opt.state and all(float(s['step'])==1 for s in opt.state.values()),'OPTIMIZER_ADVANCED_PER_MICRO'
    update_sq=0.0
    for name,p in [(canonical(n),p) for n,p in root.named_parameters()]:
        before=before_params[name]
        if before is not None:
            assert p.shape==before.shape
            update_sq+=float((p.detach().double()-before.double()).square().sum())
    value=torch.tensor(update_sq,device='cuda',dtype=torch.float64);dist.all_reduce(value)
    assert float(value)>0,'NO_QUALIFICATION_PARAMETER_UPDATE'
    if kind=='G':
        assert comparison['preclip_global_group_norms']['Reader']>0
        # Writer zero at cold start is expected and not relaxed/forced open.
        assert comparison['preclip_global_group_norms']['Writer']==0.0,'COLD_ZERO_READER_BOUNDARY_CHANGED'
    record=dict(status='PASS_NATIVE_DMD_B64_GRADIENT_MEAN',kind=kind,rank=rank,world_size=W,
        global_batch=64,local_micros=A,shared_exit_indices=shared_indices,actual_exit_choice=shared_indices[0],
        genuine_DMD_source_SHA256=sha(SOURCE/'model/dmd.py'),comparison=comparison,accumulation=report,
        preclip_shard_fingerprints=clip_before,reference_loss_logs=reference_logs,reference_sample_traces=reference_traces,
        exact_forward_loss_and_rng_replay=True,qualification_optimizer_updates=1,formal_updates=0,
        parameter_update_global_L2=float(value.sqrt()),reference_seconds=reference_seconds,accumulation_seconds=actual_seconds,
        reference_memory=reference_memory,accumulation_memory=memory(),sample_indices=[x['index'] for x in samples])
    dump(folder/(kind+'_rank'+str(rank)+'.json'),record)
    del before_params,manual,reference_traces,actual_traces;opt.zero_grad(set_to_none=True);reset_graph(model);gc.collect();torch.cuda.empty_cache()
    return record

def profile_generation(model,cfg,folder):
    """Time a real native inference/VAE/lossless encode batch; no formal row."""
    from pipeline import CausalInferencePipeline
    from torchvision.io import write_video
    accepted=rng_state();rank=dist.get_rank();device=torch.device('cuda',torch.cuda.current_device())
    model.inference_pipeline=None;gc.collect();torch.cuda.empty_cache()
    model.vae.to(device=device,dtype=torch.bfloat16).eval()
    torch.cuda.synchronize();started=time.time()
    try:
        pipeline=CausalInferencePipeline(cfg,device=device,generator=model.generator,text_encoder=model.text_encoder,vae=model.vae)
        with torch.no_grad():
            noise=torch.randn([1,21,16,60,104],device=device,dtype=torch.bfloat16)
            pixels,latents=pipeline.inference(noise=noise,text_prompts=['A person walking in a park.'],return_latents=True)
            assert torch.isfinite(pixels).all() and torch.isfinite(latents).all()
            rgb=(255*pixels[0].permute(0,2,3,1)).to(torch.uint8).cpu().contiguous()
        write_video(str(folder/('qualification_video_rank'+str(rank)+'.mp4')),rgb,fps=16,video_codec='libx264rgb',options={'crf':'0','threads':'1'})
        torch.cuda.synchronize();dist.barrier()
        seconds=time.time()-started
        del noise,pixels,latents,rgb,pipeline
        return seconds
    finally:
        model.vae.model.clear_cache();model.vae.to('cpu');model.inference_pipeline=None
        rng_restore(accepted);gc.collect();torch.cuda.empty_cache()


def main():
    check_source();rank,world,lease,pre=initialize_distributed()
    base=ROOT/'qualification/real_dmd_accumulation';folder=base/'attempt_01';folder.mkdir(parents=True,exist_ok=True)
    runid='qualification_real_rank'+str(rank);ok=False
    dump(ROOT/'state'/(runid+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category='stage_A_qualification_GPUh',resource=pre))
    try:
        assert world==SITE['nodes']*SITE['gpus_per_node'] and 64%world==0
        # The disposable model includes State so cold Reader/Writer gradients
        # and full EMA coverage are tested even before a baseline trajectory.
        cfg=config(True);cfg.accumulation_sync_policy_qualification_candidate='sync_every_micro';model,gopt,copt=build_native(cfg,folder)
        A=64//world;prep_start=time.time();samples,uncond,sampler_state=prepare_training_samples(model,cfg,2*A)
        torch.cuda.synchronize();conditioning_seconds=time.time()-prep_start
        assert len(samples)==2*A,'INSUFFICIENT_QUALIFICATION_PROMPTS'
        dump(folder/('samples_rank'+str(rank)+'.json'),dict(indices=[x['index'] for x in samples],sampler_seed=0,training_seed=2027,global_batch=64,accepted_formal_data_cursor=0))
        G=qualify_component(model,gopt,samples[:A],uncond,cfg,'G',folder)
        C=qualify_component(model,copt,samples[A:],uncond,cfg,'C',folder)
        accepted=rng_state();checkpoint_start=time.time();torch.cuda.reset_peak_memory_stats()
        snapshot=save_qualification_state(model,gopt,copt,folder,accepted,samples,uncond,sampler_state,cfg)
        torch.cuda.synchronize();save_seconds=time.time()-checkpoint_start;restore_start=time.time()
        restore_qualification_state(model,gopt,copt,snapshot,folder)
        torch.cuda.synchronize();restore_seconds=time.time()-restore_start
        generation_seconds=profile_generation(model,cfg,folder)
        local=dict(G=G['accumulation_seconds'],C=C['accumulation_seconds'],save=save_seconds,restore=restore_seconds,conditioning=conditioning_seconds,generation=generation_seconds,
                   load=read(folder/('model_load_rank'+str(rank)+'.json'))['seconds'],
                   EMA=read(folder/('EMA_profile_rank'+str(rank)+'.json')))
        measured=[None]*world;dist.all_gather_object(measured,local)
        dist.barrier()
        if rank==0:
            import math
            maximum=lambda key:max(r[key] for r in measured)
            video_batches=4*math.ceil(16/world)+2*math.ceil(96/world)
            seconds=(120*maximum('G')+600*maximum('C')+3*maximum('save')+maximum('load')+
                     80*max(r['EMA']['one_function_refresh_seconds'] for r in measured)+
                     max(r['EMA']['init_seconds'] for r in measured)+maximum('conditioning')*360+
                     maximum('generation')*video_batches)
            record=dict(status='PASS_CLUSTER_NATIVE_DMD_B64_EMA_RESUME',world_size=world,
                run_fingerprint=os.environ['SSF_RUN_FINGERPRINT'],accumulation_sync_policy='sync_every_micro',
                projected_training_hours=1.2*seconds/3600,projection_margin=1.2,
                projection_includes_native_video_generation=True,external_scoring_excluded=True,
                per_rank_timings=measured,formal_training_not_started=True,
                qualification_snapshot=str(snapshot),qualification_snapshot_bytes=snapshot.stat().st_size,
                new_topology_uses_seed2027_plus_global_rank=True,old_world2_RNG_not_reused=True)
            dump(folder/'complete.json',record);dump(base/'complete.json',record);print(json.dumps(record),flush=True)
        ok=True
    finally:
        if dist.is_initialized():dist.destroy_process_group()
        dump(ROOT/'state'/(runid+'_finished.json'),dict(ended_at_epoch=time.time(),success=ok,pid=os.getpid()))
        lease.close()


if __name__=='__main__':
    main()
