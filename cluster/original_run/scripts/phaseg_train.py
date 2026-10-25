"""Authorized Phase G trajectory. Native DMD; qualified B64 accumulation.

One baseline or SSF job, exactly600 completed outer iterations. Monitoring
uses the same loaded generator at200/400/600 and restores raw/RNG afterwards.
No teacher, State, KV, scheduler, LR, or loss algorithm is changed here.
"""
from g_common import *
import argparse,faulthandler,signal,traceback
faulthandler.enable(all_threads=True)
from g_runtime import *
from g_accumulation import SharedExitPlan,EventCounters
from g_accumulation_sync_candidate import effective_backward
from phaseg_formal_control import authorized,validate_boundary,DataCursor,Accounting
import phaseg_formal_checkpoint as checkpoints


def group_norms(root):
    values=torch.zeros(3,device='cuda',dtype=torch.float64)
    for n,p in root.named_parameters():
        if p.grad is None:continue
        assert torch.isfinite(p.grad).all(),'NONFINITE_GRADIENT '+n
        k=0 if 'state_reader.' in n else 1 if 'state_writer.' in n else 2
        values[k]+=p.grad.detach().double().square().sum()
    dist.all_reduce(values)
    return dict(zip(['Reader','Writer','backbone'],[float(v.sqrt()) for v in values]))


def run_update(model,opt,samples,uncond,cfg,kind,counter,accounting,logs):
    from pipeline import SelfForcingTrainingPipeline
    rank=dist.get_rank();root=model.generator if kind=='G' else model.fake_score
    loss_method=model.generator_loss if kind=='G' else model.critic_loss
    groups={};accepted_before=tree_hash(rng_state());torch.cuda.reset_peak_memory_stats();start=time.time()
    def loss_at(i):
        accounting.guard('training');model.inference_pipeline=None;gc.collect()
        conditional={k:v.to(device='cuda',dtype=torch.bfloat16) for k,v in samples[i]['conditional'].items()}
        unconditional={k:v.to(device='cuda',dtype=torch.bfloat16) for k,v in uncond.items()}
        with ForwardTrace(model) as trace,DrawTrace() as draws:
            loss,aux=loss_method(image_or_video_shape=list(cfg.image_or_video_shape),conditional_dict=conditional,unconditional_dict=unconditional,clean_latent=None,initial_latent=None)
        assert trace.writes==(7 if cfg.persistent_state.enabled else 0)
        assert trace.projectK==trace.projectV==30
        if cfg.persistent_state.enabled:
            assert trace.initial_state_zero and all(x==(kind=='G') for x in trace.writer_output_grad_flags)
        pipeline=model.inference_pipeline
        assert pipeline.crossattn_cache is trace.original_text_cache and all(c['is_init'] for c in pipeline.crossattn_cache)
        assert [(int(c['global_end_index']),int(c['local_end_index'])) for c in pipeline.kv_cache1]==[(32760,32760)]*30
        append(logs/'micro_forward.jsonl',dict(rank=rank,outer=counter.outer,kind=kind,micro=i,dataset_index=samples[i]['index'],prompt_SHA256=hashlib.sha256(samples[i]['prompt'].encode()).hexdigest(),
            loss=float(loss.detach()),loss_aux={k:float(v.float().mean()) for k,v in aux.items() if isinstance(v,torch.Tensor)},
            actual_stochastic_tensors=draws.draws,forward_output_trace_SHA256=tree_hash(trace.records),writer_calls=trace.writes,
            diagnostic_forward_record_not_an_accepted_update=True))
        return loss
    def before_clip(module):groups.update(group_norms(module))
    with SharedExitPlan(SelfForcingTrainingPipeline) as plan:
        result=effective_backward(root,opt,loss_at,len(samples),max_norm=10.,step=True,before_clip=before_clip)
        assert plan.calls==len(samples)
        exit_plan=plan.indices
    torch.cuda.synchronize()
    mean=torch.tensor(result['local_mean_loss'],device='cuda',dtype=torch.float64);dist.all_reduce(mean);mean/=dist.get_world_size()
    record=dict(rank=rank,kind=kind,outer_before=counter.outer,accepted_optimizer_update=True,loss_global_mean=float(mean),gradient_groups_preclip=groups,
        preclip_global_norm=result['preclip_global_norm'],clip_calls=result['clip_calls'],optimizer_steps=result['optimizer_steps'],local_micros=len(samples),global_samples=64,
        shared_exit_indices=exit_plan,actual_exit_choice=exit_plan[0],dataset_indices=[s['index'] for s in samples],accepted_rng_before=accepted_before,accepted_rng_after=tree_hash(rng_state()),
        micro_gradient_dtypes=result['micro_grad_dtypes'],seconds=time.time()-start,memory=memory())
    opt.zero_grad(set_to_none=True);model.inference_pipeline=None;gc.collect()
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['baseline','ssf'],required=True);parser.add_argument('--attempt',type=int,default=0);parser.add_argument('--resume');args=parser.parse_args()
    grant=authorized(args.arm);check_source()
    code=read(ROOT/'source_manifest/formal_code_manifest.json')
    for f in code['files']:assert sha(ROOT/'scripts'/f['name'])==f['SHA256'],'FORMAL_CODE_CHANGED'
    storage_guard(38*1024**3)
    folder=ROOT/'training'/args.arm;attempt=folder/f'attempt_{args.attempt:02d}';attempt.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK'])
    logs=attempt/f'rank{rank}';logs.mkdir(exist_ok=True);snapshots=folder/'checkpoints';snapshots.mkdir(exist_ok=True)
    rank,world,lease,pre=initialize_distributed()
    accounting=Accounting(args.arm,rank,args.attempt)
    stopped={'requested':False}
    def request_stop(*unused):stopped['requested']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    ok=False
    try:
        assert world==2,'QUALIFIED_TOPOLOGY_IS_WORLD2'
        accounting.switch('training')
        from utils.dataset import TextDataset
        from utils.distributed import EMA_FSDP
        from omegaconf import OmegaConf
        cfg=config(args.arm=='ssf');cfg.logdir=str(folder);cfg.phaseG_accumulation_sync_policy='sync_every_micro'
        model,gopt,copt=build_native(cfg,attempt)
        counter=EventCounters();cursor=DataCursor(TextDataset(cfg.data_path),rank,world);ema=None
        if args.resume:
            ema,uncond=checkpoints.restore(model,gopt,copt,counter,cursor,cfg,args.resume,args.arm)
            dump(logs/'resume_verified.json',dict(status='PASS_FULL_STATE_RESTORATION',checkpoint=args.resume,counts=counter.state_dict(),sampler=cursor.state_dict()))
        else:
            assert not list(snapshots.glob('*.COMMITTED.json')),'EXISTING_TRAJECTORY_REQUIRES_FULL_RESUME'
            with torch.no_grad():uncond={k:v.detach().cpu() for k,v in model.text_encoder(text_prompts=[cfg.negative_prompt]).items()}
        dump(logs/'accepted_start.json',dict(counts=counter.state_dict(),source=check_source(),sampler=cursor.state_dict(),RNG_SHA256=tree_hash(rng_state()),config=OmegaConf.to_container(cfg,resolve=True)))
        if args.resume and counter.outer in [200,400,600] and not (folder/f'outer_{counter.outer:06d}_MONITOR_COMPLETE.json').exists():
            metadata=read(Path(args.resume).with_suffix('.COMMITTED.json'))
            accounting.switch('generation')
            from phaseg_formal_monitor import generate
            generate(model,cfg,ema,args.arm,counter.outer,metadata,accounting)
            accounting.switch('training')
            if rank==0:dump(folder/f'outer_{counter.outer:06d}_MONITOR_COMPLETE.json',dict(counts=counter.state_dict(),checkpoint=metadata,raw_RNG_restored=True))
        while counter.outer<600:
            free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            if free<76*1024**3:
                # Save at the current accepted boundary while one complete
                # write plus32GiB still fits. No microbatch is accepted here.
                storage_guard(38*1024**3)
                checkpoints.save(model,gopt,copt,ema,counter,cursor,uncond,cfg,args.arm,snapshots,emergency=True)
                if rank==0:dump(attempt/'STOPPED_STORAGE_RECOVERABLE.json',dict(counts=counter.state_dict(),filesystem_available_bytes=free))
                return
            for kind in (['G','C'] if counter.generator_due() else ['C']):
                indices=cursor.indices(64//world);samples=[]
                with torch.no_grad():
                    for index in indices:
                        prompt=cursor.dataset[index]['prompts'];cond={k:v.detach().cpu() for k,v in model.text_encoder(text_prompts=[prompt]).items()}
                        assert all(torch.isfinite(v).all() for v in cond.values())
                        samples.append(dict(index=index,prompt=prompt,conditional=cond))
                record=run_update(model,gopt if kind=='G' else copt,samples,uncond,cfg,kind,counter,accounting,logs)
                cursor.cursor+=len(samples)
                if kind=='G':
                    counter.accepted_G()
                    if ema is not None:ema.update(model.generator)
                else:
                    counter.accepted_C_and_outer()
                    if counter.EMA_initialized and ema is None:ema=EMA_FSDP(model.generator,decay=.99)
                record['counts_after']=counter.state_dict();record['accepted_global_data_cursor']=cursor.cursor*world
                append(logs/'updates.jsonl',record)
                print('PHASEG_ACCEPTED_UPDATE',args.arm,rank,kind,counter.G,counter.C,counter.outer,'loss',record['loss_global_mean'],flush=True)
                del samples;gc.collect()
            validate_boundary(counter,cursor.cursor,world)
            if counter.outer in [200,400,600]:
                assert budget_used(accounting.categories['training'][0])+2<500,'TRAINING_BUDGET_SAVE_RESERVE'
                _,metadata=checkpoints.save(model,gopt,copt,ema,counter,cursor,uncond,cfg,args.arm,snapshots)
                accounting.switch('generation')
                from phaseg_formal_monitor import generate
                generate(model,cfg,ema,args.arm,counter.outer,metadata,accounting)
                accounting.switch('training')
                if rank==0:dump(folder/f'outer_{counter.outer:06d}_MONITOR_COMPLETE.json',dict(counts=counter.state_dict(),checkpoint=metadata,raw_RNG_restored=True))
            stop=torch.tensor(int(stopped['requested']),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if int(stop) and counter.outer<600:
                checkpoints.save(model,gopt,copt,ema,counter,cursor,uncond,cfg,args.arm,snapshots,emergency=True)
                if rank==0:dump(attempt/'STOPPED_RECOVERABLE.json',dict(counts=counter.state_dict(),sampler=cursor.state_dict()))
                return
        assert (counter.outer,counter.G,counter.C,counter.EMA_updates)==(600,120,600,80)
        if rank==0:dump(folder/'TRAINING_AND_GENERATION_COMPLETE.json',dict(status='PASS_600_OUTER_120G_600C_EMA80',counts=counter.state_dict(),sampler_global_cursor=cursor.cursor*world,formal_SSF_not_authorized_by_baseline_completion=True,health_review_pending=True))
        ok=True
    finally:
        cleaned=False
        try:
            if dist.is_initialized():dist.destroy_process_group()
            cleaned=True
        finally:
            accounting.close();lease.close()
            dump(logs/'worker_finished.json',dict(success=ok and cleaned,pid=os.getpid(),at=time.time(),arm=args.arm,process_group_cleanup_complete=cleaned))


if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/formal_training_failures.jsonl',dict(at=time.time(),rank=os.environ.get('RANK'),traceback=traceback.format_exc(),no_automatic_recipe_change=True));raise
