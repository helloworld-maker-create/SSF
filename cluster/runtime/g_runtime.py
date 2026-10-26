"""Portable native DMD construction, losses and qualification diagnostics.

No optimizer loop lives here. Used by new-path qualification; formal training
requires its separate post-G1 authorization. No Phase F routing is imported.
"""
from g_common import *
import gc,random,sys
sys.path.insert(0,str(SOURCE));os.chdir(PRODUCTION)
# Bound compiler helper processes for these NEW qualification jobs only.
# Existing references are not interrupted; no kernel/precision setting changes.
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','4')
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig,FullOptimStateDictConfig

def canonical(name):return name.replace('_fsdp_wrapped_module.','').replace('_checkpoint_wrapped_module.','')
def thash(t):
    h=hashlib.sha256();flat=t.detach().reshape(-1)
    for start in range(0,flat.numel(),1<<22):h.update(flat[start:start+(1<<22)].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def tree_hash(v):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(str((list(x.shape),str(x.dtype))).encode());h.update(thash(x).encode())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(v);return h.hexdigest()
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def rng_restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda'])
    assert tree_hash(rng_state())==tree_hash(s)
def memory():
    torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free=free,total=total)

class DrawTrace:
    """Fingerprint actual native stochastic tensors; no replacements or saves."""
    def __init__(self):self.draws=[];self.original={}
    def __enter__(self):
        for name in ['randn','randn_like','randint']:
            original=getattr(torch,name);self.original[name]=original
            def wrapped(*args,_name=name,_original=original,**kwargs):
                out=_original(*args,**kwargs)
                self.draws.append(dict(function=_name,shape=list(out.shape),dtype=str(out.dtype),device=str(out.device),SHA256=thash(out)))
                return out
            setattr(torch,name,wrapped)
        return self
    def __exit__(self,*exc):
        for name,original in self.original.items():setattr(torch,name,original)

def config(state=True):
    from omegaconf import OmegaConf
    cfg=OmegaConf.merge(OmegaConf.load(SOURCE/'configs/default_config.yaml'),OmegaConf.load(SOURCE/'configs/self_forcing_dmd.yaml'))
    cfg.seed=2027;cfg.total_batch_size=64;cfg.batch_size=1;cfg.sharding_strategy='full';cfg.gradient_checkpointing=False
    cfg.generator_ckpt=SITE['assets']['ode_init']
    cfg.real_model_path=SITE['assets']['teacher_14b']
    cfg.data_path=SITE['assets']['training_prompts']
    cfg.disable_wandb=True;cfg.no_visualize=True;cfg.no_save=True;cfg.logdir=str(ROOT/'qualification/runtime_unused')
    cfg.model_kwargs=OmegaConf.create(dict(timestep_shift=5.,local_attn_size=18,sink_size=0,cached_attention_cpu_offload=True,cached_block_cpu_offload=True))
    cfg.persistent_state=OmegaConf.create(dict(enabled=state,num_slots=8,reader_block=15,writer_block=20,attention_dim=256,num_heads=4,init_seed=0))
    assert cfg.same_step_across_blocks and not cfg.last_step_only and cfg.num_training_frames==21 and cfg.num_frame_per_block==3
    return cfg

def initialize_distributed():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    uuids=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(uuids)==int(os.environ['LOCAL_WORLD_SIZE'])
    # g_common's per-device preflight intentionally accepts only one UUID.
    original=os.environ['CUDA_VISIBLE_DEVICES'];os.environ['CUDA_VISIBLE_DEVICES']=uuids[local]
    try:
        lease=acquire_gpu();pre=resources(True)
    finally:os.environ['CUDA_VISIBLE_DEVICES']=original
    try:
        assert torch.cuda.device_count()==len(uuids)
        torch.cuda.set_device(local)
        from utils.distributed import launch_distributed_job
        launch_distributed_job()
    except BaseException:
        if dist.is_initialized():dist.destroy_process_group()
        lease.close()
        raise
    return rank,world,lease,pre

def build_native(cfg,folder):
    assets=read(ROOT/'node_checks'/socket.gethostname()/'assets.json')
    assert assets['status'] in ['PASS_USED_TRAINING_AND_INFERENCE_ASSET_BYTES','PASS_ASSET_SIZE_AND_METADATA_ONLY']
    for rec in assets['records']:
        st=Path(rec['path']).stat()
        assert (st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(rec['bytes'],rec['mtime_ns'],rec['ctime_ns']),'ASSET_METADATA_CHANGED_SINCE_PREPARATION'
    from model.dmd import DMD
    from utils.misc import set_seed
    from utils.distributed import fsdp_wrap
    from utils.persistent_state import load_generator_state_dict
    from omegaconf import OmegaConf
    rank=dist.get_rank();world=dist.get_world_size();device=torch.device('cuda',torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    set_seed(cfg.seed+rank);start=time.time()
    # Only rank0 scans checkpoint bytes; all ranks load the exact same immutable file.
    expected=read(ROOT/'reference/ODE_input_checkpoint.json')
    if rank==0:assert sha(cfg.generator_ckpt)==expected['SHA256']
    dist.barrier()
    model=DMD(cfg,device=device)
    weights=torch.load(cfg.generator_ckpt,map_location='cpu',mmap=True,weights_only=False)
    payload=weights['generator'] if 'generator' in weights else weights['model'] if 'model' in weights else weights
    assert len(payload)==825
    result=load_generator_state_dict(model.generator,payload)
    expected_missing=set(getattr(model.generator,'persistent_state_parameter_names',()))
    assert set(result.missing_keys)==expected_missing and not result.unexpected_keys
    assert len(expected_missing)==(11 if cfg.persistent_state.enabled else 0)
    branch={n:thash(p) for n,p in model.generator.named_parameters() if 'state_reader.' in n or 'state_writer.' in n}
    gathered=[None]*world;dist.all_gather_object(gathered,branch);assert all(x==gathered[0] for x in gathered),'BRANCH_INIT_MISMATCH'
    del payload,weights;gc.collect()
    total=sum(p.numel() for p in model.generator.parameters());state_count=sum(p.numel() for n,p in model.generator.named_parameters() if 'state_reader.' in n or 'state_writer.' in n)
    for name,wrapkey in [('generator','generator_fsdp_wrap_strategy'),('real_score','real_score_fsdp_wrap_strategy'),('fake_score','fake_score_fsdp_wrap_strategy'),('text_encoder','text_encoder_fsdp_wrap_strategy')]:
        setattr(model,name,fsdp_wrap(getattr(model,name),sharding_strategy='full',mixed_precision=True,wrap_strategy=getattr(cfg,wrapkey),cpu_offload=False))
    # Saved-tensor activation offload, NOT FSDP parameter CPU offload.
    original_fake=model.fake_score.forward
    def fake(*args,**kw):
        if torch.is_grad_enabled():
            with torch.autograd.graph.save_on_cpu(pin_memory=True):return original_fake(*args,**kw)
        return original_fake(*args,**kw)
    model.fake_score.forward=fake
    def check_scores(module,args,out):
        for tensor in out:assert bool(torch.isfinite(tensor).all()),'NONFINITE_NATIVE_SCORE_PREDICTION'
    model.real_score.register_forward_hook(check_scores)
    model.fake_score.register_forward_hook(check_scores)
    model.eval()
    for wrapped in [model.generator,model.fake_score,model.real_score]:
        for module in wrapped.modules():
            if hasattr(module,'gradient_checkpointing'):assert not module.gradient_checkpointing
        def host_guard(*args):
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            assert available>=SITE['host_reserve_gib']*1024**3,'BLOCKED_HOST_RAM_RESERVE'
        for module in wrapped.modules():
            if module.__class__.__name__ in ['CausalWanAttentionBlock','WanAttentionBlock']:
                module.register_forward_pre_hook(host_guard)
    gopt=torch.optim.AdamW([p for p in model.generator.parameters() if p.requires_grad],lr=2e-6,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
    copt=torch.optim.AdamW([p for p in model.fake_score.parameters() if p.requires_grad],lr=4e-7,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
    dump(folder/('model_load_rank'+str(rank)+'.json'),dict(rank=rank,world_size=world,seconds=time.time()-start,source=check_source(),total_generator_parameters=total,state_parameters=state_count,ODE_matched=825,missing_keys=sorted(expected_missing),unexpected_keys=[],initial_branch_fingerprints=branch,all_rank_branch_identical=True,memory=memory(),config=OmegaConf.to_container(cfg,resolve=True),DMD_loss_source=str(SOURCE/'model/dmd.py'),DMD_loss_SHA256=sha(SOURCE/'model/dmd.py'),compiler_workers=os.environ['TORCHINDUCTOR_COMPILE_THREADS'],FSDP_units={n:sum(isinstance(x,FSDP) for x in getattr(model,n).modules()) for n in ['generator','real_score','fake_score','text_encoder']}))
    return model,gopt,copt

def prepare_training_samples(model,cfg,total_per_rank):
    from utils.dataset import TextDataset
    from torch.utils.data.distributed import DistributedSampler
    dataset=TextDataset(cfg.data_path)
    sampler=DistributedSampler(dataset,num_replicas=dist.get_world_size(),rank=dist.get_rank(),shuffle=True,seed=0,drop_last=True)
    order=list(iter(sampler));indices=order[:total_per_rank];out=[]
    with torch.no_grad():
        uncond={k:v.detach().cpu() for k,v in model.text_encoder(text_prompts=[cfg.negative_prompt]).items()}
        assert all(torch.isfinite(v).all() for v in uncond.values())
        for index in indices:
            prompt=dataset[index]['prompts']
            assert isinstance(prompt,str)
            cond={k:v.detach().cpu() for k,v in model.text_encoder(text_prompts=[prompt]).items()}
            assert all(torch.isfinite(v).all() for v in cond.values())
            out.append(dict(index=index,prompt=prompt,conditional=cond))
    sampler_state=dict(seed=0,epoch=0,rank=dist.get_rank(),world_size=dist.get_world_size(),
        accepted_rank_cursor=total_per_rank,accepted_global_cursor=total_per_rank*dist.get_world_size(),
        next_eight_indices=order[total_per_rank:total_per_rank+8],dataset_size=len(dataset),drop_last=True,shuffle=True)
    return out,uncond,sampler_state

class ForwardTrace:
    def __init__(self,model):
        self.model=model;self.records=[];self.projectK=self.projectV=0;self.writes=0;self.h=[]
        self.original_text_cache=None;self.initial_state_zero=False;self.writer_output_grad_flags=[]
    def __enter__(self):
        # Attach only to original wrapper, after FSDP preprocessing. StateWriter
        # transitions are pipeline calls; this observer never invokes one.
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.modules.causal_model import CausalWanAttentionBlock
        for m in self.model.generator.modules():
            if isinstance(m,WanDiffusionWrapper):
                self.h.append(m.register_forward_pre_hook(self.pre,with_kwargs=True))
                self.h.append(m.register_forward_hook(self.post,with_kwargs=True))
            if isinstance(m,CausalWanAttentionBlock):
                self.h.append(m.cross_attn.k.register_forward_hook(self.key))
                self.h.append(m.cross_attn.v.register_forward_hook(self.value))
        return self
    def key(self,*args):self.projectK+=1
    def value(self,*args):self.projectV+=1
    def pre(self,module,args,kw):
        if kw.get('forward_role')=='state_write':
            assert not kw['persistent_state'].requires_grad and not kw['state_write_hidden'].requires_grad
            return
        cache=kw['crossattn_cache'].cache
        if self.original_text_cache is None:
            self.original_text_cache=cache
            assert not any(c['is_init'] for c in cache),'ROLLOUT_REUSED_PREVIOUS_TEXT_CACHE'
        assert cache is self.original_text_cache,'TEXT_CACHE_OWNERSHIP_CHANGED'
        if not self.records and kw.get('persistent_state') is not None:
            state=kw['persistent_state']
            assert list(state.shape)==[1,8,1536] and not torch.count_nonzero(state),'STATE_NOT_FRESH_ZERO'
            self.initial_state_zero=True
    def post(self,module,args,kw,out):
        if kw.get('forward_role')=='state_write':
            self.writes+=1;assert torch.isfinite(out).all();self.writer_output_grad_flags.append(out.requires_grad);return
        for x in out:assert torch.isfinite(x).all()
        handle=kw['crossattn_cache'];cache=handle.cache
        assert cache is self.original_text_cache
        assert all(x['is_init'] for x in cache),'TEXT_CACHE_DID_NOT_PERSIST'
        self.records.append(dict(role=kw.get('forward_role'),start=int(kw['current_start']),flow=thash(out[0]),clean=thash(out[1])))
    def __exit__(self,*a):
        for h in self.h:h.remove()

def full_state(module,optimizer=None):
    with FSDP.state_dict_type(module,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True),FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        params=module.state_dict()
        opt=FSDP.optim_state_dict(module,optimizer) if optimizer is not None else None
    return params,opt

def save_qualification_state(model,gopt,copt,folder,accepted_rng,samples,uncond,sampler_state,cfg):
    """ONE retained complete qualification snapshot (inside64GiB allowance)."""
    from utils.distributed import EMA_FSDP
    from omegaconf import OmegaConf
    rank=dist.get_rank();rngs=[None]*dist.get_world_size();dist.all_gather_object(rngs,accepted_rng)
    samplers=[None]*dist.get_world_size();dist.all_gather_object(samplers,sampler_state)
    unconditionals=[None]*dist.get_world_size();dist.all_gather_object(unconditionals,uncond)
    generator,g_osd=full_state(model.generator,gopt);critic,c_osd=full_state(model.fake_score,copt)
    ema_start=time.time();ema=EMA_FSDP(model.generator,decay=.99)
    torch.cuda.synchronize();EMA_init_seconds=time.time()-ema_start
    export={canonical(k):v for k,v in ema.state_dict().items()}
    assert len(export)==len(ema.state_dict())==836
    assert all(v.dtype==torch.float32 for v in export.values())
    if rank==0:
        assert set(export)==set(generator),'EMA_CANONICAL_EXPORT_COVERAGE'
        assert all(torch.equal(export[n],generator[n]) for n in generator),'EMA_INIT_NOT_PARAMETER_COPY'
    # A function-level coverage/profile test, NOT a historical EMA event.
    # Restore the exact coverage copy afterwards; formal EMA event counts stay0.
    previous_shadow={k:v.clone() for k,v in ema.shadow.items()}
    refresh_start=time.time();ema.update(model.generator);torch.cuda.synchronize()
    EMA_refresh_seconds=time.time()-refresh_start
    assert set(ema.shadow)==set(previous_shadow) and all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in ema.shadow.values())
    ema.shadow=previous_shadow
    export={canonical(k):v for k,v in ema.shadow.items()}
    dump(folder/('EMA_profile_rank'+str(rank)+'.json'),dict(init_seconds=EMA_init_seconds,one_function_refresh_seconds=EMA_refresh_seconds,
        coverage=836,FP32=True,all_State_parameters_included=True,historical_training_EMA_updates=0,coverage_copy_restored=True))
    snapshot=folder/'complete_qualification.pt'
    state=dict(generator=generator,critic=critic,generator_optimizer=g_osd,critic_optimizer=c_osd,
        generator_ema=export,EMA_native_name_map={canonical(k):k for k in ema.shadow},all_rank_RNG=rngs,
        qualification_counts={'G':1,'C':1,'outer':1,'formal_updates':0,'EMA_updates':0},
        accepted_data_cursor=128,all_rank_sampler_state=samplers,next_phase='critic_at_outer1',
        unconditional_conditioning_cache=unconditionals[0],all_rank_unconditional_conditioning=unconditionals,
        qualification_sample_indices=[x['index'] for x in samples],accumulation_position=0,
        qualification_EMA_coverage_copy_not_historical=True,
        config=OmegaConf.to_container(cfg,resolve=True),scheduler_fingerprint=tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas}),
        source_manifest_SHA256=sha(ROOT/'source_manifest/source_files.json'),qualification_only=True)
    content_hash=None
    if rank==0:
        assert budget_used('stage_A_qualification_GPUh')<46,'BLOCKED_BUDGET_FULL_SNAPSHOT_RESERVE'
        used=sum(p.stat().st_size for p in (ROOT/'qualification').rglob('*') if p.is_file())
        assert used+40*1024**3<=64*1024**3,'BLOCKED_QUALIFICATION_STORAGE_CAP; preserve any prior snapshot'
        storage_guard(40*1024**3);assert not snapshot.exists()
        with snapshot.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        content_hash=sha(snapshot)
        checked=torch.load(snapshot,map_location='cpu',mmap=True,weights_only=False)
        assert tree_hash(checked)==tree_hash(state),'FULL_READBACK_MISMATCH'
        dump(folder/'qualification_COMMITTED.json',dict(status='COMMITTED',snapshot=str(snapshot),bytes=snapshot.stat().st_size,SHA256=content_hash,formal_training_checkpoint=False,EMA_parameter_coverage=len(export),all_components_readback_equal=True))
        del checked
    dist.barrier();del state,generator,critic,g_osd,c_osd,export,ema;gc.collect()
    return snapshot

def restore_qualification_state(model,gopt,copt,path,folder):
    """Perturb disposable in-memory state, then restore; no optimizer step."""
    rank=dist.get_rank();state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    expected_rng=state['all_rank_RNG'][rank]
    assert tree_hash(rng_state())==tree_hash(expected_rng),'SAVE_UNEXPECTEDLY_ADVANCED_ACCEPTED_RNG'
    # This prevents a no-op loader from passing an in-place reload test. The
    # checkpoint and every frozen input file remain strictly unchanged.
    perturbed={}
    with torch.no_grad():
        for tag,module,opt in [('generator',model.generator,gopt),('critic',model.fake_score,copt)]:
            old=tree_hash({canonical(n):p.detach() for n,p in module.named_parameters()})
            old_opt=tree_hash(opt.state_dict())
            for p in module.parameters():
                if p.numel():p.add_(.125)
            for item in opt.state.values():
                for key in ['exp_avg','exp_avg_sq']:
                    if key in item:item[key].add_(.25)
                if 'step' in item:item['step'].add_(3.)
            new=tree_hash({canonical(n):p.detach() for n,p in module.named_parameters()})
            new_opt=tree_hash(opt.state_dict())
            assert new!=old and new_opt!=old_opt,'QUALIFICATION_PERTURBATION_NOT_EFFECTIVE'
            perturbed[tag]=dict(parameters_changed=True,moments_and_step_changed=True)
    random.random();np.random.random();torch.rand(4);torch.rand(4,device='cuda')
    assert tree_hash(rng_state())!=tree_hash(expected_rng)
    for module,opt,key,okey in [(model.generator,gopt,'generator','generator_optimizer'),(model.fake_score,copt,'critic','critic_optimizer')]:
        with FSDP.state_dict_type(module,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False),FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=False)):
            loaded=module.load_state_dict(state[key],strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            local_osd=FSDP.optim_state_dict_to_load(module,opt,state[okey])
            opt.load_state_dict(local_osd)
        restored,restored_opt=full_state(module,opt)
        if rank==0:
            assert tree_hash(restored)==tree_hash(state[key]),'PARAMETER_RESTORE '+key
            assert tree_hash(restored_opt)==tree_hash(state[okey]),'FULL_OPTIMIZER_MOMENT_MAPPING_RESTORE '+key
        del restored,restored_opt,local_osd
    rng_restore(expected_rng)
    assert tree_hash(rng_state())==tree_hash(expected_rng),'ACCEPTED_RNG_RESTORE'
    assert state['accumulation_position']==0 and state['accepted_data_cursor']==128 and state['qualification_only']
    from utils.dataset import TextDataset
    from torch.utils.data.distributed import DistributedSampler
    ss=state['all_rank_sampler_state'][rank];dataset=TextDataset(state['config']['data_path'])
    sampler=DistributedSampler(dataset,num_replicas=ss['world_size'],rank=rank,shuffle=True,seed=ss['seed'],drop_last=True)
    sampler.set_epoch(ss['epoch']);order=list(iter(sampler));cursor=ss['accepted_rank_cursor']
    assert order[cursor:cursor+8]==ss['next_eight_indices'],'SAMPLER_CURSOR_RESTORE'
    assert state['next_phase']=='critic_at_outer1' and state['qualification_counts']['outer']==1
    restored_uncond=state['all_rank_unconditional_conditioning'][rank]
    assert restored_uncond and state['qualification_EMA_coverage_copy_not_historical']
    from omegaconf import OmegaConf
    assert OmegaConf.to_container(model.args,resolve=True)==state['config'],'MODEL_CONFIG_RECONSTRUCTION'
    assert tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas})==state['scheduler_fingerprint'],'SCHEDULER_RECONSTRUCTION'
    from utils.distributed import EMA_FSDP
    ema=EMA_FSDP(model.generator,decay=.99)
    native={state['EMA_native_name_map'][k]:v for k,v in state['generator_ema'].items()}
    assert set(native)==set(ema.shadow);ema.load_state_dict(native)
    restored_EMA={canonical(k):v for k,v in ema.shadow.items()}
    assert tree_hash(restored_EMA)==tree_hash(state['generator_ema']) and all(v.dtype==torch.float32 for v in ema.shadow.values())
    from g_accumulation import EventCounters
    counter=EventCounters();counter.G=state['qualification_counts']['G'];counter.C=state['qualification_counts']['C'];counter.outer=state['qualification_counts']['outer']
    assert (counter.G,counter.C,counter.outer,counter.generator_due())==(1,1,1,False)
    dump(folder/('restore_rank'+str(rank)+'.json'),dict(status='PASS_COMPLETE_FSDP_OPTIMIZER_RNG_MAPPING',rank=rank,counts=state['qualification_counts'],cursor=state['accepted_data_cursor'],no_additional_optimizer_step=True,
        prior_in_memory_perturbation=perturbed,RNG_deliberately_advanced_then_exactly_restored=True,full_named_optimizer_state_matches_saved=True,
        sampler_next8=ss['next_eight_indices'],config_and_scheduler_exact=True,EMA_FP32_all836_coverage_restored=True,
        EMA_is_qualification_coverage_copy_not_historical=True,unconditional_conditioning_cache_SHA256=tree_hash(restored_uncond)))
    assert tree_hash(rng_state())==tree_hash(expected_rng),'RECOVERY_AUDIT_ADVANCED_ACCEPTED_RNG'
    del ema,native,restored_EMA
    del state;gc.collect()
