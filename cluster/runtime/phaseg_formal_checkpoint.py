"""Complete native FSDP state. Single file, no duplicate raw/EMA export.

Uses the same full-state and named optimizer mapping qualified in G1.
Only completed outer/accumulation-zero boundaries may be committed.
"""
from g_runtime import *
from g_common import *
from phaseg_formal_control import validate_boundary


def save(model,gopt,copt,ema,counter,cursor,uncond,cfg,arm,folder,emergency=False):
    from omegaconf import OmegaConf
    rank=dist.get_rank();world=dist.get_world_size()
    validate_boundary(counter,cursor.cursor,world)
    accepted=rng_state();rngs=[None]*world;samplers=[None]*world;unconds=[None]*world
    dist.all_gather_object(rngs,accepted);dist.all_gather_object(samplers,cursor.state_dict());dist.all_gather_object(unconds,uncond)
    if not emergency:assert counter.outer in [200,400,600]
    # Every rank checks disk before collective full-state export. Keep all old
    # snapshots; a partial file never counts as a committed recovery point.
    storage_guard(38*1024**3)
    generator,gosd=full_state(model.generator,gopt);critic,cosd=full_state(model.fake_score,copt)
    shadow=None if ema is None else {canonical(n):p for n,p in ema.shadow.items()}
    if shadow is not None:
        assert all(v.dtype==torch.float32 for v in shadow.values())
        if rank==0:assert set(shadow)==set(generator)
    state=dict(format='SSF_PHASEG_FULL_V1',qualification_only=False,arm=arm,generator=generator,critic=critic,
        generator_optimizer=gosd,critic_optimizer=cosd,generator_ema=shadow,
        EMA_native_name_map=None if ema is None else {canonical(n):n for n in ema.shadow},
        counters=counter.state_dict(),accumulation_position=0,next_phase='G_then_C' if counter.generator_due() else 'C',
        all_rank_RNG=rngs,all_rank_sampler_state=samplers,all_rank_unconditional_conditioning=unconds,
        accepted_data_cursor=cursor.cursor*world,config=OmegaConf.to_container(cfg,resolve=True),
        scheduler_fingerprint=tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas}),
        source_manifest_SHA256=sha(ROOT/'source_manifest/source_files.json'),
        formal_code_manifest_SHA256=sha(ROOT/'source_manifest/formal_code_manifest.json'),
        contract_bundle_SHA256=sha(SOURCE/'cluster/original_run/contracts/parent_version_manifest.json'),
        accumulation_sync_policy='sync_every_micro',EMA_decay=.99)
    name=('emergency_' if emergency else 'outer_')+f'{counter.outer:06d}'
    path=folder/(name+'.pt');meta=folder/(name+'.COMMITTED.json')
    if rank==0:
        assert not path.exists() and not meta.exists(),'PRESERVE_EXISTING_CHECKPOINT'
        with path.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        digest=sha(path);checked=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert tree_hash(checked)==tree_hash(state),'FULL_READBACK_FAILED'
        marker=dict(status='COMMITTED',path=str(path),SHA256=digest,bytes=path.stat().st_size,
            counts=counter.state_dict(),cursor=cursor.cursor*world,arm=arm,accumulation_position=0,
            generator_raw_tensor_SHA256=tree_hash(generator),generator_EMA_tensor_SHA256=None if shadow is None else tree_hash(shadow),
            formal_code_manifest_SHA256=state['formal_code_manifest_SHA256'],readback_all_components_exact=True)
        with meta.open('x') as out:
            json.dump(marker,out,indent=2);out.flush();os.fsync(out.fileno())
        directory=os.open(folder,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        del checked
    dist.barrier()
    assert tree_hash(rng_state())==tree_hash(accepted),'CHECKPOINT_ADVANCED_ACCEPTED_RNG'
    del state,generator,critic,gosd,cosd,shadow;gc.collect()
    return path,read(meta)


def restore(model,gopt,copt,counter,cursor,cfg,path,arm):
    from omegaconf import OmegaConf
    from utils.distributed import EMA_FSDP
    rank=dist.get_rank();path=Path(path);meta=read(path.with_suffix('.COMMITTED.json'))
    assert meta['status']=='COMMITTED' and meta['path']==str(path)
    if rank==0:assert sha(path)==meta['SHA256'],'RESUME_CHECKPOINT_SHA_MISMATCH'
    dist.barrier();state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert state['format']=='SSF_PHASEG_FULL_V1' and not state['qualification_only'] and state['arm']==arm
    assert len(state['all_rank_RNG'])==dist.get_world_size(),'RESUME_TOPOLOGY_MISMATCH'
    assert state['accumulation_position']==0 and state['accumulation_sync_policy']=='sync_every_micro'
    assert state['config']==OmegaConf.to_container(cfg,resolve=True),'RESUME_CONFIG_MISMATCH'
    assert state['source_manifest_SHA256']==sha(ROOT/'source_manifest/source_files.json')
    assert state['formal_code_manifest_SHA256']==sha(ROOT/'source_manifest/formal_code_manifest.json')
    for module,opt,key,okey in [(model.generator,gopt,'generator','generator_optimizer'),(model.fake_score,copt,'critic','critic_optimizer')]:
        with FSDP.state_dict_type(module,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False),FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=False)):
            loaded=module.load_state_dict(state[key],strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            local=FSDP.optim_state_dict_to_load(module,opt,state[okey]);opt.load_state_dict(local)
        actual,actual_opt=full_state(module,opt)
        if rank==0:
            assert tree_hash(actual)==tree_hash(state[key]),'RESUME_PARAMETER_MAPPING'
            assert tree_hash(actual_opt)==tree_hash(state[okey]),'RESUME_MOMENT_MAPPING'
        del actual,actual_opt,local
    for k,v in state['counters'].items():setattr(counter,k,v)
    cursor.restore(state['all_rank_sampler_state'][rank]);validate_boundary(counter,cursor.cursor,dist.get_world_size())
    assert state['accepted_data_cursor']==cursor.cursor*dist.get_world_size()
    assert state['scheduler_fingerprint']==tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas})
    ema=None
    if counter.EMA_initialized:
        ema=EMA_FSDP(model.generator,decay=.99)
        native={state['EMA_native_name_map'][k]:v for k,v in state['generator_ema'].items()}
        assert set(native)==set(ema.shadow);ema.load_state_dict(native)
        assert tree_hash({canonical(n):p for n,p in ema.shadow.items()})==tree_hash(state['generator_ema'])
    else:assert state['generator_ema'] is None
    uncond=state['all_rank_unconditional_conditioning'][rank]
    rng_restore(state['all_rank_RNG'][rank]);del state;gc.collect()
    return ema,uncond
