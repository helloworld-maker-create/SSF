"""Frozen Phase G rows using the existing FSDP generator/text and native VAE.

All ranks execute equal-length native inference calls. Raw parameters and
accepted RNG are restored exactly before any subsequent training update.
No extra Wan instance or optimizer operation is created here.
"""
from g_runtime import *
from g_common import *


def generate(model,cfg,ema,arm,outer,checkpoint,accounting):
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    from torchvision.io import write_video
    from omegaconf import OmegaConf
    import av
    rank=dist.get_rank();world=dist.get_world_size()
    stage='B' if arm=='baseline' else 'C'
    rows=[r for r in matrix(stage) if r['arm']==arm and int(r['completed_outer'])==outer and r['condition']=='normal']
    assert len(rows)==(192 if outer==600 else 32)
    accepted=rng_state();scheduler_before=tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas})
    raw={canonical(n):p.detach().cpu().clone() for n,p in model.generator.named_parameters()}
    before=tree_hash(raw);model.inference_pipeline=None;gc.collect();torch.cuda.empty_cache()
    # This is the same BF16 VAE execution used by Stage A. VAE is frozen and is
    # not part of the text-only DMD objective; training modules are not moved.
    model.vae.to(device=torch.device('cuda',torch.cuda.current_device()),dtype=torch.bfloat16).eval()
    try:
        for weight in ['raw','EMA']:
            selected=[r for r in rows if r['weight'].lower()==weight.lower()]
            assert len(selected)%world==0
            if weight=='EMA':
                assert ema is not None
                with FSDP.summon_full_params(model.generator,writeback=True),torch.no_grad():
                    for n,p in model.generator.module.named_parameters():
                        p.copy_(ema.shadow[n].to(device=p.device,dtype=p.dtype))
            pipeline=CausalInferencePipeline(cfg,device=torch.device('cuda',torch.cuda.current_device()),generator=model.generator,text_encoder=model.text_encoder,vae=model.vae)
            assert pipeline.denoising_step_list.tolist()==[1000.,937.5,833.3333129882812,625.]
            for slot in range(rank,len(selected),world):
                row=selected[slot];accounting.guard('generation');storage_guard(512*1024**2)
                target=ROOT/row['expected_output'];record=ROOT/'generation/records'/(row['video_id']+'.json')
                exists=target.exists() and record.exists()
                ready=torch.tensor(int(exists),device='cuda');dist.all_reduce(ready)
                if int(ready)==world:
                    previous=read(record)
                    assert previous['matrix_row']==row and previous['checkpoint_SHA256']==checkpoint['SHA256']
                    assert previous['video']['lossless'] and sha(target)==previous['video']['SHA256']
                    continue
                assert int(ready)==0 and not target.exists() and not record.exists(),'PARTIAL_MONITOR_BLOCK_PRESERVED_REQUIRES_REVIEW'
                start=time.time();set_seed(int(row['seed']))
                flows=[];writes=[]
                def observe(module,args,kw,out):
                    tensors=out if isinstance(out,tuple) else (out,)
                    assert all(bool(torch.isfinite(x).all()) for x in tensors),'NONFINITE_MONITOR_FORWARD'
                    if kw.get('forward_role')=='state_write':writes.append(float(out.float().norm()))
                    else:flows.append(thash(out[0]))
                hook=model.generator.register_forward_hook(observe,with_kwargs=True)
                decoder=model.vae.model.decode
                def finite_decode(*a,**kw):
                    out=decoder(*a,**kw);assert torch.isfinite(out).all(),'NONFINITE_VAE_PRECLAMP';return out
                model.vae.model.decode=finite_decode
                try:
                    with torch.no_grad():
                        noise=torch.randn([1,21,16,60,104],device='cuda',dtype=torch.bfloat16)
                        with DrawTrace() as draws:
                            pixels,latents=pipeline.inference(noise=noise,text_prompts=[row['prompt']],return_latents=True)
                finally:
                    hook.remove();model.vae.model.decode=decoder
                with torch.no_grad():
                    assert torch.isfinite(pixels).all() and torch.isfinite(latents).all()
                    assert list(pixels.shape)==[1,81,3,480,832] and list(latents.shape)==[1,21,16,60,104]
                    assert len(draws.draws)==21
                    assert len(flows)==35 and len(writes)==(7 if arm=='ssf' else 0)
                    assert [(int(c['global_end_index']),int(c['local_end_index'])) for c in pipeline.kv_cache1]==[(32760,28080)]*30
                    assert all(c['is_init'] for c in pipeline.crossattn_cache)
                    rgb=(255*pixels[0].permute(0,2,3,1)).to(torch.uint8).cpu().contiguous()
                target.parent.mkdir(parents=True,exist_ok=True)
                expected=thash(rgb);write_video(str(target),rgb,fps=16,video_codec='libx264rgb',options={'crf':'0','threads':'1'})
                h=hashlib.sha256();frames=0
                with av.open(str(target)) as container:
                    stream=container.streams.video[0]
                    assert stream.width==832 and stream.height==480 and float(stream.average_rate)==16
                    for frame in container.decode(video=0):h.update(frame.to_ndarray(format='rgb24').tobytes());frames+=1
                assert frames==81 and h.hexdigest()==expected,'LOSSLESS_ROUNDTRIP'
                dump(record,dict(matrix_row=row,arm=arm,weight=weight,completed_outer=outer,prompt=row['prompt'],prompt_id=row['prompt_id'],seed=int(row['seed']),
                    checkpoint=checkpoint,checkpoint_SHA256=checkpoint['SHA256'],
                    weight_tensor_SHA256=checkpoint['generator_raw_tensor_SHA256'] if weight=='raw' else checkpoint['generator_EMA_tensor_SHA256'],
                    initial_noise_SHA256=thash(noise),renoise_trace=draws.draws,latent_SHA256=thash(latents),latent_finite=True,
                    raw_flow_SHA256=flows,State_writer_calls=len(writes),State_norms=writes,
                    video=dict(path=str(target),SHA256=sha(target),lossless=True,RGB_SHA256=expected,frames=81,fps=16,width=832,height=480,
                        pixel_mean=float(rgb.float().mean()),pixel_std=float(rgb.float().std())),
                    source_manifest_SHA256=sha(ROOT/'source_manifest/source_files.json'),config_SHA256=tree_hash(OmegaConf.to_container(cfg,resolve=True)),seconds=time.time()-start,
                    no_optimizer_operation=True))
                print('PHASEG_VIDEO_PASS',row['video_id'],flush=True)
                model.vae.model.clear_cache();del noise,pixels,latents,rgb;gc.collect()
            del pipeline;model.vae.model.clear_cache();dist.barrier()
    finally:
        # Restore owned local shards only after summon_full_params has exited.
        with torch.no_grad():
            for n,p in model.generator.named_parameters():p.copy_(raw[canonical(n)].to(device=p.device,dtype=p.dtype))
        assert tree_hash({canonical(n):p.detach().cpu() for n,p in model.generator.named_parameters()})==before,'MONITOR_CHANGED_RAW_WEIGHTS'
        assert tree_hash({'timesteps':model.scheduler.timesteps,'sigmas':model.scheduler.sigmas})==scheduler_before,'MONITOR_CHANGED_SCHEDULER'
        model.vae.model.clear_cache();model.vae.to('cpu');model.inference_pipeline=None
        rng_restore(accepted);del raw;gc.collect();torch.cuda.empty_cache();dist.barrier()
