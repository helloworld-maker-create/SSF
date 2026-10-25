"""Authorized Stage A only: real released EMA and ODE, unchanged native inference."""
from g_common import *
import argparse,contextlib,faulthandler,gc,sys,traceback
faulthandler.enable(all_threads=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['released_DMD','ODE'],required=True);ap.add_argument('--attempt',type=int,default=0);a=ap.parse_args()
    approved_stage_A();check_source();lease=acquire_gpu();pre=resources(True)
    runid='reference_'+a.arm+('' if a.attempt==0 else '_attempt'+str(a.attempt).zfill(2))
    dump(ROOT/'state'/(runid+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category='stage_A_reference_GPUh',resource=pre))
    ok=False
    try:
        sys.path.insert(0,str(SOURCE));os.chdir(PRODUCTION)
        import torch
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline
        from utils.misc import set_seed
        from utils.persistent_state import load_generator_state_dict
        from torchvision.io import write_video
        import av,inspect
        assert str(SOURCE) in inspect.getfile(CausalInferencePipeline)
        assert torch.__version__=='2.8.0+cu129'
        torch.set_grad_enabled(False)
        ck=Path('/data/tuyijing/TTS/Self-Forcing/checkpoints/self_forcing_dmd.pt' if a.arm=='released_DMD' else '/data/zhouli/Self-Forcing/checkpoints/ode_init.pt')
        started=time.time();digest=sha(ck)
        if a.arm=='ODE':assert digest=='b5396b8076ab3b920c9e4f4a2b52daa2c98c9983fb5e067ae5160fdf778dce21'
        identity=dict(path=str(ck),SHA256=digest,bytes=ck.stat().st_size,checked_at=time.time(),hash_seconds=time.time()-started)
        identity_path=ROOT/'reference'/(a.arm+'_input_checkpoint.json')
        if identity_path.exists():
            old=read(identity_path);assert old['SHA256']==identity['SHA256'] and old['bytes']==identity['bytes']
            append(ROOT/'reference/input_reverification.jsonl',dict(attempt=a.attempt,**identity))
        else:dump(identity_path,identity)
        rows=[r for r in matrix() if r['arm']==a.arm]
        completed=set()
        for row in rows:
            p=ROOT/'reference/records'/(row['video_id']+'.json')
            if p.exists():
                rec=read(p);assert rec['matrix_row']==row and rec['checkpoint']['SHA256']==digest and rec['finite']
                assert sha(rec['video']['path'])==rec['video']['SHA256'] and rec['video']['lossless']
                completed.add(row['video_id'])
        windows=sorted({int(r['local_attn_size']) for r in rows if r['video_id'] not in completed})
        for window in windows:
            budget_guard('stage_A_reference_GPUh',8);resources(True)
            model_start=time.time()
            cfg=OmegaConf.create(dict(independent_first_frame=False,warp_denoising_step=True,num_frame_per_block=3,
                context_noise=0,denoising_step_list=[1000,750,500,250],
                model_kwargs=dict(timestep_shift=5.,local_attn_size=window,sink_size=0),persistent_state=dict(enabled=False)))
            set_seed(2026);pipeline=CausalInferencePipeline(cfg,device=torch.device('cuda:0'))
            weights=torch.load(ck,map_location='cpu',mmap=True,weights_only=False)
            if a.arm=='released_DMD':assert set(weights)=={'generator_ema'};payload=weights['generator_ema']
            else:
                payload=weights['generator'] if 'generator' in weights else weights['model'] if 'model' in weights else weights
            loaded=load_generator_state_dict(pipeline.generator,payload)
            assert len(payload)==825 and not any('state_reader' in n or 'state_writer' in n for n in payload)
            del payload,weights
            resources(True);pipeline=pipeline.to(dtype=torch.bfloat16).to('cuda:0').eval()
            assert not pipeline.generator.model.gradient_checkpointing
            assert pipeline.denoising_step_list.tolist()==[1000.,937.5,833.3333129882812,625.]
            print('MODEL_LOADED',json.dumps(dict(arm=a.arm,window=window,seconds=time.time()-model_start,checkpoint=identity,load_result=str(loaded))),flush=True)
            targets=[r for r in rows if int(r['local_attn_size'])==window and r['video_id'] not in completed]
            # Fixed source order; no outcome selection or quality-based regeneration.
            for row in targets:
                budget_guard('stage_A_reference_GPUh',8);resources(False)
                video_id=row['video_id'];recpath=ROOT/'reference/records'/(video_id+'.json')
                outpath=ROOT/'reference/videos'/(video_id+'.mp4')
                assert not recpath.exists() and not outpath.exists(),'Preserve existing attempt; no overwrite'
                begin=time.time();set_seed(int(row['seed']))
                def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                noise=torch.randn([1,21,16,60,104],device='cuda:0',dtype=torch.bfloat16)
                noise_hash=thash(noise);renoise=[];flows=[];original=torch.randn_like
                def draw(x,*args,**kwargs):
                    y=original(x,*args,**kwargs);renoise.append(thash(y));return y
                def observe(module,args,kwargs,result):
                    assert not module.model.gradient_checkpointing
                    if kwargs.get('forward_role')=='state_write':raise AssertionError('State must remain disabled')
                    assert len(result)==2
                    for tensor in result:assert bool(torch.isfinite(tensor).all()),'NONFINITE_GENERATOR'
                    flows.append(thash(result[0]))
                h=pipeline.generator.register_forward_hook(observe,with_kwargs=True)
                decode=pipeline.vae.model.decode
                def finite_decode(*args,**kwargs):
                    y=decode(*args,**kwargs);assert bool(torch.isfinite(y).all()),'NONFINITE_VAE_PRECLAMP';return y
                pipeline.vae.model.decode=finite_decode
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                torch.randn_like=draw
                try:
                    pixels,latents=pipeline.inference(noise=noise,text_prompts=[row['prompt']],return_latents=True)
                    torch.cuda.synchronize()
                finally:
                    torch.randn_like=original;h.remove();pipeline.vae.model.decode=decode
                elapsed=time.time()-begin
                assert len(renoise)==21 and len(flows)==35
                assert bool(torch.isfinite(latents).all()) and bool(torch.isfinite(pixels).all())
                assert list(latents.shape)==[1,21,16,60,104]
                assert list(pixels.shape)==[1,81,3,480,832]
                ends=[(int(c['global_end_index']),int(c['local_end_index'])) for c in pipeline.kv_cache1]
                assert ends==[(32760,32760 if window==-1 else 28080)]*30
                assert all(c['is_init'] for c in pipeline.crossattn_cache)
                encoded=(255*pixels[0].permute(0,2,3,1)).to(torch.uint8).cpu().contiguous()
                expected=thash(encoded);outpath.parent.mkdir(parents=True,exist_ok=True);storage_guard(512*1024**2)
                write_video(str(outpath),encoded,fps=16,video_codec='libx264rgb',options={'crf':'0','threads':'1'})
                hh=hashlib.sha256();n=0
                with av.open(str(outpath)) as c:
                    s=c.streams.video[0];assert s.width==832 and s.height==480 and float(s.average_rate)==16
                    for frame in c.decode(video=0):hh.update(frame.to_ndarray(format='rgb24').tobytes());n+=1
                assert n==81 and hh.hexdigest()==expected,'LOSSLESS_RGB_ROUNDTRIP_FAILED'
                f,t=torch.cuda.mem_get_info();stats=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free=f,total=t)
                fp=encoded.float();mean=float(fp.mean());std=float(fp.std());del fp
                result=dict(matrix_row=row,prompt=row['prompt'],prompt_id=row['prompt_id'],seed=int(row['seed']),arm=a.arm,
                    checkpoint=identity,weight='generator_ema' if a.arm=='released_DMD' else 'generator',
                    config=OmegaConf.to_container(cfg,resolve=True),config_SHA256=hashlib.sha256(json.dumps(OmegaConf.to_container(cfg,resolve=True),sort_keys=True).encode()).hexdigest(),
                    initial_noise_SHA256=noise_hash,renoise_SHA256=renoise,raw_flow_SHA256=flows,latent_SHA256=thash(latents),
                    finite=True,KV_ends=ends,text_cache_initialized=True,wall_seconds=elapsed,memory=stats,
                    video=dict(path=str(outpath),SHA256=sha(outpath),RGB_SHA256=expected,lossless=True,frames=n,fps=16,width=832,height=480,
                        pixel_mean=mean,pixel_std=std,very_dark=mean<=1,very_bright=mean>=254,low_contrast=std<=1,
                        finite_poor_results_retained=True))
                # Same prompt/seed, both checkpoints and windows must consume identical exogenous noise.
                for other in (ROOT/'reference/records').glob('*.json'):
                    previous=read(other)
                    if previous['prompt']==row['prompt'] and previous['seed']==int(row['seed']):
                        assert noise_hash==previous['initial_noise_SHA256'] and renoise==previous['renoise_SHA256'],'PAIRED_NOISE_FAILED'
                dump(recpath,result)
                append(ROOT/'logs/reference_progress.jsonl',dict(video_id=video_id,pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],wall_seconds=elapsed,at=time.time(),memory=stats))
                print('VIDEO_PASS',video_id,'seconds',elapsed,'bytes',outpath.stat().st_size,flush=True)
                pipeline.vae.model.clear_cache();del pixels,latents,encoded,noise;gc.collect()
            del pipeline;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        assert sha(ck)==identity['SHA256'],'INPUT_CHECKPOINT_CHANGED'
        ok=True;dump(ROOT/'reference'/(a.arm+'_complete.json'),dict(status='PASS_TECHNICAL_REFERENCE',count=len(rows),checkpoint=identity,quality_health_not_yet_judged=True))
    finally:
        dump(ROOT/'state'/(runid+'_finished.json'),dict(ended_at_epoch=time.time(),success=ok,pid=os.getpid()))
        lease.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/reference_failures.jsonl',dict(at=time.time(),pid=os.getpid(),traceback=traceback.format_exc()))
        raise
