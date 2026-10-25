"""Bounded two-rank gradient-mean and AdamW qualification, not formal training."""
from g_common import *
import argparse,copy,faulthandler,sys,traceback
faulthandler.enable(all_threads=True)
sys.path.insert(0,str(SOURCE))
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from g_accumulation import effective_backward

def main():
    from g_runtime import initialize_distributed
    approved_stage_A();check_source();rank,world,lease,pre=initialize_distributed()
    folder=ROOT/'qualification/fsdp_toy';folder.mkdir(exist_ok=True)
    runid='qualification_toy_rank'+str(rank);ok=False
    dump(ROOT/'state'/(runid+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category='stage_A_qualification_GPUh',resource=pre))
    try:
        assert world==2
        from utils.distributed import fsdp_wrap,EMA_FSDP
        torch.manual_seed(2027)
        cpu=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.SiLU(),torch.nn.Linear(8,3))
        ref=copy.deepcopy(cpu).cuda()
        root=fsdp_wrap(cpu,sharding_strategy='full',mixed_precision=False,wrap_strategy='size',min_num_params=16)
        gen=torch.Generator().manual_seed(2027)
        x=torch.randn(64,5,generator=gen).cuda();y=torch.randn(64,3,generator=gen).cuda()
        opt=torch.optim.AdamW(root.parameters(),lr=2e-6,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
        ropt=torch.optim.AdamW(ref.parameters(),lr=2e-6,betas=(0.0,.999),eps=1e-8,weight_decay=.01)
        ema=EMA_FSDP(root,decay=.99);old_shadow={n:p.clone() for n,p in ema.shadow.items()}
        torch.nn.functional.mse_loss(ref(x),y).backward()
        target={n:p.grad.clone() for n,p in ref.named_parameters()};errors=[]
        def before(module):
            with FSDP.summon_full_params(module,recurse=True,writeback=False,with_grads=True):
                for n,p in module.module.named_parameters():
                    key=n.replace('_fsdp_wrapped_module.','')
                    errors.append(float((p.grad-target[key]).abs().max()))
                    torch.testing.assert_close(p.grad,target[key],rtol=1e-5,atol=1e-7)
        out=effective_backward(root,opt,lambda micro:torch.nn.functional.mse_loss(root(x[micro*world+rank:micro*world+rank+1]),y[micro*world+rank:micro*world+rank+1]),64//world,max_norm=.1,step=True,before_clip=before)
        expected_norm=torch.nn.utils.clip_grad_norm_(ref.parameters(),.1);ropt.step()
        assert abs(out['preclip_global_norm']-float(expected_norm))<1e-5
        with FSDP.summon_full_params(root,recurse=True,writeback=False):
            for n,p in root.module.named_parameters():
                torch.testing.assert_close(p,dict(ref.named_parameters())[n.replace('_fsdp_wrapped_module.','')],rtol=1e-5,atol=1e-7)
        ema.update(root);assert all(p.dtype==torch.float32 for p in ema.shadow.values())
        with FSDP.summon_full_params(root,recurse=True,writeback=False):
            for n,p in root.module.named_parameters():
                expected=.99*old_shadow[n]+.01*p.detach().cpu().float()
                torch.testing.assert_close(ema.shadow[n],expected,rtol=1e-5,atol=1e-7)
        result=dict(status='PASS_FSDP64_MEAN_FINAL_SYNC_ROOT_CLIP_ADAMW',rank=rank,world_size=world,resource=pre,max_abs_gradient_error=max(errors),native_reference_norm=float(expected_norm),actual=out,EMA_FP32=True,parameters_fixed_during_accumulation=True)
        dump(folder/('rank'+str(rank)+'.json'),result);dist.barrier()
        if rank==0:dump(folder/'complete.json',dict(status='PASS_FSDP64_MEAN_FINAL_SYNC_ROOT_CLIP_ADAMW',ranks=world))
        print(json.dumps(result),flush=True)
        ok=True
    finally:
        if dist.is_initialized():dist.destroy_process_group()
        dump(ROOT/'state'/(runid+'_finished.json'),dict(pid=os.getpid(),ended_at_epoch=time.time(),success=ok))
        lease.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/qualification_failures.jsonl',dict(job='fsdp_toy',rank=os.environ.get('RANK'),traceback=traceback.format_exc(),at=time.time()));raise
