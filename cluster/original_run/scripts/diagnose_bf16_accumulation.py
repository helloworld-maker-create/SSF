"""Bounded diagnostic of the failed G1 gradient mean; no model/optimizer update.

The synchronized-every-micro control is DIAGNOSTIC ONLY. It is not applied
to the frozen formal training contract or the real DMD qualification code.
"""
from g_common import *
import contextlib, copy, faulthandler, gc, sys, traceback
faulthandler.enable(all_threads=True)
sys.path.insert(0,str(SOURCE))
import torch
import torch.distributed as dist
from g_runtime import initialize_distributed


def main():
    approved_stage_A();check_source()
    rank,world,lease,pre=initialize_distributed()
    folder=ROOT/'qualification/bf16_accumulation_diagnostic'
    folder.mkdir(exist_ok=True)
    runid='qualification_bf16_diagnostic_rank'+str(rank)
    dump(ROOT/'state'/(runid+'_started.json'),dict(pid=os.getpid(),started_at_epoch=time.time(),GPU_count=1,budget_category='stage_A_qualification_GPUh',resource=pre))
    ok=False
    try:
        assert world==2
        from utils.distributed import fsdp_wrap
        torch.set_num_threads(2)
        torch.manual_seed(2027)
        base=torch.nn.Sequential(torch.nn.Linear(128,128,bias=False),torch.nn.SiLU(),torch.nn.Linear(128,32,bias=False))
        gen=torch.Generator().manual_seed(2027)
        x=torch.randn(64,128,generator=gen).to(device='cuda',dtype=torch.bfloat16)
        y=torch.randn(64,32,generator=gen).to(device='cuda',dtype=torch.float32)
        micros=64//world
        results={};reference=None;reference_outputs=None;reference_losses=None
        for arm in ['independent_reference','no_sync_first31','sync_every_micro_diagnostic']:
            root=fsdp_wrap(copy.deepcopy(base),sharding_strategy='full',mixed_precision=True,wrap_strategy='size',min_num_params=16)
            root.eval();root.zero_grad(set_to_none=True)
            before={n:p.detach().clone() for n,p in root.named_parameters()}
            streamed={};dtypes=[];outputs=[];losses=[]
            torch.cuda.synchronize();start=time.time()
            for micro in range(micros):
                budget_guard('stage_A_qualification_GPUh',48)
                independent=arm=='independent_reference'
                if independent:root.zero_grad(set_to_none=True)
                ctx=root.no_sync() if arm=='no_sync_first31' and micro<micros-1 else contextlib.nullcontext()
                with ctx:
                    idx=micro*world+rank
                    output=root(x[idx:idx+1])
                    loss=torch.nn.functional.mse_loss(output.float(),y[idx:idx+1])
                    (loss if independent else loss/micros).backward()
                torch.cuda.synchronize()
                dtypes.append(sorted({str(p.grad.dtype) for p in root.parameters() if p.grad is not None}))
                outputs.append(output.detach().clone());losses.append(loss.detach().clone())
                if independent:
                    for name,p in root.named_parameters():
                        if p.grad is None:continue
                        if name not in streamed:streamed[name]=torch.zeros_like(p.grad,dtype=torch.float32)
                        streamed[name].add_(p.grad.float(),alpha=1/micros)
            if arm=='independent_reference':
                reference=streamed;reference_outputs=outputs;reference_losses=losses
            sums=torch.zeros(3,device='cuda',dtype=torch.float64);max_abs=0.
            exact_forward=True
            for a,b in zip(outputs,reference_outputs):exact_forward &= bool(torch.equal(a,b))
            for a,b in zip(losses,reference_losses):exact_forward &= bool(torch.equal(a,b))
            for name,p in root.named_parameters():
                assert torch.equal(p.detach(),before[name]),'DIAGNOSTIC_CHANGED_PARAMETER'
                actual=streamed.get(name) if arm=='independent_reference' else p.grad
                ref=reference.get(name)
                assert (actual is None)==(ref is None)
                if actual is None:continue
                assert torch.isfinite(actual).all()
                diff=actual.float()-ref
                sums[0]+=diff.double().square().sum();sums[1]+=ref.double().square().sum();sums[2]+=actual.double().square().sum()
                if diff.numel():max_abs=max(max_abs,float(diff.abs().max()))
            dist.all_reduce(sums)
            results[arm]=dict(gradient_relative_L2=float(sums[0].sqrt()/sums[1].sqrt().clamp_min(1e-30)),reference_global_norm=float(sums[1].sqrt()),actual_global_norm=float(sums[2].sqrt()),max_abs_local_error=max_abs,micro_gradient_dtypes=dtypes,forward_and_loss_exact_equal=exact_forward,seconds=time.time()-start)
            assert exact_forward
            del root,before,streamed,outputs,losses;gc.collect();torch.cuda.empty_cache()
        result=dict(status='DIAGNOSTIC_COMPLETE_NO_FORMAL_POLICY_CHANGE',rank=rank,world=world,results=results,optimizer_steps=0,model_parameters_changed=False,full_Wan_allocations=0,production_source_modified=False)
        dump(folder/('rank'+str(rank)+'.json'),result)
        dist.barrier()
        if rank==0:dump(folder/'complete.json',dict(status='DIAGNOSTIC_COMPLETE_NO_FORMAL_POLICY_CHANGE',ranks=world))
        print(json.dumps(result),flush=True);ok=True
    finally:
        if dist.is_initialized():dist.destroy_process_group()
        dump(ROOT/'state'/(runid+'_finished.json'),dict(pid=os.getpid(),ended_at_epoch=time.time(),success=ok))
        lease.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/qualification_failures.jsonl',dict(job='bf16_accumulation_diagnostic',rank=os.environ.get('RANK'),traceback=traceback.format_exc(),at=time.time()));raise
