"""Qualification-only candidate: native FP32 FSDP reduction each micro.

The original no_sync implementation and frozen formal contract are untouched.
Parameters, losses, rank averaging, one final root clip/step are unchanged.
Additional communication avoids accumulating unreduced BF16 gradients.
Formal adoption requires an explicit post-qualification contract decision.
"""
import torch


def effective_backward(root,optimizer,loss_fn,micros,max_norm=10.,step=False,before_clip=None,after_clip=None):
    assert micros>=1
    optimizer.zero_grad(set_to_none=True)
    losses=[];micro_grad_dtypes=[]
    for i in range(micros):
        loss=loss_fn(i)
        assert loss.numel()==1 and bool(torch.isfinite(loss.detach()).all()),'NONFINITE_LOSS'
        (loss/micros).backward()
        losses.append(float(loss.detach()))
        dtypes=sorted({str(p.grad.dtype) for p in root.parameters() if p.grad is not None})
        assert dtypes and all(x in ['torch.float32','torch.float64'] for x in dtypes),'CANDIDATE_GRAD_ACCUMULATOR_NOT_FULL_PRECISION '+str(dtypes)
        micro_grad_dtypes.append(dtypes)
    if before_clip is not None:before_clip(root)
    norm=root.clip_grad_norm_(max_norm)
    assert bool(torch.isfinite(norm).all()),'NONFINITE_GLOBAL_GRADIENT_NORM'
    if after_clip is not None:after_clip(root)
    if step:optimizer.step()
    return dict(micro_losses=losses,local_mean_loss=sum(losses)/len(losses),preclip_global_norm=float(norm),
                backward_calls=micros,clip_calls=1,optimizer_steps=int(step),micro_grad_dtypes=micro_grad_dtypes,
                sync_policy='sync_every_micro_qualification_candidate',no_sync_calls=0,
                formal_policy_adoption='PENDING')
