"""Effective-batch infrastructure only. DMD/rollout/attention remain native."""
import contextlib
import copy
import torch

class SharedExitPlan:
    """Native rank0 sampling/broadcast ONCE; replay same list within logical batch.

    The native pipeline still chooses exit_flags[0] for every chunk when
    same_step_across_blocks=True. This wrapper never selects another list item.
    """
    def __init__(self,pipeline_type,fixture=None):
        self.pipeline_type=pipeline_type;self.fixture=fixture;self.indices=None;self.calls=0
    def __enter__(self):
        self.original=self.pipeline_type.generate_and_sync_list
        owner=self
        def generate(instance,num_blocks,num_denoising_steps,device):
            assert instance.same_step_across_blocks and not instance.last_step_only
            if owner.indices is None:
                owner.indices=list(owner.fixture) if owner.fixture is not None else owner.original(instance,num_blocks,num_denoising_steps,device)
            assert len(owner.indices)==num_blocks and all(0<=v<num_denoising_steps for v in owner.indices)
            owner.calls+=1
            return list(owner.indices)
        self.pipeline_type.generate_and_sync_list=generate
        return self
    def __exit__(self,*exc):
        self.pipeline_type.generate_and_sync_list=self.original

def effective_backward(root,optimizer,loss_fn,micros,max_norm=10.,step=False,before_clip=None,after_clip=None):
    """64/W local means, native FSDP rank mean, exactly one ROOT clip and step.

    loss_fn(i) must run ALL forward computation within this context, including
    the real generator rollout and generator-side DMD loss. No grad freezing.
    References manually averaged elsewhere are not used as training losses.
    """
    assert micros>=1
    optimizer.zero_grad(set_to_none=True)
    losses=[];micro_grad_dtypes=[]
    for i in range(micros):
        sync=root.no_sync() if i<micros-1 else contextlib.nullcontext()
        with sync:
            loss=loss_fn(i)
            assert loss.numel()==1 and bool(torch.isfinite(loss.detach()).all()),'NONFINITE_LOSS'
            (loss/micros).backward()
            losses.append(float(loss.detach()))
        micro_grad_dtypes.append(sorted({str(p.grad.dtype) for p in root.parameters() if p.grad is not None}))
    if before_clip is not None:before_clip(root)
    # For real use this is FSDP.clip_grad_norm_, NOT torch's local-shard norm.
    norm=root.clip_grad_norm_(max_norm)
    assert bool(torch.isfinite(norm).all()),'NONFINITE_GLOBAL_GRADIENT_NORM'
    if after_clip is not None:after_clip(root)
    if step:optimizer.step()
    return dict(micro_losses=losses,local_mean_loss=sum(losses)/len(losses),preclip_global_norm=float(norm),
                backward_calls=micros,clip_calls=1,optimizer_steps=int(step),micro_grad_dtypes=micro_grad_dtypes)

class EventCounters:
    def __init__(self):
        self.outer=0;self.G=0;self.C=0;self.EMA_initialized=False;self.EMA_updates=0
    def generator_due(self):return self.outer%5==0
    def accepted_G(self):
        assert self.generator_due()
        self.G+=1
        if self.EMA_initialized:self.EMA_updates+=1
    def accepted_C_and_outer(self):
        self.C+=1;self.outer+=1
        if self.outer==200:
            assert not self.EMA_initialized;self.EMA_initialized=True
    def state_dict(self):return copy.deepcopy(vars(self))

