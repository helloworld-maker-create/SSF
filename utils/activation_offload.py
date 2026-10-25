"""Opt-in saved-tensor transport; no forward recomputation or numerical casts."""
from contextlib import nullcontext

import torch


def cached_attention_save_context(enabled, query):
    """Offload tensors saved inside the caller's cached attention/block scope.

    Mutable KV reads must still be snapshotted before attention. The public
    PyTorch hook stores those values on CPU and restores their original device
    when backward needs them; it does not truncate their autograd edges.
    No-grad/inference calls retain the original execution and storage path.
    """
    if enabled and torch.is_grad_enabled():
        return torch.autograd.graph.save_on_cpu(pin_memory=query.device.type == "cuda")
    return nullcontext()
