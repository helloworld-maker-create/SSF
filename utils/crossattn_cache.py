"""Identity-preserving transport for the existing caller-owned text KV cache.

FSDP rebuilds nested list/dict inputs. This plain object stays opaque through
both the Wan wrapper and causal-model FSDP boundaries; only the model forward
body unwraps it. It neither owns tensors nor changes cache or autograd semantics.
"""
import torch


class CrossAttnCacheHandle:
    __slots__ = ("cache",)

    def __init__(self, cache):
        self.cache = cache


def unwrap_crossattn_cache(cache):
    """Also accept legacy raw lists for ordinary non-FSDP model callers."""
    return cache.cache if isinstance(cache, CrossAttnCacheHandle) else cache


def _assert_floating_inputs(value, device, name):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and (value.dtype != torch.bfloat16 or value.device != device):
            raise ValueError(f"FSDP input {name} must already be BF16 on {device}; "
                             f"got {value.dtype} on {value.device}")
    elif isinstance(value, CrossAttnCacheHandle):
        _assert_floating_inputs(value.cache, device, name + ".cache")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_floating_inputs(child, device, name + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_floating_inputs(child, device, f"{name}[{index}]")


def cached_generator_forward(generator, *args, **kwargs):
    """Prepare cached generator inputs before the outer FSDP boundary.

No copying, casting, device movement, detachment or tensor operations occur.
    FSDP callers must supply local CUDA BF16 model inputs and opaque cache.
    Timesteps keep their legacy dtype: normal FSDP root casting still applies
    to them (unlike the opaque cache), including FP32 warped timesteps.
Ordinary CPU/FP32/non-FSDP calls retain their existing dtype/device contract.
"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    cache = kwargs.get("crossattn_cache")
    if cache is not None and not isinstance(cache, CrossAttnCacheHandle):
        kwargs["crossattn_cache"] = CrossAttnCacheHandle(cache)
    if isinstance(generator, FSDP):
        device = torch.device("cuda", torch.cuda.current_device())
        _assert_floating_inputs(args, device, "args")
        _assert_floating_inputs(
            {key: value for key, value in kwargs.items() if key not in ("timestep", "aug_t")},
            device, "kwargs")
    return generator(*args, **kwargs)
