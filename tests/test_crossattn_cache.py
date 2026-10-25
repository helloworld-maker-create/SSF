"""CPU ownership regression using the actually installed FSDP preprocessing."""
from types import SimpleNamespace

import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp._runtime_utils import _root_cast_forward_input
from torch.distributed.utils import _to_kwargs

from utils.crossattn_cache import (
    CrossAttnCacheHandle, cached_generator_forward, unwrap_crossattn_cache,
    _assert_floating_inputs,
)


def boundary(handle):
    args, kwargs = _to_kwargs((), {"crossattn_cache": handle}, torch.device("cpu"), False)
    state = SimpleNamespace(_handle=None, _use_full_prec_in_eval=False,
                            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16))
    _, kwargs = _root_cast_forward_input(state, torch.nn.Identity(), args[0], kwargs[0])
    return kwargs["crossattn_cache"]


def test_two_boundaries_preserve_handle_list_and_layer_identity():
    original = [dict(k=torch.zeros(2), v=torch.ones(2), is_init=False)]
    handle = CrossAttnCacheHandle(original)
    received = boundary(boundary(handle))
    assert received is handle
    assert received.cache is original
    assert received.cache[0] is original[0]
    assert unwrap_crossattn_cache(received) is original


def test_initialization_and_tensor_replacement_are_visible_to_caller():
    original = [dict(is_init=False)]
    cache = unwrap_crossattn_cache(boundary(boundary(CrossAttnCacheHandle(original))))
    k, v = torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)
    cache[0].update(k=k, v=v, is_init=True)
    assert original[0]["is_init"] is True
    assert original[0]["k"] is k and original[0]["v"] is v


def test_transport_never_copies_casts_clones_or_detaches_tensors():
    leaf = torch.randn(4, requires_grad=True)
    tensor = leaf * 2
    original = [dict(k=tensor, v=leaf, is_init=False)]
    ptr, version, fn = tensor.data_ptr(), tensor._version, tensor.grad_fn
    received = boundary(boundary(CrossAttnCacheHandle(original)))
    actual = unwrap_crossattn_cache(received)[0]["k"]
    assert actual is tensor
    assert (actual.data_ptr(), actual._version, actual.grad_fn) == (ptr, version, fn)
    assert actual.dtype == torch.float32 and actual.requires_grad
    actual.sum().backward()
    assert torch.equal(leaf.grad, torch.full_like(leaf, 2))


def test_legacy_non_fsdp_raw_list_none_and_return_contract():
    original = [dict(is_init=False)]
    assert unwrap_crossattn_cache(original) is original
    assert unwrap_crossattn_cache(None) is None
    result = object()
    class Generator(torch.nn.Module):
        def forward(self, *, crossattn_cache):
            cache = unwrap_crossattn_cache(crossattn_cache)
            assert cache is original
            cache[0]["is_init"] = True
            return result
    assert cached_generator_forward(Generator(), crossattn_cache=original) is result
    assert original[0]["is_init"] is True


def test_adapter_preserves_preexisting_handle():
    handle = CrossAttnCacheHandle([])
    class Generator(torch.nn.Module):
        def forward(self, *, crossattn_cache):
            assert crossattn_cache is handle
            return crossattn_cache
    assert cached_generator_forward(Generator(), crossattn_cache=handle) is handle


def test_dtype_device_guard_inspects_opaque_contents_without_casting():
    import pytest
    bad = torch.ones(2, dtype=torch.float32)
    handle = CrossAttnCacheHandle([dict(k=bad)])
    with pytest.raises(ValueError, match="must already be BF16"):
        _assert_floating_inputs(handle, torch.device("cpu"), "cache")
    assert handle.cache[0]["k"] is bad and bad.dtype == torch.float32
    good = torch.ones(2, dtype=torch.bfloat16)
    _assert_floating_inputs(CrossAttnCacheHandle([dict(k=good)]), torch.device("cpu"), "cache")


def test_actual_causal_forward_body_unwraps_only_at_model_entry():
    # Execute the actual dispatch method without allocating/importing Wan.
    from test_persistent_state import model_method
    original = [dict(is_init=False)]
    class Model:
        forward = model_method("forward")
        def _forward_inference(self, *args, **kwargs):
            assert kwargs["crossattn_cache"] is original
            return original
    assert Model().forward(kv_cache=[{}], crossattn_cache=boundary(
        boundary(CrossAttnCacheHandle(original)))) is original
