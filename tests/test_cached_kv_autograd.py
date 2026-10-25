"""Real cached self-attention source, tiny CPU tensors and deterministic kernels.

The reference snapshots inside the synthetic kernel (before saving for backward).
No Wan weights/GPU/optional attention imports are required.
"""
import math

import pytest
import torch
from torch import nn

from test_persistent_state import isolated_class
from utils.activation_offload import cached_attention_save_context


def cpu_attention(q, k, v):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return (scores.softmax(-1) @ v).transpose(1, 2)


def make_attention(dtype=torch.float32, legacy=False, reference=False, reads=None):
    def kernel(q, k, v):
        if reads is not None:
            reads.append((k.untyped_storage().data_ptr(), v.untyped_storage().data_ptr()))
        if reference:
            k, v = k.clone(), v.clone()
        return cpu_attention(q, k, v)
    cls = isolated_class("wan/modules/causal_model.py", "CausalWanSelfAttention",
                         dict(torch=torch, nn=nn, math=math, attention=kernel,
                              causal_rope_apply=lambda x, *a, **kw: x,
                              cached_attention_save_context=cached_attention_save_context), legacy=legacy)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(173)
        module = cls(8, 2, local_attn_size=18, sink_size=0, qk_norm=False).to(dtype)
    # Two spatial tokens per frame instead of full-Wan 1560.
    module.max_attention_size = 18 * 2
    return module


def cache(dtype=torch.float32, frames=21):
    return dict(k=torch.zeros(1, frames * 2, 2, 4, dtype=dtype),
                v=torch.zeros(1, frames * 2, 2, 4, dtype=dtype),
                global_end_index=torch.zeros(1, dtype=torch.long),
                local_end_index=torch.zeros(1, dtype=torch.long))


def inputs(dtype=torch.float32, chunks=8):
    rng = torch.Generator().manual_seed(812)
    return [torch.randn(1, 6, 8, generator=rng).to(dtype) for _ in range(chunks)]


def forward(module, x, kv, index):
    return module(x, torch.tensor([6]), torch.tensor([[3, 1, 2]]), None, None,
                  kv_cache=kv, current_start=index * 6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nograd_eight_chunks_exact_legacy_fifo_and_aliasing(dtype):
    reads = []
    ours, old = make_attention(dtype, reads=reads), make_attention(dtype, legacy=True)
    a, b = cache(dtype, 18), cache(dtype, 18)
    original_objects = {name: a[name] for name in a}
    with torch.no_grad():
        for i, x in enumerate(inputs(dtype)):
            y, ref = forward(ours, x, a, i), forward(old, x, b, i)
            assert torch.equal(y, ref)
            # A repeated context refresh of this same chunk must not evict twice.
            z, ref_z = forward(ours, x * 0.5, a, i), forward(old, x * 0.5, b, i)
            assert torch.equal(z, ref_z)
            assert all(torch.equal(a[name], b[name]) for name in a)
            assert all(a[name] is original_objects[name] for name in a)
            assert int(a['global_end_index']) == 6 * (i + 1)
            assert int(a['local_end_index']) == min(36, 6 * (i + 1))
            assert reads[-1] == (a['k'].untyped_storage().data_ptr(),
                                 a['v'].untyped_storage().data_ptr())
    # After chunks 7/8, the contents are precisely the most recent six chunks.
    expected_k, expected_v = [], []
    with torch.no_grad():
        for x in inputs(dtype)[2:]:
            expected_k.append(ours.k(x * 0.5).view(1, 6, 2, 4))
            expected_v.append(ours.v(x * 0.5).view(1, 6, 2, 4))
    assert torch.equal(a['k'], torch.cat(expected_k, dim=1))
    assert torch.equal(a['v'], torch.cat(expected_v, dim=1))


def rollout(module, kv, xs):
    outputs = []
    for i, x in enumerate(xs):
        outputs.append(forward(module, x, kv, i))
        with torch.no_grad():
            forward(module, x * 0.5, kv, i)
    return torch.stack(outputs)


def test_legacy_saved_views_fail_after_context_refresh():
    old = make_attention(legacy=True)
    loss = rollout(old, cache(), inputs(chunks=7)).square().mean()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_seven_chunk_backward_matches_immutable_kernel_reference(dtype):
    reads = []
    ours = make_attention(dtype, reads=reads)
    ref = make_attention(dtype, legacy=True, reference=True)
    a, b = cache(dtype), cache(dtype)
    xs = inputs(dtype, chunks=7)
    y, expected = rollout(ours, a, xs), rollout(ref, b, xs)
    assert torch.equal(y, expected)
    assert all(torch.equal(a[name], b[name]) for name in a)
    # Grad-enabled reads get private storage; no-grad context uses the live cache.
    pointer = (a['k'].untyped_storage().data_ptr(), a['v'].untyped_storage().data_ptr())
    assert all(p != pointer for p in reads[::2])
    assert all(p == pointer for p in reads[1::2])
    y.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, p), (ref_name, ref_p) in zip(ours.named_parameters(), ref.named_parameters()):
        assert name == ref_name
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.equal(p.grad, ref_p.grad), name
    assert ours.k.weight.grad.norm() > 0 and ours.v.weight.grad.norm() > 0
    assert int(a['global_end_index']) == int(a['local_end_index']) == 42


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_saved_tensor_offload_exact_outputs_gradients_and_cache(dtype, monkeypatch):
    original = torch.autograd.graph.save_on_cpu
    records, restored, options = [], [], []

    class TracedSaveOnCPU(original):
        def __init__(self, **kwargs):
            options.append(kwargs)
            super().__init__(**kwargs)
            pack, unpack = self.pack_hook, self.unpack_hook

            def trace_pack(t):
                result = pack(t)
                assert result[1].device.type == "cpu"
                records.append((result[1], result[1].clone()))
                return result

            def trace_unpack(saved):
                result = unpack(saved)
                restored.append(result.device)
                return result

            self.pack_hook, self.unpack_hook = trace_pack, trace_unpack

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", TracedSaveOnCPU)
    offloaded, reference = make_attention(dtype), make_attention(dtype)
    offloaded.cached_attention_cpu_offload = True
    a, b = cache(dtype), cache(dtype)
    xs = inputs(dtype, chunks=7)
    # Also inspect input edges, not just shared parameter gradient norms.
    left = [x.clone().requires_grad_() for x in xs]
    right = [x.clone().requires_grad_() for x in xs]
    y, expected = rollout(offloaded, a, left), rollout(reference, b, right)
    assert len(options) == 7 and all(o == {"pin_memory": False} for o in options)
    assert records and all(torch.equal(t, saved) for t, saved in records)
    assert torch.equal(y, expected)
    assert all(torch.equal(a[name], b[name]) for name in a)
    y.float().square().mean().backward()
    expected.float().square().mean().backward()
    assert restored
    for x, ref_x in zip(left, right):
        assert x.grad is not None and torch.equal(x.grad, ref_x.grad)
    for (name, p), (ref_name, ref_p) in zip(offloaded.named_parameters(), reference.named_parameters()):
        assert name == ref_name and p.grad is not None
        assert torch.isfinite(p.grad).all() and torch.equal(p.grad, ref_p.grad), name


def test_offload_disabled_and_nograd_do_not_enter_saved_tensor_hooks(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("No-grad/disabled path must not enter CPU offload")
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    module = make_attention()
    forward(module, inputs(chunks=1)[0], cache(), 0)
    module.cached_attention_cpu_offload = True
    with torch.no_grad():
        forward(module, inputs(chunks=1)[0], cache(), 0)
