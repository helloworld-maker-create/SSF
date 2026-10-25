"""Actual cached DiT loop and recurrent transition with tiny CPU modules."""
import copy

import pytest
import torch

from test_persistent_state import wrapper, forward_args
from test_persistent_state_b2 import activate_reader
from utils.persistent_state import initialize_rollout_state, state_forward_kwargs, update_rollout_state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_block_offload_preserves_recurrent_outputs_and_all_gradients(dtype):
    torch.manual_seed(913)
    reference = wrapper(enabled=True).to(dtype)
    activate_reader(reference.model.state_reader)
    offloaded = copy.deepcopy(reference)
    offloaded.model.cached_block_cpu_offload = True
    arguments = forward_args()
    arguments["noisy_image_or_video"] = arguments["noisy_image_or_video"].to(dtype)
    arguments["conditional_dict"]["prompt_embeds"] = arguments["conditional_dict"]["prompt_embeds"].to(dtype)
    outputs, states, guards, caches = [], [], [], []
    for generator in (reference, offloaded):
        kwargs = copy.deepcopy(arguments)
        state = initialize_rollout_state(generator, kwargs["noisy_image_or_video"])
        state.requires_grad_()
        with torch.no_grad():
            context = generator(**kwargs, **state_forward_kwargs(state, "context_refresh"))
        # Deliberately give both inputs grad fields to test the explicit detach.
        hidden = context[2].detach().requires_grad_()
        next_state = update_rollout_state(generator, state, (None, None, hidden))
        next_state.retain_grad()
        result = generator(**kwargs, **state_forward_kwargs(next_state, "denoise"))[0]
        result.float().square().mean().backward()
        assert state.grad is None and hidden.grad is None
        assert next_state.grad is not None and next_state.grad.float().norm() > 0
        assert generator.state_writer.attention.out.weight.grad.float().norm() > 0
        outputs.append(result.detach())
        states.append((next_state.detach(), next_state.grad.detach()))
        caches.append(kwargs["kv_cache"])
    assert torch.equal(*outputs)
    assert all(torch.equal(a, b) for a, b in zip(*states))
    for (name, left), (ref_name, right) in zip(reference.named_parameters(), offloaded.named_parameters()):
        assert name == ref_name
        if left.grad is None:
            assert right.grad is None
        else:
            assert torch.isfinite(right.grad).all() and torch.equal(left.grad, right.grad), name
    for left, right in zip(*caches):
        assert torch.equal(left["global_end_index"], right["global_end_index"])
        assert torch.equal(left["local_end_index"], right["local_end_index"])
        assert all(i == j and torch.equal(a, b)
                   for (i, a), (j, b) in zip(left["writes"], right["writes"]))
