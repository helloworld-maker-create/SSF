"""CPU autograd qualification of the actual state-transition helper and rollout.

Only tiny synthetic DiT blocks are constructed. The DMD integration probe uses
the unchanged source surrogate with synthetic score gradients; it is not training.
"""
import json
from types import SimpleNamespace
from typing import Optional, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from utils.persistent_state import initialize_rollout_state, update_rollout_state
from test_persistent_state import (
    cpu_only, cpu_threads, isolated_class, pipeline, run_rollout, wrapper,
)


def parameter_grad_norm(module):
    grads = [p.grad.detach().double().flatten() for p in module.parameters() if p.grad is not None]
    return torch.cat(grads).norm().item() if grads else 0.0


def tensor_grad_norm(tensor):
    return 0.0 if tensor.grad is None else tensor.grad.detach().double().norm().item()


def activate_reader(reader):
    # Test-only positive control; production initialization stays exactly zero.
    with torch.no_grad():
        reader.attention.out.weight.normal_(
            std=0.1, generator=torch.Generator(device="cpu").manual_seed(177))


def synthetic_case(active=True):
    torch.manual_seed(2026)
    generator = wrapper(enabled=True)
    if active:
        activate_reader(generator.model.state_reader)
    previous_state = torch.randn(2, 8, 16, requires_grad=True)
    context = torch.randn(2, 11, 16, requires_grad=True)
    target_hidden = torch.randn(2, 5, 16)
    target = torch.randn_like(target_hidden)
    return generator, previous_state, context, target_hidden, target


def transition(generator, previous_state, context):
    result = update_rollout_state(generator, previous_state, (None, None, context))
    assert result.requires_grad and result.grad_fn is not None
    result.retain_grad()
    return result


def future_loss(generator, state, hidden, target):
    return F.mse_loss(generator.model.state_reader(hidden, state), target)


def report(label, generator, state, previous_state, context):
    metrics = dict(
        reader=parameter_grad_norm(generator.model.state_reader),
        reader_output=generator.model.state_reader.attention.out.weight.grad.double().norm().item(),
        writer=parameter_grad_norm(generator.state_writer),
        S_next=tensor_grad_norm(state), S_prev=tensor_grad_norm(previous_state),
        H_ctx=tensor_grad_norm(context),
    )
    print(f"B2 {label}: {json.dumps(metrics, sort_keys=True)}")
    assert previous_state.grad is None and context.grad is None
    return metrics


def test_a_cold_start_reader_opens_writer_credit_is_exactly_zero():
    g, previous, context, hidden, target = synthetic_case(active=False)
    assert torch.count_nonzero(g.model.state_reader.attention.out.weight).item() == 0
    state = transition(g, previous, context)
    future_loss(g, state, hidden, target).backward()
    metrics = report("A cold start", g, state, previous, context)
    assert metrics["reader_output"] > 0
    assert metrics["writer"] == metrics["S_next"] == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad).item() == 0
               for p in g.state_writer.parameters())


@pytest.mark.parametrize("boundary", ["B active Reader", "C previous state", "D context hidden"])
def test_b_c_d_active_reader_credit_and_detached_inputs(boundary):
    g, previous, context, hidden, target = synthetic_case()
    writer_inputs = []
    handle = g.state_writer.register_forward_pre_hook(lambda module, args: writer_inputs.append(args))
    state = transition(g, previous, context)
    handle.remove()
    assert len(writer_inputs) == 1
    assert all(not tensor.requires_grad and tensor.grad_fn is None for tensor in writer_inputs[0])
    assert torch.equal(writer_inputs[0][0], previous)
    assert torch.equal(writer_inputs[0][1], context)
    future_loss(g, state, hidden, target).backward()
    metrics = report(boundary, g, state, previous, context)
    assert metrics["S_next"] > 0 and metrics["writer"] > 0
    assert g.state_writer.state_token_embeddings.grad.norm().item() > 0


def test_e_negative_control_detached_output_blocks_writer_credit():
    g, previous, context, hidden, target = synthetic_case()
    state = transition(g, previous, context)
    future_loss(g, state.detach(), hidden, target).backward()
    metrics = report("E detached S_next", g, state, previous, context)
    assert metrics["reader"] > 0 and metrics["writer"] == metrics["S_next"] == 0
    assert state.grad is None
    assert all(p.grad is None for p in g.state_writer.parameters())


def test_f_horizon_one_inspects_intermediate_tensors_with_shared_writer():
    g, initial, context1, hidden, target = synthetic_case()
    context2 = torch.randn_like(context1, requires_grad=True)
    state1 = transition(g, initial, context1)
    state2 = transition(g, state1, context2)
    # Inspect states and context tensors; shared parameter gradients alone cannot
    # identify which invocation of the same Writer received credit.
    loss = future_loss(g, state2, hidden, target)
    derivatives = torch.autograd.grad(
        loss, (state1, state2, initial, context1, context2),
        allow_unused=True, retain_graph=True)
    assert derivatives[0] is None and derivatives[2:] == (None, None, None)
    assert derivatives[1].norm().item() > 0
    state1.grad = state2.grad = None  # retain_grad may also run during autograd.grad.
    loss.backward()
    metrics = report("F horizon one", g, state2, initial, context2)
    assert metrics["writer"] > 0 and metrics["S_next"] > 0
    assert state1.grad is None and context1.grad is None
    print("B2 F earlier transition: S1.grad=None, S0.grad=None, H_ctx1.grad=None")


def test_g_cold_start_then_mock_optimizer_step_opens_writer_credit():
    g, previous, context, hidden, target = synthetic_case(active=False)
    optimizer = torch.optim.SGD(
        list(g.model.state_reader.parameters()) + list(g.state_writer.parameters()), lr=0.01)
    before_writer = [p.detach().clone() for p in g.state_writer.parameters()]
    first_state = transition(g, previous, context)
    future_loss(g, first_state, hidden, target).backward()
    first = report("G first backward", g, first_state, previous, context)
    assert first["writer"] == first["S_next"] == 0 and first["reader_output"] > 0
    optimizer.step()  # Tiny CPU module only; no training entry point is invoked.
    assert all(torch.equal(old, new) for old, new in zip(before_writer, g.state_writer.parameters()))
    assert g.model.state_reader.attention.out.weight.norm().item() > 0
    optimizer.zero_grad(set_to_none=True)
    # Numerically continue the recurrence with new parameters and a fresh graph.
    second_state = transition(g, first_state, context)
    future_loss(g, second_state, hidden, target).backward()
    second = report("G second backward", g, second_state, previous, context)
    assert second["writer"] > 0 and second["S_next"] > 0
    assert tensor_grad_norm(first_state) == 0


@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_transition_respects_critic_and_inference_gradient_modes(mode):
    g, previous, context, _, _ = synthetic_case()
    with getattr(torch, mode)():
        result = update_rollout_state(g, previous, (None, None, context))
    assert not result.requires_grad and result.grad_fn is None


def synthetic_dmd_loss(generator, output):
    # Execute the repository's exact holistic surrogate with deterministic
    # synthetic score gradients; no full teacher, critic, or Wan is constructed.
    namespace = dict(torch=torch, F=F, Optional=Optional, Tuple=Tuple,
                     SelfForcingModel=nn.Module, SelfForcingTrainingPipeline=object)
    cls = isolated_class("model/dmd.py", "DMD", namespace)
    dmd = cls.__new__(cls)
    nn.Module.__init__(dmd)
    dmd.scheduler = generator.get_scheduler()
    dmd.num_frame_per_block = 3
    dmd.ts_schedule = dmd.ts_schedule_max = False
    dmd.min_score_timestep, dmd.num_train_timestep = 0, 1000
    dmd.timestep_shift, dmd.min_step, dmd.max_step = 1, 20, 980
    dmd._get_timestep = lambda low, high, batch, frames, *a, **kw: torch.full((batch, frames), 300)
    score_grad = torch.linspace(0.1, 1.0, output.numel()).reshape_as(output)
    dmd._compute_kl_grad = lambda **kw: (score_grad, {})
    loss, _ = dmd.compute_distribution_matching_loss(output, {}, {})
    return loss


@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_training_rollout_holistic_dmd_backward_and_critic_reset(checkpointing):
    torch.manual_seed(2026)
    g = wrapper(enabled=True)
    activate_reader(g.model.state_reader)
    if checkpointing:
        g.enable_gradient_checkpointing()
    p = pipeline(g)
    writes, events = [], []
    def record_write(module, args, result):
        assert torch.is_grad_enabled()
        assert all(not value.requires_grad for value in args)
        result.retain_grad()
        writes.append(result)
    handle = g.state_writer.register_forward_hook(record_write)
    original_forward = g.forward
    def record_forward(*args, **kwargs):
        role = kwargs.get("forward_role")
        result = original_forward(*args, **kwargs)
        events.append((role, torch.is_grad_enabled(), kwargs.get("persistent_state")))
        if role == "context_refresh":
            assert not torch.is_grad_enabled() and not result[2].requires_grad
        return result
    g.forward = record_forward
    noise = torch.randn(1, 12, 2, 1, 1)
    output = run_rollout(p, noise)[0]
    assert len(writes) == 4
    assert sum(role == "denoise" and grad for role, grad, _ in events) == 4
    assert sum(role == "context_refresh" for role, _, _ in events) == 4
    for chunk in range(4):
        block_events = events[chunk * 6:(chunk + 1) * 6]
        assert [role for role, _, _ in block_events] == ["denoise"] * 4 + ["context_refresh", "state_write"]
        incoming = block_events[0][2]
        assert all(state is incoming for _, _, state in block_events[:5])
        if chunk:
            assert incoming is writes[chunk - 1]
        else:
            assert torch.count_nonzero(incoming).item() == 0
    before_calls = [block.calls for block in g.model.blocks]
    synthetic_dmd_loss(g, output).backward()
    after_calls = [block.calls for block in g.model.blocks]
    if checkpointing:
        assert all(after > before for before, after in zip(before_calls, after_calls))
    else:
        assert before_calls == after_calls
    assert len(writes) == 4  # No transition during checkpoint recomputation.
    state_norms = [tensor_grad_norm(state) for state in writes]
    assert all(value > 0 for value in state_norms[:-1]) and writes[-1].grad is None
    assert parameter_grad_norm(g.state_writer) > 0
    assert parameter_grad_norm(g.model.state_reader) > 0
    print(f"B2 actual rollout checkpointing={checkpointing}: "
          f"Writer_grad={parameter_grad_norm(g.state_writer)}, "
          f"Reader_grad={parameter_grad_norm(g.model.state_reader)}, state_grads={state_norms}, "
          f"updates=4, block_calls={before_calls}->{after_calls}")
    handle.remove()
    generator_initial_state = events[0][2]
    events.clear()
    with torch.no_grad():
        critic_output = run_rollout(p, noise)[0]
    critic_initial_state = events[0][2]
    assert critic_initial_state.data_ptr() != generator_initial_state.data_ptr()
    assert torch.count_nonzero(critic_initial_state).item() == 0
    assert not critic_output.requires_grad
    assert all(not grad for _, grad, _ in events)


def test_disabled_transition_still_has_original_noop_contract():
    assert update_rollout_state(SimpleNamespace(), None, ()) is None
    g = wrapper(enabled=True)
    first = initialize_rollout_state(g, torch.ones(2, 3, 2, 1, 1))
    second = initialize_rollout_state(g, torch.ones(2, 3, 2, 1, 1))
    assert first.data_ptr() != second.data_ptr()
    assert torch.count_nonzero(first).item() == torch.count_nonzero(second).item() == 0
