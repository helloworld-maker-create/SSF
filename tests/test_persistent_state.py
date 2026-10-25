"""CPU-only Persistent State regression tests: execute real source methods with small synthetic DiT blocks.

AST isolation avoids importing WAN's eager asset/GPU/optional-package stack.
It does not replace the wrapper, rollout, or cached block-loop implementations.
Legacy versions come from the audited baseline commit for paired checks.
"""
import ast
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
from types import SimpleNamespace, MethodType
from typing import List, Optional

import pytest
import torch
import torch.utils.checkpoint
from torch import nn
from omegaconf import OmegaConf

from utils.persistent_state import (
    PersistentStateConfig, StateReader, StateWriter, build_state_modules,
    initialize_rollout_state, state_forward_kwargs, update_rollout_state,
    load_generator_state_dict,
)
from utils.scheduler import FlowMatchScheduler, SchedulerInterface
from utils.crossattn_cache import cached_generator_forward, unwrap_crossattn_cache
from utils.activation_offload import cached_attention_save_context


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "33593df3e81fa3ec10239271dd2c100facac6de1"


def source(path, legacy=False):
    if legacy:
        return subprocess.check_output(
            ["git", "show", BASELINE_COMMIT + ":" + path], cwd=ROOT, text=True)
    return (ROOT / path).read_text()


def isolated_class(path, name, namespace, legacy=False):
    tree = ast.parse(source(path, legacy))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace["cached_generator_forward"] = cached_generator_forward
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


def model_method(name, legacy=False):
    tree = ast.parse(source("wan/modules/causal_model.py", legacy))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalWanModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(torch=torch, unwrap_crossattn_cache=unwrap_crossattn_cache,
                     cached_attention_save_context=cached_attention_save_context, math=__import__("math"),
                     sinusoidal_embedding_1d=lambda dim, t: t[:, None].expand(-1, dim) / 1000)
    exec(compile(module, str(ROOT / "wan/modules/causal_model.py"), "exec"), namespace)
    return namespace[name]


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.calls = 0
        self.last_hidden = None

    def forward(self, x, kv_cache=None, current_start=0, **kwargs):
        self.calls += 1
        y = x + self.linear(x).tanh()
        self.last_hidden = y
        if kv_cache is not None:
            kv_cache["writes"].append((current_start, y.detach().clone()))
            kv_cache["global_end_index"].fill_(current_start + x.shape[1])
            kv_cache["local_end_index"].fill_(current_start + x.shape[1])
        return y


class TinyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Linear(dim, 2)

    def forward(self, x, e):
        return self.out(x)


def tiny_model(legacy=False, depth=3):
    class TinyDiT(nn.Module):
        _forward_inference = model_method("_forward_inference", legacy)
        unpatchify = model_method("unpatchify", legacy)
        forward = model_method("forward", legacy)

        def __init__(self, local_attn_size=-1, sink_size=0):
            super().__init__()
            self.dim, self.freq_dim, self.text_len = 16, 4, 4
            self.out_dim, self.patch_size = 2, (1, 1, 1)
            self.model_type = "t2v"
            self.local_attn_size, self.sink_size = local_attn_size, sink_size
            self.patch_embedding = nn.Conv3d(2, 16, 1)
            self.time_embedding = nn.Linear(4, 16)
            self.time_projection = nn.Linear(16, 96)
            self.text_embedding = nn.Linear(16, 16)
            self.blocks = nn.ModuleList([TinyBlock(16) for _ in range(depth)])
            self.head = TinyHead(16)
            self.freqs = torch.zeros(1)
            self.block_mask = None
            self.gradient_checkpointing = False

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls(**kwargs)

        def _forward_train(self, x, **kwargs):
            if kwargs.get("classify_mode"):
                return x * 0.2, torch.ones(x.shape[0], 4)
            return x * 0.2

        def enable_gradient_checkpointing(self):
            self.gradient_checkpointing = True

    return TinyDiT


def wrapper(enabled=False, legacy=False, depth=3, state_options=None):
    namespace = dict(torch=torch, nn=nn, types=__import__("types"), List=List,
                     Optional=Optional, CausalWanModel=tiny_model(legacy, depth),
                     FlowMatchScheduler=FlowMatchScheduler, SchedulerInterface=SchedulerInterface,
                     PersistentStateConfig=PersistentStateConfig,
                     build_state_modules=build_state_modules)
    cls = isolated_class("utils/wan_wrapper.py", "WanDiffusionWrapper", namespace, legacy)
    kwargs = dict(is_causal=True, timestep_shift=5, local_attn_size=18, sink_size=0)
    if not legacy:
        kwargs["persistent_state"] = dict(enabled=enabled, num_slots=8,
                                          reader_block=1, writer_block=2,
                                          attention_dim=8, num_heads=2)
        kwargs["persistent_state"].update(state_options or {})
    return cls(**kwargs)


def caches():
    return [dict(writes=[], global_end_index=torch.zeros(1, dtype=torch.long),
                 local_end_index=torch.zeros(1, dtype=torch.long)) for _ in range(3)]


def forward_args():
    return dict(noisy_image_or_video=torch.randn(2, 3, 2, 1, 1),
                conditional_dict={"prompt_embeds": torch.randn(2, 2, 16)},
                timestep=torch.full((2, 3), 750), kv_cache=caches(),
                crossattn_cache=[{} for _ in range(3)], current_start=0)


def pipeline(generator, inference=False, legacy=False):
    namespace = dict(torch=torch, nn=nn, Optional=Optional, List=List,
                     dist=torch.distributed, WanDiffusionWrapper=type(generator),
                     SchedulerInterface=SchedulerInterface,
                     PersistentStateConfig=PersistentStateConfig,
                     initialize_rollout_state=initialize_rollout_state,
                     state_forward_kwargs=state_forward_kwargs,
                     update_rollout_state=update_rollout_state)
    if inference:
        cls = isolated_class("pipeline/causal_inference.py", "CausalInferencePipeline", namespace, legacy)
        args = SimpleNamespace(denoising_step_list=[1000, 750, 500, 250],
                               warp_denoising_step=False, num_frame_per_block=3,
                               independent_first_frame=False, context_noise=0)
        class Text(nn.Module):
            def forward(self, text_prompts):
                return {"prompt_embeds": torch.ones(len(text_prompts), 2, 16)}
        class VAE(nn.Module):
            def decode_to_pixel(self, x, use_cache=False):
                return x
        p = cls(args, "cpu", generator=generator, text_encoder=Text(), vae=VAE())
    else:
        cls = isolated_class("pipeline/self_forcing_training.py", "SelfForcingTrainingPipeline", namespace, legacy)
        p = cls(torch.tensor([1000, 750, 500, 250]), generator.get_scheduler(), generator,
                same_step_across_blocks=True)
        p.generate_and_sync_list = lambda n, steps, device: [steps - 1] * n
    p.num_transformer_blocks, p.frame_seq_length = 3, 1
    def init_kv(self, **kwargs):
        self.kv_cache1 = caches()
    def init_cross(self, **kwargs):
        self.crossattn_cache = [dict(is_init=False) for _ in range(3)]
    p._initialize_kv_cache = MethodType(init_kv, p)
    p._initialize_crossattn_cache = MethodType(init_cross, p)
    return p


def run_rollout(p, noise, inference=False):
    if inference:
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            return p.inference(noise, ["synthetic"] * noise.shape[0], return_latents=True)
    return p.inference_with_trajectory(noise, prompt_embeds=torch.ones(noise.shape[0], 2, 16))


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # Legacy training converts schedule tensors with .cuda(); isolate that
    # device operation rather than altering production scheduling behavior.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    yield
    assert not torch.cuda.is_initialized()


def test_a_legacy_wrapper_contract_and_disabled_registration():
    torch.manual_seed(91)
    old = wrapper(legacy=True)
    torch.manual_seed(91)
    new = wrapper()
    assert set(old.state_dict()) == set(new.state_dict())
    assert not hasattr(new, "state_writer") and not hasattr(new.model, "state_reader")
    args = forward_args()
    for kwargs in (args, {**args, "kv_cache": None}):
        expected, actual = old(**copy.deepcopy(kwargs)), new(**copy.deepcopy(kwargs))
        assert isinstance(actual, tuple) and len(actual) == 2
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    # Original positional call and the existing classifier three-tuple.
    positional = new(args["noisy_image_or_video"], args["conditional_dict"], args["timestep"])
    assert len(positional) == 2
    for g in (old, new):
        g._register_tokens = g._cls_pred_branch = g._gan_ca_blocks = None
    expected = old(**{**args, "kv_cache": None, "classify_mode": True})
    actual = new(**{**args, "kv_cache": None, "classify_mode": True})
    assert len(actual) == 3 and all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert initialize_rollout_state(new, args["noisy_image_or_video"]) is None
    assert state_forward_kwargs(None, "denoise") == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_b_zero_reader_and_future_trainability(dtype):
    torch.manual_seed(23)
    reader = StateReader(1536).to(dtype=dtype)
    hidden = torch.randn(2, 37, 1536, dtype=dtype)
    state = torch.randn(2, 8, 1536, dtype=dtype)
    output = reader(hidden, state)
    error = (output.float() - hidden.float())
    print(f"Reader {dtype}: max_abs={error.abs().max().item()}, MSE={error.square().mean().item()}")
    assert torch.equal(output, hidden)
    output.float().square().mean().backward()
    assert reader.attention.out.weight.grad.abs().sum() > 0


def test_c_d_state_shapes_writer_change_and_differentiability():
    config = PersistentStateConfig(enabled=True)
    generator = SimpleNamespace(persistent_state_config=config, persistent_state_hidden_dim=1536)
    state = initialize_rollout_state(generator, torch.zeros(2, 3, 16, 1, 1))
    hidden = torch.randn(2, 4680, 1536)
    writer = StateWriter(1536)
    output = writer(state, hidden)
    assert state.shape == output.shape == (2, 8, 1536)
    change = (output - state).norm().item()
    print(f"Writer shape={tuple(output.shape)}, state_change_norm={change}")
    assert change > 0
    (output * torch.randn_like(output)).sum().backward()
    assert writer.attention.out.weight.grad.abs().sum() > 0
    assert writer.attention.v.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_token_identities_break_zero_state_symmetry(dtype):
    config = PersistentStateConfig(enabled=True)
    generator = SimpleNamespace(persistent_state_config=config, persistent_state_hidden_dim=1536)
    _, writer = build_state_modules(1536, config)
    writer = writer.to(dtype=dtype)
    state = initialize_rollout_state(generator, torch.zeros(2, 3, 16, 1, 1, dtype=dtype))
    context_rng = torch.Generator(device="cpu").manual_seed(91)
    hidden = torch.randn(2, 4680, 1536, generator=context_rng).to(dtype=dtype)
    assert torch.count_nonzero(state).item() == 0
    output = writer(state, hidden)
    assert output.shape == state.shape == (2, 8, 1536)
    # Each sequence has 28 distinct state-token pairs; test all, not just one pair.
    distances = torch.stack([torch.pdist(tokens.float()) for tokens in output])
    assert (distances > 0).all()
    assert torch.count_nonzero(state).item() == 0
    print(f"State-token diversity {dtype}: mean_pairwise_L2={distances.mean().item()}, "
          f"min_pairwise_L2={distances.min().item()}, S_0_nonzero=0")
    if dtype == torch.float32:
        (output * torch.randn(output.shape, generator=context_rng)).sum().backward()
        assert writer.state_token_embeddings.grad.abs().sum() > 0


def test_state_token_count_configuration_and_parameter_delta():
    reader, writer = build_state_modules(1536, PersistentStateConfig(enabled=True))
    assert writer.state_token_embeddings.shape == (8, 1536)
    assert writer.state_token_embeddings.numel() == 12288
    assert sum(p.numel() for m in (reader, writer) for p in m.parameters()) == 3161088
    assert not list(writer.query_norm.parameters())
    _, other = build_state_modules(16, PersistentStateConfig(enabled=True, num_slots=5))
    assert other(torch.zeros(2, 5, 16), torch.randn(2, 9, 16)).shape == (2, 5, 16)
    with pytest.raises(ValueError, match="State must have shape"):
        other(torch.zeros(2, 8, 16), torch.randn(2, 9, 16))


@pytest.mark.parametrize("inference", [False, True])
def test_e_f_reset_and_one_writer_per_chunk(inference):
    g = wrapper(enabled=True)
    p = pipeline(g, inference)
    events = []
    original = g.forward
    def record(*args, **kwargs):
        state = kwargs.get("persistent_state")
        events.append((kwargs.get("forward_role"), kwargs.get("current_start"), state))
        return original(*args, **kwargs)
    g.forward = record
    writes = []
    handle = g.state_writer.register_forward_hook(lambda module, args, output: writes.append(output))
    batch_size = 2 if inference else 1  # Shipped training uses batch_size=1.
    noise = torch.randn(batch_size, 21, 2, 1, 1)
    for sequence in range(2):
        start = len(events)
        with torch.set_grad_enabled(not sequence and not inference):
            run_rollout(p, noise, inference)
        current = events[start:]
        assert sum(role == "state_write" for role, _, _ in current) == 7
        assert sum(role == "context_refresh" for role, _, _ in current) == 7
        assert sum(role == "denoise" for role, _, _ in current) == 28
        first = current[0][2]
        assert first.shape == (batch_size, 8, 16) and torch.count_nonzero(first) == 0
        if sequence:
            assert first.data_ptr() != prior_first.data_ptr()
        prior_first = first
        for offset in range(0, 21, 3):
            reads = [s for role, pos, s in current if role == "denoise" and pos == offset]
            refresh = next(s for role, pos, s in current if role == "context_refresh" and pos == offset)
            assert len(reads) == 4 and all(s is refresh for s in reads)
            if offset:
                assert reads[0] is writes[sequence * 7 + offset // 3 - 1]
                assert torch.count_nonzero(reads[0]) > 0
        assert "state" not in vars(p) and "persistent_state" not in vars(g)
    handle.remove()
    assert len(writes) == 14
    print(f"{'Inference' if inference else 'Training'}: 2 sequences, 7 chunks each, Writer calls={len(writes)}")


@pytest.mark.parametrize("inference", [False, True])
def test_paired_legacy_disabled_and_zero_enabled_rollout(inference):
    noise = torch.randn(2 if inference else 1, 21, 2, 1, 1)
    outputs, kv, rng = [], [], []
    for enabled, legacy in [(False, True), (False, False), (True, False)]:
        torch.manual_seed(43)
        g = wrapper(enabled, legacy)
        p = pipeline(g, inference, legacy)
        torch.manual_seed(67)
        outputs.append(run_rollout(p, noise, inference))
        kv.append(p.kv_cache1)
        rng.append(torch.random.get_rng_state())
    for result in outputs[1:]:
        assert type(result) is type(outputs[0]) and len(result) == len(outputs[0])
        for a, b in zip(result, outputs[0]):
            assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    for cache in kv[1:]:
        for a, b in zip(cache, kv[0]):
            assert torch.equal(a["global_end_index"], b["global_end_index"])
            assert len(a["writes"]) == len(b["writes"])
            assert all(i == j and torch.equal(x, y) for (i, x), (j, y) in zip(a["writes"], b["writes"]))
    assert all(torch.equal(rng[0], r) for r in rng[1:])
    error = outputs[2][0] - outputs[0][0]
    print(f"Paired {'inference' if inference else 'training'}: max_abs={error.abs().max().item()}, MSE={error.square().mean().item()}")


def test_g_checkpoint_recompute_cannot_repeat_writer():
    g = wrapper(enabled=True)
    g.enable_gradient_checkpointing()
    args = forward_args()
    state = initialize_rollout_state(g, args["noisy_image_or_video"])
    writes = []
    handle = g.state_writer.register_forward_hook(lambda *a: writes.append(1))
    result = g(**args, **state_forward_kwargs(state, "context_refresh"))
    assert torch.equal(result[2], g.model.blocks[2].last_hidden)
    assert not writes
    next_state = update_rollout_state(g, state, result)
    assert len(writes) == 1 and next_state.requires_grad
    before = [b.calls for b in g.model.blocks]
    result[0].square().mean().backward()
    after = [b.calls for b in g.model.blocks]
    assert all(a > b for a, b in zip(after, before))
    assert len(writes) == 1
    handle.remove()
    print(f"Checkpoint block calls: {before} -> {after}; Writer calls=1")


def test_explicit_roles_and_capture_boundary():
    g = wrapper(enabled=True)
    args = forward_args()
    state = initialize_rollout_state(g, args["noisy_image_or_video"])
    reads = []
    handle = g.model.state_reader.register_forward_hook(lambda *a: reads.append(1))
    # A zero timestep is still denoise when explicitly designated as denoise.
    g(**{**args, "timestep": torch.zeros_like(args["timestep"])},
      **state_forward_kwargs(state, "denoise"))
    assert len(reads) == 1
    result = g(**args, **state_forward_kwargs(state, "context_refresh"))
    assert len(reads) == 1 and torch.equal(result[2], g.model.blocks[2].last_hidden)
    with pytest.raises(ValueError):
        g(**args, persistent_state=state)
    with pytest.raises(ValueError):
        g(**args, persistent_state=state, forward_role="denoise", return_context_hidden=True)
    with pytest.raises(ValueError):
        g(**args, persistent_state=state[:, :7], forward_role="denoise")
    handle.remove()


def test_rank_identical_initialization_and_global_rng_preservation():
    branches = []
    for rank_seed in [19, 1234]:
        torch.manual_seed(rank_seed)
        before = torch.random.get_rng_state().clone()
        branches.append(build_state_modules(1536, PersistentStateConfig(enabled=True)))
        assert torch.equal(before, torch.random.get_rng_state())
    for left, right in zip(branches[0], branches[1]):
        assert all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters()))
    assert torch.equal(branches[0][1].state_token_embeddings, branches[1][1].state_token_embeddings)


def test_registration_and_scoped_checkpoint_loading():
    base = wrapper()
    g = wrapper(enabled=True)
    branch_keys = set(g.persistent_state_parameter_names)
    assert branch_keys.issubset(dict(g.named_parameters()))
    assert "state_writer.state_token_embeddings" in branch_keys
    load_generator_state_dict(g, base.state_dict())
    load_generator_state_dict(g, g.state_dict())
    broken = dict(base.state_dict())
    broken.pop("model.patch_embedding.weight")
    with pytest.raises(RuntimeError):
        load_generator_state_dict(g, broken)
    partial = dict(base.state_dict())
    partial[next(iter(branch_keys))] = g.state_dict()[next(iter(branch_keys))]
    with pytest.raises(RuntimeError):
        load_generator_state_dict(g, partial)
    with pytest.raises(RuntimeError):
        load_generator_state_dict(base, broken)


def test_configs_and_unchanged_baseline_algorithms():
    shipped = OmegaConf.load(ROOT / "configs/self_forcing_dmd.yaml")
    baseline = OmegaConf.load(ROOT / "configs/sf_local18.yaml")
    ours = OmegaConf.load(ROOT / "configs/ssf_local18_state.yaml")
    for cfg in (baseline, ours):
        assert cfg.model_kwargs.local_attn_size == 18 and cfg.model_kwargs.sink_size == 0
        cfg = OmegaConf.to_container(cfg, resolve=True)
        del cfg["persistent_state"]
        del cfg["model_kwargs"]["local_attn_size"]
        del cfg["model_kwargs"]["sink_size"]
        assert cfg == OmegaConf.to_container(shipped, resolve=True)
    assert not baseline.persistent_state.enabled and ours.persistent_state.enabled
    for path in ["model/dmd.py", "utils/scheduler.py", "configs/self_forcing_dmd.yaml"]:
        assert source(path) == source(path, legacy=True)
    current = ast.parse(source("wan/modules/causal_model.py"))
    legacy = ast.parse(source("wan/modules/causal_model.py", True))
    def attention(tree):
        return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalWanSelfAttention")
    # G4 needs two read snapshots for mutable KV lifetime, independent of State.
    # Remove only those precisely validated clone expressions. All cache writes,
    # FIFO/window arithmetic and attention math must still match the baseline.
    class RemoveReadSnapshots(ast.NodeTransformer):
        count = 0
        offload_scopes = 0

        def visit_With(self, node):
            assert len(node.items) == 1 and len(node.body) == 1
            assert ast.unparse(node.items[0].context_expr) == (
                "cached_attention_save_context(getattr(self, 'cached_attention_cpu_offload', False), roped_query)")
            assert node.items[0].optional_vars is None
            assert isinstance(node.body[0], ast.Assign)
            assert ast.unparse(node.body[0].targets[0]) == "x"
            assert ast.unparse(node.body[0].value.func) == "attention"
            self.offload_scopes += 1
            return self.visit(node.body[0])

        def visit_IfExp(self, node):
            if ast.unparse(node.test) == "torch.is_grad_enabled()":
                assert isinstance(node.body, ast.Call) and not node.body.args and not node.body.keywords
                assert isinstance(node.body.func, ast.Attribute) and node.body.func.attr == "clone"
                assert ast.dump(node.body.func.value) == ast.dump(node.orelse)
                self.count += 1
                return node.orelse
            return self.generic_visit(node)
    snapshots = RemoveReadSnapshots()
    normalized_attention = snapshots.visit(copy.deepcopy(attention(current)))
    assert snapshots.count == 2
    assert snapshots.offload_scopes == 1
    assert ast.dump(normalized_attention) == ast.dump(attention(legacy))
    for path in ["pipeline/self_forcing_training.py", "pipeline/causal_inference.py"]:
        tree = ast.parse(source(path))
        assert not any(isinstance(n, ast.Attribute) and n.attr in {"state_writer", "state_reader"}
                       for n in ast.walk(tree))


def test_config_validation():
    for values in [dict(num_slots=0), dict(attention_dim=7, num_heads=4),
                   dict(reader_block=-1), dict(writer_block=-1),
                   dict(enabled="false"), dict(num_slots=8.5)]:
        with pytest.raises(ValueError):
            PersistentStateConfig.from_config(values)


def test_candidate_boundaries_in_real_cached_loop():
    # Thirty tiny blocks exercise the actual candidate indices without a large DiT.
    g = wrapper(enabled=True, depth=30,
                state_options=dict(reader_block=15, writer_block=20))
    args = forward_args()
    args["kv_cache"] = caches() * 10
    args["crossattn_cache"] = [{} for _ in range(30)]
    state = initialize_rollout_state(g, args["noisy_image_or_video"])
    reads = []
    handle = g.model.state_reader.register_forward_pre_hook(lambda module, args: reads.append(args[0]))
    g(**args, **state_forward_kwargs(state, "denoise"))
    assert len(reads) == 1 and reads[0] is g.model.blocks[15].last_hidden
    result = g(**args, **state_forward_kwargs(state, "context_refresh"))
    assert len(reads) == 1
    assert result[2] is g.model.blocks[20].last_hidden
    assert not torch.equal(result[2], g.model.blocks[29].last_hidden)
    handle.remove()


def test_writer_calls_enclosing_wrapper_and_registration_order():
    class EnclosingWrapper(nn.Module):
        """Attribute forwarding like FSDP; this is not a distributed FSDP test."""
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.roles = []
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("module"), name)
        def forward(self, *args, **kwargs):
            self.roles.append(kwargs.get("forward_role"))
            return self.module(*args, **kwargs)
    wrapped = EnclosingWrapper(wrapper(enabled=True))
    p = pipeline(wrapped)
    run_rollout(p, torch.randn(1, 21, 2, 1, 1))
    assert wrapped.roles.count("state_write") == 7
    assert len(wrapped.roles) == 42
    trainer = ast.parse(source("trainer/distillation.py"))
    calls = [n for n in ast.walk(trainer) if isinstance(n, ast.Call)]
    model_line = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "DMD")
    wrap_line = min(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "fsdp_wrap")
    optimizer_line = min(n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "AdamW")
    assert model_line < wrap_line < optimizer_line


@pytest.mark.parametrize("inference,independent", [(False, False), (True, False), (True, True)])
def test_initial_context_updates_once(inference, independent):
    g = wrapper(enabled=True)
    p = pipeline(g, inference)
    p.independent_first_frame = independent
    written = []
    handle = g.state_writer.register_forward_hook(lambda module, args, output: written.append(output))
    noise = torch.randn(1, 6, 2, 1, 1)
    initial = torch.randn(1, 1 if independent or not inference else 3, 2, 1, 1)
    if inference:
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            p.inference(noise, ["synthetic"], initial_latent=initial)
    else:
        p.inference_with_trajectory(noise, initial_latent=initial,
                                    prompt_embeds=torch.ones(1, 2, 16))
    assert len(written) == 3  # One conditioning context + two generated chunks.
    handle.remove()
