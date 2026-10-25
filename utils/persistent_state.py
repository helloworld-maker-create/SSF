"""Small state modules and explicit, sequence-local rollout plumbing."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PersistentStateConfig:
    enabled: bool = False
    num_slots: int = 8  # Legacy config spelling: count of generic state tokens.
    reader_block: int = 15
    writer_block: int = 20
    attention_dim: int = 256
    num_heads: int = 4
    init_seed: int = 0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("Persistent State enabled must be a boolean")
        dimensions = (self.num_slots, self.attention_dim, self.num_heads)
        indices = (self.reader_block, self.writer_block, self.init_seed)
        if any(type(value) is not int for value in dimensions + indices):
            raise ValueError("Persistent State dimensions, indices and seed must be integers")
        if min(dimensions) <= 0:
            raise ValueError("State token count and attention dimensions must be positive")
        if self.attention_dim % self.num_heads:
            raise ValueError("State attention_dim must be divisible by num_heads")
        if min(indices) < 0:
            raise ValueError("State block indices and seed must be nonnegative")

    @classmethod
    def from_config(cls, config=None):
        if isinstance(config, cls):
            return config
        return cls(**dict(config or {}))


class StateCrossAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim=256, num_heads=4):
        super().__init__()
        if attention_dim <= 0 or num_heads <= 0 or attention_dim % num_heads:
            raise ValueError("Invalid state attention dimensions")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.q = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.k = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.out = nn.Linear(attention_dim, hidden_dim, bias=False)

    def forward(self, query, context):
        if (query.ndim != 3 or context.ndim != 3 or
                query.shape[0] != context.shape[0] or
                query.shape[-1] != self.hidden_dim or
                context.shape[-1] != self.hidden_dim):
            raise ValueError("State attention expects matching [B, tokens, hidden_dim] tensors")
        def heads(x):
            return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            heads(self.q(query)), heads(self.k(context)), heads(self.v(context)),
            dropout_p=0.0,
        )
        return self.out(x.transpose(1, 2).flatten(2))


class StateReader(nn.Module):
    def __init__(self, hidden_dim, attention_dim=256, num_heads=4):
        super().__init__()
        self.attention = StateCrossAttention(hidden_dim, attention_dim, num_heads)
        nn.init.zeros_(self.attention.out.weight)

    def forward(self, hidden, state):
        # Always compute the residual: the zero output projection can learn.
        return hidden + self.attention(hidden, state)


class StateWriter(nn.Module):
    def __init__(self, hidden_dim, attention_dim=256, num_heads=4, num_state_tokens=8):
        super().__init__()
        if type(num_state_tokens) is not int or num_state_tokens <= 0:
            raise ValueError("num_state_tokens must be a positive integer")
        self.attention = StateCrossAttention(hidden_dim, attention_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        # Token identities are learned Writer parameters, never rollout state.
        self.state_token_embeddings = nn.Parameter(torch.empty(num_state_tokens, hidden_dim))
        nn.init.normal_(self.state_token_embeddings, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, previous_state, context_hidden):
        if previous_state.ndim != 3 or previous_state.shape[1:] != self.state_token_embeddings.shape:
            raise ValueError("State must have shape [batch, num_state_tokens, hidden_dim]")
        query = self.query_norm(previous_state + self.state_token_embeddings.unsqueeze(0))
        return self.norm(previous_state + self.attention(query, context_hidden))


def build_state_modules(hidden_dim, config):
    """Identical CPU initialization on every rank, without consuming caller RNG.

    Seed only the CPU generator; torch.manual_seed would also seed CUDA.
    Branch construction occurs after pretrained DiT loading and before FSDP.
    """
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(config.init_seed)
        reader = StateReader(hidden_dim, config.attention_dim, config.num_heads)
        writer = StateWriter(hidden_dim, config.attention_dim, config.num_heads,
                             num_state_tokens=config.num_slots)
    return reader, writer


def initialize_rollout_state(generator, noise):
    config = getattr(generator, "persistent_state_config", None)
    if config is None or not config.enabled:
        return None
    return noise.new_zeros(
        noise.shape[0], config.num_slots, generator.persistent_state_hidden_dim)


def state_forward_kwargs(state, role):
    # Disabled callers pass precisely their original kwargs/return contract.
    if state is None:
        return {}
    return {
        "forward_role": role,
        "persistent_state": state,
        "return_context_hidden": role == "context_refresh",
    }


def update_rollout_state(generator, state, context_result):
    if state is None:
        return None
    # TBPTT-1: cut both incoming edges, but keep this Writer's output attached.
    # Called after the training pipeline exits context-refresh no_grad. Preserve
    # caller grad mode: critic/inference no_grad must not be overridden here.
    return generator(
        forward_role="state_write", persistent_state=state.detach(),
        state_write_hidden=context_result[2].detach())


def load_generator_state_dict(generator, state_dict):
    """Load legacy baseline weights, allowing only absent *whole* state branches.

    Full augmented checkpoints still require every parameter. This never hides
    missing backbone parameters or unexpected keys, and uses FSDP's own loader.
    """
    config = getattr(generator, "persistent_state_config", None)
    if config is None or not config.enabled:
        return generator.load_state_dict(state_dict, strict=True)
    branch_keys = set(generator.persistent_state_parameter_names)
    provided = branch_keys.intersection(state_dict)
    if provided and provided != branch_keys:
        raise RuntimeError("Checkpoint contains an incomplete Persistent State branch")
    result = generator.load_state_dict(state_dict, strict=False)
    allowed_missing = branch_keys if not provided else set()
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Generator checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}")
    return result
