# Persistent State — architecture through Phase B2

Scope: architecture plumbing, zero-contribution invariance, one-transition
recurrent credit assignment, and CPU unit/static tests. No training, dependency
changes, or asset wiring. B2 validation is documented in `persistent_state_b2.md`.
The reference baseline commit is `33593df3e81fa3ec10239271dd2c100facac6de1`.

## Configuration

Use `configs/sf_local18.yaml` for the bounded-KV baseline and
`configs/ssf_local18_state.yaml` for the state branch. Both retain the shipped
DMD configuration, with `model_kwargs.local_attn_size=18` and `sink_size=0`.
Asset paths are unchanged.

`persistent_state` defaults:

| Key | Default |
| --- | --- |
| `enabled` | `false` |
| `num_slots` (legacy spelling for state-token count) | `8` |
| `reader_block` | `15` (zero-based, after block output) |
| `writer_block` | `20` (zero-based, after block output) |
| `attention_dim` | `256` |
| `num_heads` | `4` |
| `init_seed` | `0` |

`num_slots` is retained as a public config key for compatibility. These are
generic Persistent State tokens, with no semantic/object-slot interpretation.

State width is derived from the loaded generator hidden width: 1536 for Wan 1.3B.
The candidate branch adds 3,161,088 trainable parameters at that width, including
12,288 learned Writer state-token identity parameters introduced in B1.1. Attention
uses bias-free Q/K/V/output projections and SDPA with zero dropout. It does not
pool video tokens or add semantic labels or extra encoders.

## Explicit lifecycle and contracts

Both rollout entry points allocate a local `S_0` using `noise.new_zeros`.
No module or pipeline stores the recurrent tensor as an attribute or buffer.

```text
initialize_rollout_state(generator, noise)
  -> S_0 [B, 8, hidden_dim]
for each chunk:
  generator(forward_role="denoise", persistent_state=S_k)
    -> CausalWanModel._forward_inference()
    -> block 15 output -> StateReader(H, S_k)
    -> existing flow / clean-latent prediction
  existing context refresh, with its original input/noise/timestep behavior
  generator(forward_role="context_refresh", persistent_state=S_k,
            return_context_hidden=True)
    -> unchanged DiT/KV computations, no Reader
    -> capture block 20 output H_ctx
  update_rollout_state(generator, S_k, context_result)
    -> enclosing generator(forward_role="state_write",
                           persistent_state=S_k, state_write_hidden=H_ctx)
    -> StateWriter(S_k.detach(), H_ctx.detach()) -> attached S_{k+1}
return existing pipeline outputs; discard local state
```

All denoising steps within a chunk receive the same incoming tensor. Provided
initial conditioning chunks also produce one Writer update per context forward.
Writer operations never run inside transformer blocks or checkpoint closures.

Legacy/default calls return the original `(flow_pred, pred_x0)` tuple (or the
existing classifier three-tuple). Explicit context capture returns
`(flow_pred, pred_x0, H_ctx)`. Explicit `state_write` returns a state tensor and
does not execute DiT, scheduler, or KV operations. Roles are never inferred from
timestep, training mode, or gradient mode.

The Reader output projection starts at zero. Its attention still executes, and
the output projection has a usable gradient for nonzero incoming state. No
trainable-zero gate shortcut is used.

## FSDP and initialization

Reader is registered as `generator.model.state_reader`; Writer is registered as
`generator.state_writer`. Construction follows pretrained DiT loading and
precedes FSDP wrapping, optimizer creation, and EMA parameter discovery.

Branch construction forks only the CPU RNG (`fork_rng(devices=[])`), seeds the
CPU default generator with `init_seed`, constructs on CPU, and restores the
caller CPU RNG. It does not seed or initialize CUDA. Identical configuration
therefore produces bit-identical branch parameters despite rank-specific seeds.
No change is made to `sync_module_states` or the existing backbone initialization.

Writer calls go through the enclosing generator's forward, including when that
generator is FSDP-wrapped. Pipeline helpers inspect only configuration metadata;
they never invoke sharded Reader/Writer submodules directly.

The scoped checkpoint loader permits a legacy checkpoint to omit the whole new
branch. Missing backbone keys, unexpected keys, and partial branch checkpoints
still fail. Disabled generators keep strict loading. Existing checkpoint
container selection and asset paths remain unchanged.

## TBPTT-1 gradient boundary and remaining risks

Training context-refresh DiT remains under `torch.no_grad()`. After leaving
that block, `update_rollout_state` detaches incoming state and captured context
hidden, then invokes the enclosing generator with the caller gradient mode.
The Writer output is not detached. A future chunk can train the immediately
preceding Writer transition through its Reader, while incoming-state and context
DiT gradients are blocked. Critic/inference callers under `no_grad` stay graph-free.
This is TBPTT horizon one, qualified with CPU synthetic tests in B2; no per-chunk
training loss or DMD change is introduced.

B1.1 breaks initial state-token symmetry with learned Writer parameters
`E_state [num_state_tokens, hidden_dim]`, initialized from a normal distribution with
standard deviation 0.02 inside the existing `init_seed` CPU RNG fork. Writer uses
`Q = LayerNorm(S_prev + E_state)` with a non-affine query LayerNorm, then computes
`S_next = LayerNorm(S_prev + CrossAttention(Q, H_ctx, H_ctx))` using the existing
output LayerNorm. Embeddings are added only to the query; recurrent state still
initializes to exactly zero. This removes exact query symmetry without claiming
learned semantic specialization. The registered parameter is now
`state_writer.state_token_embeddings` (renamed from the untrained B1.1
`slot_embeddings` parameter). The strict branch loader does not migrate old
experimental branch keys; baseline checkpoints remain supported.

CPU tests isolate actual source methods with AST extraction, synthetic blocks,
small caches, and a CPU replacement of the legacy schedule `.cuda()` calls.
They do not establish full Wan, FlashAttention, or distributed FSDP equivalence.
Existing mutable KV/checkpoint behavior and the training/inference context-noise
difference remain unchanged. A legacy context-noise broadcasting error for
training batch sizes above one remains; shipped training uses batch size one.
The independent streaming demo is not extended in B1.

Run the CPU suite without bytecode or pytest cache writes:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m pytest \
  -p no:cacheprovider -q -s tests/test_persistent_state.py
```

GPU/runtime qualification and training remain separate from this phase.

## B1.1 CPU validation

The complete suite passes: 22 tests, including all 19 B1 regression cases.
With zero initial state and synthetic context `[2, 4680, 1536]`, all 28 state-token
pairs per sequence are distinct. Mean/minimum pairwise L2 distances are
25.794794 / 21.042202 in FP32 and 25.794088 / 21.037291 in BF16.
Initial state remains exactly zero, and state-token embeddings receive a nonzero
gradient in a standalone differentiability test. Reader max absolute error and
MSE are both exactly zero in FP32/BF16 and in paired synthetic training/inference
rollouts. Different simulated rank seeds produce identical branch parameters
without consuming the caller RNG. Registration, checkpoint loading, sequence
reset, one update per chunk, and checkpoint recomputation regressions pass.
