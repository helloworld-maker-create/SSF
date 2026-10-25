# Phase B2: one-transition recurrent credit assignment

Status: CPU autograd and source-method qualification passed. No full Wan,
teacher/critic model, GPU, training entry point, download, or dependency change
was used. GPU/distributed runtime qualification remains a separate next phase.

## Implementation and boundaries

`SelfForcingTrainingPipeline.inference_with_trajectory` already performs each
context-refresh DiT forward and KV refresh inside `torch.no_grad()`, then calls
`update_rollout_state` after exiting that block. The same separation applies to
provided initial context. These pipeline statements were preserved.

`update_rollout_state` now performs:

```python
return generator(
    forward_role="state_write",
    persistent_state=state.detach(),
    state_write_hidden=context_result[2].detach(),
)
```

It preserves the caller's gradient mode instead of forcing `no_grad` or
`enable_grad`. Generator rollouts therefore construct a Writer graph; critic
and inference calls enclosed by `no_grad`/`inference_mode` do not. The enclosing
generator forward remains the execution boundary for FSDP. No Writer submodule
is invoked directly by the pipeline.

```text
context-refresh DiT / KV under no_grad -> H_ctx_k.detach() --+
                                                          |
S_k.detach() ------------------------------------------> Writer_k
                                                          |
                                                attached S_{k+1}
                                                          |
                                               future Reader / DiT
                                                          |
                                               future generated chunk
                                                          |
                                               existing holistic DMD

next transition: S_{k+1}.detach() -> Writer_{k+1}
```

The horizon-one restriction concerns the recurrent state-input edge. Writer
parameters are shared and may receive independent credits from multiple
transitions in the holistic loss. Numerically, state continues to roll forward.
All denoising calls in one chunk receive the same incoming state. Each context
refresh produces one update; checkpoint closures never contain Writer updates.
Fresh local zero state is allocated for each independent rollout, including
generator and critic rollouts that reuse a pipeline object.

The existing random exit denoising step and last-21-frame gradient eligibility
remain unchanged. Earlier denoising steps stay under their original `no_grad`
blocks. DMD, scheduler/context-noise behavior, Local18 KV, and Reader/Writer block
boundaries (15/20) were preserved. No per-chunk production loss was added.

## Terminology and compatibility

The recurrent vectors are generic **Persistent State tokens**. Writer uses
`state_token_embeddings` and its constructor names the count `num_state_tokens`.
The public/config key `persistent_state.num_slots` is retained with default 8
to preserve config compatibility; it means state-token count. This naming cleanup
does not change initialization, tensor dimensions, or mathematical behavior.
The token embedding parameter still has 12,288 elements at width 1536; total
branch parameter count remains 3,161,088.

The registered key changed from the untrained B1.1
`state_writer.slot_embeddings` to `state_writer.state_token_embeddings`.
No trained Persistent State checkpoint exists in this experiment. The strict
loader does not silently migrate earlier experimental branch keys. Whole-branch
absence in baseline checkpoints remains supported; partial branches still fail.

## Numerical autograd results

CPU FP32 tests execute the real transition helper and Wan wrapper's state-write
source method with a synthetic width-16 backbone, eight state tokens, and
two sequences. Parameter norms are aggregate L2 norms over all available
parameter gradients, accumulated in FP64. Reader output norms are also shown
separately. An unused tensor is reported as norm zero and checked as `grad=None`.

| Test | Reader norm | Reader output norm | Writer norm | S_next norm |
| --- | ---: | ---: | ---: | ---: |
| A: exact cold start | 0.139993535 | 0.139993535 | 0 | 0 |
| B: active Reader | 0.170540043 | 0.137772412 | 0.036403246 | 0.011398611 |
| C: previous state truncated | 0.170540043 | 0.137772412 | 0.036403246 | 0.011398611 |
| D: context hidden truncated | 0.170540043 | 0.137772412 | 0.036403246 | 0.011398611 |
| E: detached-output negative control | 0.170540043 | 0.137772412 | 0 | 0 |
| F: second transition only | 0.175164921 | 0.142467924 | 0.032070271 | 0.011330982 |
| G: first backward | 0.139993535 | 0.139993535 | 0 | 0 |
| G: after mock SGD step | 0.145675625 | 0.145674938 | 0.000282051 | 0.000063176 |

In A, the zero Reader output projection blocks the upstream state/Writer
gradient exactly, while that output projection can learn. This is expected;
production initialization was not changed. G uses a tiny CPU-only SGD step
with learning rate 0.01; Writer parameters stay exactly unchanged after the
first step, and the opened Reader transmits credit on the next pass.

Across A-G, original previous-state and context tensors receive `grad=None`.
In F, `autograd.grad` separately checks `S1`, `S2`, `S0`, and both context
tensors using the same Writer twice: only `S2` has a nonzero derivative.
After backward, `S1.grad` remains None. This identifies the credited transition
independently of shared parameter gradients. In E, all Writer parameter gradients
and the attached pre-detach state gradient are None.

## Pipeline integration and regressions

The real training-rollout source executes four three-frame chunks and four
denoising calls per chunk with tiny synthetic blocks. The integration probe
executes the exact existing `DMD.compute_distribution_matching_loss` source
after the full rollout, replacing only score-gradient estimation with a
deterministic synthetic value. It does not change the repository DMD code or
validate a real teacher/critic.

With checkpointing both disabled and enabled:

- Writer gradient norm: 0.037973563.
- Reader gradient norm: 0.117641824.
- Per-transition state gradient norms:
  `[0.000897272, 0.001383476, 0.001898245, 0]`.
- Four Writer updates occur, including after backward/recomputation.
- The terminal state has no consumer and has `grad=None`.
- Context-refresh forwards remain graph-free; the subsequent critic rollout
  receives fresh zero state and remains entirely graph-free.

All 22 B1/B1.1 cases passed. The former B1 checkpoint test's expectation that
Writer output lacks gradients was updated to B2's attached-output contract;
its no-extra-update and checkpoint-recompute assertions remain. Regressions
preserve legacy disabled calls, return contracts, zero Reader contribution,
FP32/BF16 max error and MSE exactly zero, state shapes/reset, update counts,
checkpoint loading/registration, rank-identical initialization, KV write records,
and caller RNG. State-token pairwise distances remain unchanged:
FP32 mean/minimum 25.794794 / 21.042202; BF16 25.794088 / 21.037291.

Run both suites without bytecode or pytest cache writes:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B -m pytest \
  -p no:cacheprovider -q -s \
  tests/test_persistent_state.py tests/test_persistent_state_b2.py
```

## Remaining runtime risks

CPU source-method probes do not qualify full Wan/FlashAttention, distributed
FSDP resharding across multiple forwards and backward, mixed-precision recurrent
gradients, optimizer/shard behavior, or peak activation memory. FSDP runtime
qualification must cover the attached state across separate enclosing forwards
and confirm Writer parameter/optimizer participation. Existing mutable-KV
checkpoint recomputation behavior and context-noise train/inference differences
remain unchanged. Existing training batch-size-above-one context-noise
broadcasting limitations also remain.

The first cold-start pass deliberately gives zero Writer gradient; later credit
depends on Reader learning to use state. Token diversity removes exact identity
symmetry but does not imply semantic specialization or empirical usefulness.

Recommendation: PASS to begin a separately authorized runtime qualification.
This phase does not authorize or start GPU qualification or real training.
