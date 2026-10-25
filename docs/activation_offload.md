# Matched smoke-test saved-tensor CPU offload

The full DMD generator rollout retains seven chunk graphs. Context refresh
overwrites mutable video KV buffers, so grad-enabled attention reads need
immutable K/V snapshots. Those snapshots alone add about 21.69 GiB per rank
for the Local18 / 21-frame / 1.3B configuration.

Two opt-in generator `model_kwargs` are available (both default `false`):

- `cached_attention_cpu_offload`: save cached self-attention's backward tensors
  on CPU using installed PyTorch `torch.autograd.graph.save_on_cpu`.
- `cached_block_cpu_offload`: extend the scope to the ordinary cached DiT block
  call, including normalization, projections, cross-attention and FFN backward
  tensors. Reader/Writer placement and recurrence stay outside this scope.

The hook copies saved values to CPU and restores the original dtype/device
when backward requests them. CUDA inputs use pinned host memory. It performs
no forward recomputation, no numerical dtype conversion, no gradient detach,
and no change to attention math, cache ownership/writes, window bounds, FIFO,
scheduler, State recurrence, or DMD. Immutable grad-read snapshots are retained.
No-grad/inference calls do not enter saved-tensor offload.

Both baseline and SSF must use the identical memory policy and distributed
topology. Generator gradient checkpointing must remain disabled: this does
not repair the known checkpoint cache side effects.

Qualification evidence is stored in
`/data/wangyuqi_intern26/ssf_runs/goal_5s_20261009_123839/resume_offload_20261009_151314`:

- 51 CPU regressions passed: all previous State/cache tests plus FP32/BF16
  exact attention outputs/gradients and actual cached-loop recurrent gradient
  equality with offload disabled/enabled.
- Real BF16 FlashAttention paired microtest: output and Q/K/V gradients were
  bit-exact with offload disabled/enabled.
- Both attention-scope and block-scope dual-H20 FSDP recurrence/backward gates
  passed, including caller cache identity, projection reuse, detach boundaries,
  exact zero Reader invariance, and independent gradient synchronization checks.
- Attention-only offload still OOMed in the real three-H20 DMD stack after all
  seven generator chunks, at fake-score FSDP's first-use full-precision buffer.
- Block-scope full DMD qualification is being measured separately; small
  FSDP success alone does not establish full-DMD or optimizer-state viability.

Costs/limits: host RAM and PCIe transfers increase; backward may be slower.
Do not describe this as a quality improvement or a completed training gate
until the actual full-DMD result and matched smoke artifacts exist.
