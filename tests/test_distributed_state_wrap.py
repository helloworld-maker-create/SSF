"""Writer-only FSDP boundary; all legacy wrapping decisions stay identical."""
from functools import partial

from torch import nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from utils.distributed import _state_writer_auto_wrap_policy
from utils.persistent_state import StateWriter


def test_writer_wraps_below_size_threshold_and_recurses():
    writer = StateWriter(32, 16, 4, num_state_tokens=8)
    base = partial(size_based_auto_wrap_policy, min_num_params=50_000_000)
    for recurse in (True, False):
        assert _state_writer_auto_wrap_policy(
            writer, recurse, sum(p.numel() for p in writer.parameters()), base)


def test_other_modules_keep_original_policy_decisions():
    base = partial(size_based_auto_wrap_policy, min_num_params=100)
    for module in (nn.Linear(4, 4), nn.Sequential(nn.Linear(16, 16)), nn.LayerNorm(16)):
        for recurse in (True, False):
            for count in (0, 50, 100, 256):
                assert _state_writer_auto_wrap_policy(module, recurse, count, base) == base(
                    module=module, recurse=recurse, nonwrapped_numel=count)
