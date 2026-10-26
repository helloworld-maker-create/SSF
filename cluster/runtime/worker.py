"""torchrun worker; append stdout/stderr to a separate log for each rank."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['qualify', 'train'], required=True)
    ap.add_argument('--arm', choices=['baseline', 'ssf'], required=True)
    ap.add_argument('--resume')
    ap.add_argument('--attempt', type=int, default=0)
    args = ap.parse_args()
    root = Path(os.environ['SSF_RUN_ROOT'])
    rank = int(os.environ['RANK'])
    lease = None
    if rank == 0:
        lease = (root / 'run.lock').open('a')
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = root / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    # Redirect descriptors too: native CUDA/NCCL diagnostics reach the same log.
    with (folder / f'{args.mode}_{args.arm}_rank{rank}.log').open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            sys.argv = [sys.argv[0], '--arm', args.arm]
            if args.resume:
                sys.argv += ['--resume', args.resume]
            sys.argv += ['--attempt', str(args.attempt)]
            if args.mode == 'qualify':
                from qualify_dmd_accumulation_sync_candidate import main as run
                sys.argv = [sys.argv[0]]
            else:
                from phaseg_train import main as run
            run()
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
