"""Observe the already-requested recoverable stop; never signal or launch a GPU job."""
from g_common import *


def live_worker(pid):
    proc = Path('/proc') / str(pid)
    try:
        command = (proc / 'cmdline').read_bytes()
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (fields[0] != 'Z' and fields[19] == '326953642'
            and b'phaseg_train.py' in command and b'--arm\x00baseline' in command)


def last(path):
    if not path.exists():
        return None
    try:
        return json.loads(subprocess.check_output(['tail', '-n', '1', str(path)], text=True))
    except json.JSONDecodeError:
        return None


def main():
    start = datetime.datetime.now().astimezone()
    tag = start.strftime('%Y%m%d_%H%M%S')
    output = ROOT / 'resource_snapshots' / ('safe_stop_observation_' + tag + '.jsonl')
    dump(ROOT / 'state' / ('safe_stop_observer_' + tag + '.json'),
         dict(pid=os.getpid(), at=start.isoformat(), log=str(output),
              workers=[3169256, 3169257], maximum_minutes=90,
              scope='CPU read-only, no model, signal, restart, or optimizer operation'))
    attempt = ROOT / 'training/baseline/attempt_00'
    checkpoints = ROOT / 'training/baseline/checkpoints'
    deadline = time.time() + 90 * 60
    while True:
        workers = []
        for rank, pid in enumerate([3169256, 3169257]):
            folder = attempt / f'rank{rank}'
            update = last(folder / 'updates.jsonl')
            micro = last(folder / 'micro_forward.jsonl')
            finished = folder / 'worker_finished.json'
            workers.append(dict(rank=rank, pid=pid, live=live_worker(pid),
                accepted=None if update is None else update['counts_after'],
                micro=None if micro is None else {k: micro[k] for k in ['kind', 'outer', 'micro']},
                finished=read(finished) if finished.exists() else None))
        markers = sorted(checkpoints.glob('emergency_*.COMMITTED.json'))
        marker = read(markers[-1]) if markers else None
        pending = [dict(path=str(p), bytes=p.stat().st_size)
                   for p in checkpoints.glob('emergency_*.pt')]
        stopped_path = attempt / 'STOPPED_RECOVERABLE.json'
        stopped = read(stopped_path) if stopped_path.exists() else None
        any_live = any(w['live'] for w in workers)
        cleanup = all(w['finished'] is not None and w['finished']['process_group_cleanup_complete']
                      for w in workers)
        complete = (not any_live and cleanup and stopped is not None and marker is not None
                    and marker['status'] == 'COMMITTED' and marker['readback_all_components_exact'])
        status = ('RECOVERABLE_STOP_COMPLETE' if complete else
                  'COMMITTED_CLEANUP_PENDING' if marker is not None else
                  'CHECKPOINT_WRITE_OR_VALIDATION' if pending else
                  'FINISHING_CURRENT_OUTER' if any_live else 'WORKERS_EXITED_CHECK_STOP_EVIDENCE')
        fs = os.statvfs(ROOT)
        row = dict(at=datetime.datetime.now().astimezone().isoformat(), status=status,
                   workers=workers, checkpoint_marker=marker, checkpoint_files=pending,
                   stopped_record=stopped, cleanup_complete=cleanup,
                   filesystem_available_bytes=fs.f_bavail * fs.f_frsize,
                   no_training_or_signal_operation_by_observer=True)
        append(output, row)
        print(json.dumps(row), flush=True)
        if complete or not any_live:
            return
        if time.time() >= deadline:
            print(json.dumps(dict(status='OBSERVATION_WINDOW_ENDED_NOT_A_TRAINING_TERMINAL',
                                  at=time.time())), flush=True)
            return
        time.sleep(55)


if __name__ == '__main__':
    main()
