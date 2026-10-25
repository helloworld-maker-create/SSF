"""Read-only observation of the existing Baseline PID, never a GPU launcher."""
from g_common import *
import argparse


def process_identity(pid):
    proc=Path('/proc')/str(pid)
    try:
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    if 'phaseg_train.py' not in command or '--arm baseline' not in command or fields[0]=='Z':return None
    return fields[19]


def last_record(path):
    if not path.exists():return None
    try:return json.loads(subprocess.check_output(['tail','-n','1',str(path)],text=True))
    except json.JSONDecodeError:return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--until-g',type=int,required=True)
    p.add_argument('--until-c',type=int,required=True)
    p.add_argument('--max-minutes',type=int,default=90)
    a=p.parse_args();assert 0<a.max_minutes<=120
    launch=read(ROOT/'state/launch_formal_baseline_attempt00.json')
    pids=[3169256,3169257];identities=[process_identity(pid) for pid in pids]
    assert all(identities),'VERIFY_REAL_TRAINING_WORKERS_FIRST'
    stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    log=ROOT/'resource_snapshots'/('baseline_observation_'+stamp+'.jsonl')
    dump(ROOT/'state'/('baseline_observer_'+stamp+'.json'),dict(pid=os.getpid(),at=time.time(),
        watched_PIDs=pids,watched_start_times=identities,until_G=a.until_g,until_C=a.until_c,
        max_minutes=a.max_minutes,log=str(log),code_SHA256=sha(__file__),
        scope='read-only; no model/RNG/optimizer/launch/source modification'))
    deadline=time.time()+60*a.max_minutes
    while True:
        ranks=[];alive=True
        for rank,pid in enumerate(pids):
            live=process_identity(pid)==identities[rank];alive=alive and live
            folder=ROOT/f'training/baseline/attempt_00/rank{rank}'
            u=last_record(folder/'updates.jsonl');m=last_record(folder/'micro_forward.jsonl')
            ranks.append(dict(rank=rank,live=live,
                accepted=None if u is None else u['counts_after'],
                last_loss=None if u is None else u['loss_global_mean'],
                last_grad_norm=None if u is None else u['preclip_global_norm'],
                micro=None if m is None else {k:m[k] for k in ['kind','outer','micro','loss']}))
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        stat=os.statvfs(ROOT)
        ready=all(r['accepted'] is not None and r['accepted']['G']>=a.until_g
                  and r['accepted']['C']>=a.until_c for r in ranks)
        status='OBSERVED_TARGET_ACCEPTED_COUNTS' if ready else 'LIVE_TRAINING' if alive else 'REAL_WORKER_HANDLE_MISSING'
        row=dict(at=datetime.datetime.now().astimezone().isoformat(),observer_PID=os.getpid(),status=status,ranks=ranks,
            host_MemAvailable_bytes=int(mem['MemAvailable'].split()[0])*1024,
            filesystem_available_bytes=stat.f_bavail*stat.f_frsize,
            budget_training_GPUh=budget_used('stage_B_training_GPUh'),
            no_GPU_job_or_optimizer_operation_by_observer=True)
        append(log,row);print(json.dumps(row),flush=True)
        if ready or not alive:return
        if time.time()>=deadline:
            print(json.dumps(dict(status='OBSERVATION_WINDOW_ENDED_REAL_TRAINING_NOT_STOPPED',at=time.time())),flush=True);return
        time.sleep(55)


if __name__=='__main__':main()
