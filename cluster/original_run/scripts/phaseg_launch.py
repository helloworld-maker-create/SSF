"""Bounded launcher for ONE authorized Phase G arm. No automatic SSF launch."""
from g_common import *
import argparse,socket,sys,traceback
from phaseg_formal_control import authorized
from stage_A_runner import snapshot,env_base


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','ssf'],default='baseline');p.add_argument('--attempt',type=int,default=0);p.add_argument('--resume');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    authorized(a.arm);check_source()
    manifest=read(ROOT/'source_manifest/formal_code_manifest.json')
    for r in manifest['files']:assert sha(ROOT/'scripts'/r['name'])==r['SHA256'],'FORMAL_CODE_CHANGED'
    assert read(ROOT/'qualification/formal_control_CPU_result.json')['status']=='PASS'
    if a.resume:
        path=Path(a.resume).resolve();expected=(ROOT/'training'/a.arm/'checkpoints').resolve()
        assert path.is_relative_to(expected),'RESUME_MUST_USE_THIS_ARM_COMPLETE_CHECKPOINT'
        meta=read(path.with_suffix('.COMMITTED.json'))
        assert meta['status']=='COMMITTED' and meta['arm']==a.arm and meta['path']==str(path)
    tag=f'formal_{a.arm}_attempt{a.attempt:02d}'
    assert not (ROOT/'state'/('launch_'+tag+'.json')).exists(),'EXISTING_LAUNCH_REQUIRES_INSPECTION'
    deadline=time.time()+30*60
    while True:
        s=snapshot();append(ROOT/'resource_snapshots'/(tag+'_preflight.jsonl'),s)
        free=[g for g in s['GPUs'] if g['idle']]
        host=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        q=read(ROOT/'resource_snapshots/stage_A_postflight_readonly.json')
        # User accepted current resources; retain a measured working-set plus
        #128GiB check rather than assuming eight legacy host stacks will fit.
        host_ok=host>=q['conservative_RSS_sum_bytes']+128*1024**3
        storage_guard(38*1024**3)
        if a.preflight_only:
            print(json.dumps(dict(status='READY' if len(free)>=2 and host_ok else 'BLOCKED_RESOURCE',idle_GPUs=[g['index'] for g in free],host_available_GiB=host/2**30,requires_two_qualified_H20s=True,no_GPU_launched=True)),flush=True);return
        if len(free)>=2 and host_ok:
            selected=free[:2];again=snapshot();by={g['uuid']:g for g in again['GPUs']}
            if all(by[g['uuid']]['idle'] for g in selected):break
        assert time.time()<deadline,'BLOCKED_RESOURCE: no two idle qualified GPUs/host RAM in bounded30minute wait'
        print('RESOURCE_WAIT',json.dumps(dict(at=time.time(),idle_GPUs=[g['index'] for g in free],host_available_GiB=host/2**30)),flush=True)
        time.sleep(60)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=env_base([g['uuid'] for g in selected]);env.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    command=['/data/tuyijing/anaconda3/envs/self/bin/python','-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2','--master_addr=127.0.0.1','--master_port='+str(port),str(ROOT/'scripts/phaseg_train.py'),'--arm',a.arm,'--attempt',str(a.attempt)]
    if a.resume:command+=['--resume',str(Path(a.resume).resolve())]
    dump(ROOT/'resource_snapshots'/(tag+'_reserved.json'),dict(at=time.time(),GPUs=selected,command=command,port=port,owner_pid=os.getpid()))
    with (ROOT/'logs'/(tag+'.log')).open('xb') as log:
        child=subprocess.Popen(command,env=env,cwd=ROOT/'scripts',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'state'/('launch_'+tag+'.json'),dict(pid=child.pid,at=time.time(),command=command,GPUs=selected,log=str(ROOT/'logs'/(tag+'.log')),code_manifest_SHA256=sha(ROOT/'source_manifest/formal_code_manifest.json'),arm=a.arm,formal=True))
    result=child.wait()
    dump(ROOT/'state'/(tag+'_launcher_finished.json'),dict(at=time.time(),child_pid=child.pid,returncode=result,no_auto_other_arm=True))
    assert result==0,'FORMAL_JOB_FAILURE '+str(result)


if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/formal_launch_failures.jsonl',dict(at=time.time(),pid=os.getpid(),traceback=traceback.format_exc(),no_auto_recipe_change=True));raise
