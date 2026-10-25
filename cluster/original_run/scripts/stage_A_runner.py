"""Bounded authorized Stage A job chain. NEVER starts Baseline or SSF training.

Waits for specifically identified live reference processes, then runs the
registered scoring and new-infrastructure qualifications. Resource waits after
dependencies finish are capped at30 minutes. No scientific retries or cleanup.
"""
from g_common import *
import argparse,faulthandler,importlib.util,socket,sys,traceback
faulthandler.enable(all_threads=True)
PY='/data/tuyijing/anaconda3/envs/self/bin/python'
METRIC_PY='/data/tuyijing/anaconda3/envs/vbench2/bin/python'

def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.splitlines():
        i,u,n,v,f,t=[x.strip() for x in line.split(',')]
        live=[x for x in procs.splitlines() if u in x]
        rows.append(dict(index=int(i),uuid=u,name=n,util=float(v),free_MiB=int(f),total_MiB=int(t),processes=live,
            idle=n=='NVIDIA H20-3e' and float(v)<=5 and int(f)>=120*1024 and not live))
    return dict(at=time.time(),GPUs=rows,processes=procs,filesystem_available=storage_guard())

def wait_idle(n,label,deadline=None):
    end=deadline or time.time()+30*60
    while True:
        budget_guard('stage_A_reference_GPUh',8);budget_guard('stage_A_qualification_GPUh',48)
        s=snapshot();idle=[g for g in s['GPUs'] if g['idle']]
        append(ROOT/'resource_snapshots/stage_A_resources.jsonl',dict(label=label,**s))
        if len(idle)>=n:
            # Recheck immediately before launch; never trust an old snapshot.
            again=snapshot();by={g['uuid']:g for g in again['GPUs']}
            chosen=idle[:n]
            if all(by[g['uuid']]['idle'] for g in chosen):return chosen
        assert time.time()<end,'BLOCKED_RESOURCE after bounded30minute retry '+label
        time.sleep(min(60,max(1,end-time.time())))

def wait_reference_processes(arms=('released_DMD','ODE')):
    expected={'released_DMD':32,'ODE':104}
    while True:
        missing=[]
        for arm,count in expected.items():
            if arm not in arms:continue
            complete=ROOT/'reference'/(arm+'_complete.json')
            if complete.exists():
                x=read(complete);assert x['status']=='PASS_TECHNICAL_REFERENCE' and x['count']==count
                continue
            candidates=sorted((ROOT/'state').glob('launch_'+arm+'*.json'))
            alive=[]
            for p in candidates:
                launch=read(p);pid=launch['pid'];proc=Path('/proc')/str(pid)
                try:
                    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                    status=(proc/'stat').read_text().split()[2]
                except FileNotFoundError:continue
                if status!='Z' and str(ROOT/'scripts/reference_worker.py') in cmd and '--arm '+arm in cmd:
                    alive.append(dict(pid=pid,cmd=cmd))
            assert alive,'REFERENCE_TERMINAL_WITHOUT_COMPLETE '+arm
            missing.append(dict(arm=arm,live_handles=alive))
        if not missing:return
        budget_guard('stage_A_reference_GPUh',8);storage_guard()
        append(ROOT/'logs/stage_A_dependency_wait.jsonl',dict(at=time.time(),waiting_for_specific_live_reference_handles=missing))
        time.sleep(30)

def verify_references():
    rows=matrix();records={}
    for row in rows:
        p=ROOT/'reference/records'/(row['video_id']+'.json');rec=read(p)
        assert rec['matrix_row']==row and rec['finite'] and rec['video']['lossless']
        assert sha(rec['video']['path'])==rec['video']['SHA256']
        records[row['video_id']]=rec
    assert len(records)==136
    native={r['prompt']:r for r in records.values() if r['arm']=='released_DMD' and r['matrix_row']['local_attn_size']=='-1'}
    local={r['prompt']:r for r in records.values() if r['matrix_row']['purpose']=='reference_detailed_Local18'}
    comparisons=[]
    for prompt in native:
        a,b=native[prompt],local[prompt]
        assert a['initial_noise_SHA256']==b['initial_noise_SHA256'] and a['renoise_SHA256']==b['renoise_SHA256']
        assert a['raw_flow_SHA256'][:30]==b['raw_flow_SHA256'][:30],'UNEXPECTED_PRE_EVICTION_PREFIX_DIFFERENCE'
        comparisons.append(dict(prompt_id=a['prompt_id'],same_initial_and_renoise=True,first6_chunks_bit_exact=True,
            final_chunk_equal=a['raw_flow_SHA256'][30:]==b['raw_flow_SHA256'][30:]))
    dump(ROOT/'reference/generation_audit.json',dict(status='PASS136_LOSSLESS_NATIVE_INFERENCE_REFERENCE',videos=136,
        released_EMA_SHA256=read(ROOT/'reference/released_DMD_input_checkpoint.json')['SHA256'],
        ODE_SHA256=read(ROOT/'reference/ODE_input_checkpoint.json')['SHA256'],Local18_comparisons=comparisons,
        no_paper_score_reproduction_claim=True,health_or_MVP_PASS_not_inferred=True))

def env_base(uuids):
    return dict(os.environ,PYTHONNOUSERSITE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(uuids),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

def spawn(name,command,env,gpus):
    for p in (ROOT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
    log=ROOT/'logs'/(name+'.log')
    # Reservation records exist immediately at spawn, before CUDA contexts.
    dump(ROOT/'resource_snapshots'/(name+'_reserved.json'),dict(at=time.time(),owner_pid=os.getpid(),GPUs=gpus,
        command=command,MASTER_PORT=env.get('MASTER_PORT'),new_stage_A_only=True))
    with log.open('xb') as f:child=subprocess.Popen(command,env=env,cwd=ROOT/'scripts',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'state'/('launch_'+name+'.json'),dict(pid=child.pid,command=command,log=str(log),GPUs=gpus,
        scripts_SHA256={p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},started_at=time.time(),formal_training=False))
    return child

def wait_children(children):
    while any(p.poll() is None for p in children):
        append(ROOT/'logs/stage_A_child_wait.jsonl',dict(at=time.time(),live_child_handles=[dict(pid=p.pid,poll=p.poll()) for p in children]))
        time.sleep(15)
    for child in children:assert child.returncode==0,'CHILD_FAILURE '+str(child.pid)+' exit'+str(child.returncode)

def score_references(arms=('released_DMD','ODE'),max_workers=2):
    # Existing proven distinct-world1-rendezvous utility, no metric changes.
    helper=ROOT.parent/'phaseF_reader_credit_20261023_143148/scripts/evaluation_rendezvous.py'
    spec=importlib.util.spec_from_file_location('_existing_eval_rendezvous',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tasks=[(arm,seed) for arm in arms for seed in [2026,2027]]
    for start in range(0,len(tasks),max_workers):
        batch=tasks[start:start+max_workers]
        selected=wait_idle(len(batch),'reference_metrics_batch'+str(start)+'_'+','.join(arms));held=[];children=[]
        for (arm,seed),gpu in zip(batch,selected):
            env,port=module.isolated_world1_env(env_base([gpu['uuid']]),held);held.append(port)
            env['PYTHONPATH']='/data/wangyuqi_intern26/repos/Self_Gradient_Forcing/d2_deps'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            cmd=[METRIC_PY,'-B','-X','faulthandler','-u',str(ROOT/'scripts/reference_metrics.py'),'--arm',arm,'--seed',str(seed)]
            children.append(spawn('reference_metrics_'+arm+'_'+str(seed),cmd,env,[gpu]))
        wait_children(children)

def new_qualifications():
    selected=wait_idle(2,'G1_toy')
    uuids=[g['uuid'] for g in selected]
    for name,worker in [('G1_toy','qualify_fsdp_toy.py'),('G1_real_DMD','qualify_dmd_accumulation.py')]:
        # Keep the same qualified2-GPU physical topology through toy and real gate.
        deadline=time.time()+30*60
        while True:
            s=snapshot();by={g['uuid']:g for g in s['GPUs']}
            if all(by[u]['idle'] for u in uuids):break
            assert time.time()<deadline,'BLOCKED_RESOURCE fixed_G1_topology'
            time.sleep(60)
        if name=='G1_real_DMD':
            assert read(ROOT/'qualification/fsdp_toy/complete.json')['status']=='PASS_FSDP64_MEAN_FINAL_SYNC_ROOT_CLIP_ADAMW'
            storage_guard(40*1024**3)
            avail=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            assert 1.25*(2*300*1024**3)+128*1024**3<=avail,'BLOCKED_HOST_PROJECTED_INITIALIZATION'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=env_base(uuids);env.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
        cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2','--master_addr=127.0.0.1','--master_port='+str(port),str(ROOT/'scripts'/worker)]
        child=spawn(name,cmd,env,selected);wait_children([child])

def stage_A_report():
    q=read(ROOT/'qualification/real_dmd_accumulation/complete.json')
    routine=q['qualification_snapshot_bytes']/1024**3
    vbytes=sum(p.stat().st_size for p in (ROOT/'reference/videos').glob('*.mp4'))
    available=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize/1024**3
    cost=q['nominal_steady_formal_GPUh_per_arm']
    # Actual projection, not permission to consume the future budget. No old
    # artifact cleanup and no change to the accepted scientific configuration.
    projection=dict(measured_qualification_full_snapshot_GiB=routine,current_available_GiB=available,
        remaining_B_routine3_GiB=3*routine,remaining_B_emergency_plus_pending2_GiB=2*routine,
        remaining_B_video_allowance_GiB=12,operational_remaining_floor_GiB=32,
        nominal_formal_GPUh_per_arm=cost,save_monitor_overhead_not_yet_in_cost=True,
        B_approval_still_required=True,complete_two_arm_storage_not_assumed=True)
    status='AWAITING_STAGE_B_APPROVAL_AND_MEASURED_FINAL_COST'
    if cost>500:status='BLOCKED_BUDGET'
    elif available<5*routine+12+32:status='BLOCKED_STORAGE_STAGE_B_FORECAST'
    report=dict(status=status,stage_A_qualifications=q,projection=projection,
        reference_generation_audit=str(ROOT/'reference/generation_audit.json'),formal_training_started=False,
        stage_B_started=False,stage_C_started=False,user_health_scores=None)
    dump(ROOT/'state/stage_A_complete.json',report)
    print(json.dumps(report,indent=2),flush=True)

def main():
    approved_stage_A();check_source()
    lock=(ROOT/'state/stage_A_runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'state/stage_A_runner_started.json',dict(pid=os.getpid(),at=time.time(),stage='A only',no_formal_training=True))
    # Use the released-reference GPU for scoring while the longer ODE job
    # continues. At most2 GPU workers total, no shared/occupied device.
    wait_reference_processes(('released_DMD',));score_references(('released_DMD',),max_workers=1)
    wait_reference_processes(('ODE',));verify_references();score_references(('ODE',),max_workers=2)
    new_qualifications();stage_A_report()

if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/stage_A_runner_failures.jsonl',dict(at=time.time(),pid=os.getpid(),traceback=traceback.format_exc(),automatic_scientific_retry=False));raise
