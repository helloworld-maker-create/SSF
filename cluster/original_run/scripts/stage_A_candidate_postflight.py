"""CPU-only postflight for the already launched, specifically identified job.

Never waits for GPU availability, launches GPU work, retries, or starts
formal training. It verifies terminal handles and writes cost/storage review.
"""
from g_common import *
import sys,traceback


def identity(pid):
    try:
        proc=Path('/proc')/str(pid)
        fields=(proc/'stat').read_text().split(') ',1)[1].split()
        return dict(start=fields[19],state=fields[0],command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    approved_stage_A();check_source()
    launch=read(ROOT/'state/launch_G1_real_DMD_sync_candidate.json')
    pid=launch['pid'];initial=identity(pid)
    assert initial and initial['state']!='Z' and str(ROOT/'scripts/qualify_dmd_accumulation_sync_candidate.py') in initial['command']
    summary=ROOT/'scripts/stage_A_measured_summary.py';expected=sha(summary)
    dump(ROOT/'state/candidate_postflight_started.json',dict(pid=os.getpid(),target_pid=pid,target_identity=initial,summary_script_SHA256=expected,started_at=time.time(),CPU_only=True,does_not_launch_GPU_or_formal_training=True))
    while True:
        live=identity(pid)
        if not live or live['state']=='Z' or live['start']!=initial['start']:break
        assert str(ROOT/'scripts/qualify_dmd_accumulation_sync_candidate.py') in live['command']
        budget_guard('stage_A_qualification_GPUh',48)
        time.sleep(30)
    qpath=ROOT/'qualification/real_dmd_accumulation/complete.json'
    if not qpath.exists():
        dump(ROOT/'state/stage_A_candidate_terminal_failure.json',dict(at=time.time(),status='QUALIFICATION_CANDIDATE_FAILED',target_pid=pid,formal_training_started=False,no_automatic_retry=True,inspect_log=launch['log']))
        return
    assert read(qpath)['status']=='PASS_G1_B64_SYNC_CANDIDATE_NATIVE_DMD_EMA_RESUME'
    for rank in range(2):
        finished=read(ROOT/'state'/('qualification_real_attempt01_rank'+str(rank)+'_finished.json'))
        assert finished['success'],'WORKER_CLEANUP_OR_TERMINAL_FAILURE'
    assert sha(summary)==expected,'POSTFLIGHT_SUMMARY_SOURCE_CHANGED'
    with (ROOT/'logs/stage_A_measured_summary_v1.log').open('xb') as f:
        subprocess.run(['/data/tuyijing/anaconda3/envs/self/bin/python','-B',str(summary)],env=dict(os.environ,PYTHONNOUSERSITE='1'),cwd=ROOT/'scripts',stdout=f,stderr=subprocess.STDOUT,check=True)
    for cmd,name in [(['git','status','--porcelain=v1'],'git_status_after_stage_A.txt'),(['git','diff','--binary','HEAD'],'git_diff_after_stage_A.patch')]:
        result=subprocess.check_output(cmd,cwd=PRODUCTION)
        with (ROOT/'source_manifest'/name).open('xb') as f:f.write(result)
    dump(ROOT/'state/candidate_postflight_complete.json',dict(at=time.time(),status='MEASURED_REVIEW_WRITTEN_FORMAL_APPROVAL_PENDING',summary=str(ROOT/'state/stage_A_measured_summary_v1.json'),formal_training_started=False,user_health_scores=None,all_rank_success_and_process_group_cleanup_confirmed=True))


if __name__=='__main__':
    try:main()
    except BaseException:
        append(ROOT/'logs/stage_A_postflight_failures.jsonl',dict(at=time.time(),traceback=traceback.format_exc(),no_GPU_launch=True));raise
