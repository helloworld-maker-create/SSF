"""CPU-only result collector/publisher. No training or model imports on startup."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode=True
from result_snapshot import collect, check_public_bytes, js, sha, FILE_LIMIT, PACKAGE_LIMIT


def write_new(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)


def remote_name(remote):
    patterns=[r'https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?',
              r'git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?']
    for pattern in patterns:
        m=re.fullmatch(pattern,remote)
        if m and '..' not in m.groups():return '/'.join(m.groups())
    raise ValueError('Use a credential-free GitHub HTTPS or git@github.com SSH URL')


def validate_branch(branch):
    if not re.fullmatch(r'results/[A-Za-z0-9][A-Za-z0-9_-]{0,110}',branch):
        raise ValueError('Publication is restricted to results/<run_id>-<arm>; never a code branch')


def git(argv, cwd, stdin=None, allow_failure=False):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_AUTHOR_NAME='SSF Results Publisher',
             GIT_AUTHOR_EMAIL='ssf-results@localhost',GIT_COMMITTER_NAME='SSF Results Publisher',
             GIT_COMMITTER_EMAIL='ssf-results@localhost')
    try:
        input_args={'input':stdin} if stdin is not None else {'stdin':subprocess.DEVNULL}
        r=subprocess.run(['git',*argv],cwd=cwd,**input_args,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Git operation timed out; stop publishing and inspect remote state before retry') from None
    if r.returncode and not allow_failure:
        # Credential helpers may write sensitive values. Do not echo stderr,
        # command-line credentials, environment variables, or helper output.
        raise RuntimeError('Git '+argv[0]+' failed (exit %d); no automatic retry or alternate credentials'%r.returncode)
    return r.returncode,r.stdout


def snapshot(args, command_returncode=None):
    summary,files=collect(args.run_root,args.arm,args.world_size,
                          command_returncode=command_returncode,eval_root=args.eval_root)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    package_id=stamp+'_'+uuid.uuid4().hex[:8]
    folder=args.state_dir/'packages'/package_id
    folder.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        target=folder/name
        if not target.resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe result artifact path')
        write_new(target,data)
    return folder,summary


def package_files(folder):
    folder=Path(folder).resolve();manifest=json.loads((folder/'manifest.json').read_text())
    out={}
    total=0
    for item in manifest['files']:
        name=item['path'];p=folder/name
        allowed=(name in ['summary.json','training_config.json','checkpoint_manifest.json',
                          'training_completion.json','updates.csv','loss.svg','README.md']
                 or re.fullmatch(r'raw/updates_rank\d+\.jsonl\.gz',name)
                 or re.fullmatch(r'logs/rank\d+_tail\.log',name)
                 or re.fullmatch(r'provenance/(source_files|formal_code_manifest)\.json',name)
                 or re.fullmatch(r'evaluation/(summary\.json|per_video\.csv|report\.md)',name))
        if not allowed:raise ValueError('Unapproved result artifact: '+Path(name).name)
        if not p.resolve().is_relative_to(folder) or p.resolve()!=p.absolute() or not p.is_file():
            raise ValueError('Unsafe snapshot artifact')
        if p.stat().st_size>FILE_LIMIT:raise ValueError('Result file exceeded limit')
        data=p.read_bytes();total+=len(data)
        if len(data)!=item['bytes'] or sha(data)!=item['sha256']:raise ValueError('Snapshot content changed')
        if name.endswith('.gz'):
            import gzip
            with gzip.GzipFile(fileobj=__import__('io').BytesIO(data)) as f:plain=f.read(FILE_LIMIT+1)
            check_public_bytes(name,plain)
        else:check_public_bytes(name,data)
        out[name]=data
    if total>PACKAGE_LIMIT:raise ValueError('Result package exceeded limit')
    out['manifest.json']=(folder/'manifest.json').read_bytes()
    check_public_bytes('manifest.json',out['manifest.json'])
    return manifest,out


def publish(folder, remote, branch, state_dir, runner=git):
    repository=remote_name(remote);validate_branch(branch)
    manifest,files=package_files(folder)
    expected_branch='results/'+manifest['run_id']+'-'+manifest['arm']
    if branch!=expected_branch:raise ValueError('Branch must identify this exact run and arm')
    workspace=Path(state_dir)/'transport'/(Path(folder).name+'_'+uuid.uuid4().hex[:8])
    workspace.mkdir(parents=True,exist_ok=False)
    marker=dict(schema_version=1,kind='SSF_RESULT_BRANCH',repository=repository,
                run_id=manifest['run_id'],arm=manifest['arm'])
    runner(['init','--quiet'],workspace)
    ref='refs/heads/'+branch
    _,out=runner(['ls-remote','--heads',remote,ref],workspace)
    lines=out.decode().strip().splitlines()
    if len(lines)>1:raise ValueError('Ambiguous remote result ref')
    parent=lines[0].split()[0] if lines else None
    prefix='experiments/'+manifest['run_id']+'/snapshots/'+Path(folder).name
    if parent:
        if not re.fullmatch(r'[0-9a-f]{40}',parent):raise ValueError('Invalid remote commit SHA')
        runner(['fetch','--quiet','--depth=1','--no-tags',remote,ref],workspace)
        _,fetched=runner(['rev-parse','FETCH_HEAD'],workspace)
        if fetched.decode().strip()!=parent:raise RuntimeError('Remote advanced during fetch; stop, never force push')
        _,body=runner(['show',parent+':RESULT_BRANCH.json'],workspace)
        if json.loads(body)!=marker:raise ValueError('Existing result branch ownership marker differs')
        code,previous=runner(['show',parent+':'+prefix+'/manifest.json'],workspace,allow_failure=True)
        if code==0:
            if previous!=files['manifest.json']:raise ValueError('Immutable remote snapshot differs')
            return dict(status='ALREADY_PUBLISHED',branch=branch,commit=parent,snapshot=prefix)
        # This index belongs to a newly-created publisher repository. No user
        # checkout, source index or historical result is reset or discarded.
        runner(['read-tree',parent+'^{tree}'],workspace)
    payload={prefix+'/'+k:v for k,v in files.items()}
    payload['RESULT_BRANCH.json']=js(marker)
    payload['latest.json']=js(dict(run_id=manifest['run_id'],arm=manifest['arm'],snapshot=prefix,
                                 manifest_sha256=sha(files['manifest.json'])))
    payload['README.md']=('# Automatic SSF results\n\n'
        'Code branches are unchanged. This branch contains immutable result snapshots.\n\n'
        '[Latest snapshot]('+prefix+'/README.md) · [latest.json](latest.json)\n\n'
        'A running snapshot is not final training, health review, or quality evidence.\n').encode()
    for name,data in sorted(payload.items()):
        if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe git artifact path')
        _,blob=runner(['hash-object','-w','--stdin'],workspace,stdin=data)
        blob=blob.decode().strip()
        if not re.fullmatch(r'[0-9a-f]{40}',blob):raise ValueError('Invalid Git blob identity')
        runner(['update-index','--add','--cacheinfo','100644',blob,name],workspace)
    _,tree=runner(['write-tree'],workspace)
    args=['commit-tree',tree.decode().strip()]
    if parent:args+=['-p',parent]
    args+=['-m','Publish '+manifest['run_id']+' '+manifest['arm']+' '+Path(folder).name]
    _,commit=runner(args,workspace);commit=commit.decode().strip()
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Invalid result commit')
    # Ordinary fast-forward push only. A concurrent advance rejects this push;
    # there is deliberately no retry, force flag, branch reset or credential fallback.
    runner(['push','--porcelain',remote,commit+':'+ref],workspace)
    _,verify=runner(['ls-remote','--heads',remote,ref],workspace)
    if verify.decode().strip().split()[0]!=commit:
        raise RuntimeError('Remote head differs after push; preserve package and inspect before retry')
    return dict(status='PUBLISHED',repository=repository,branch=branch,commit=commit,
                tree=tree.decode().strip(),snapshot=prefix,
                url='https://github.com/'+repository+'/blob/'+branch+'/'+prefix+'/README.md')


def record_event(args, event):
    with (args.state_dir/'events.jsonl').open('ab') as f:
        f.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**event),ensure_ascii=False).encode()+b'\n')
    print(json.dumps(event,ensure_ascii=False),flush=True)


def terminal(root, arm, world):
    armroot=Path(root)/'training'/arm
    if (armroot/'TRAINING_AND_GENERATION_COMPLETE.json').is_file():return True
    attempts=sorted([p for p in armroot.glob('attempt_*') if re.fullmatch(r'attempt_\d+',p.name)],
                    key=lambda p:int(p.name.split('_')[1]))
    return bool(attempts and all((attempts[-1]/('rank%d'%rank)/'worker_finished.json').is_file() for rank in range(world)))


def automate(args, child=None):
    upload_enabled=True;upload_failed=False;next_at=0.;last_key=None
    while True:
        rc=None if child is None else child.poll()
        done=(child is not None and rc is not None) or (child is None and terminal(args.run_root,args.arm,args.world_size))
        if time.monotonic()>=next_at or done:
            if args.run_root.exists() or done:
                try:
                    folder,summary=snapshot(args,command_returncode=rc)
                    key=json.dumps({k:summary[k] for k in ['counts','status','full_checkpoint_markers','evaluation_status','workers_finished','command_returncode']},sort_keys=True)
                    if key!=last_key or done:
                        record_event(args,dict(status='LOCAL_SNAPSHOT',folder=str(folder),observed=summary['status'],counts=summary['counts']))
                        if upload_enabled:
                            try:
                                result=publish(folder,args.remote,args.branch,args.state_dir)
                                record_event(args,result)
                            except Exception as exc:
                                upload_enabled=False;upload_failed=True
                                record_event(args,dict(status='UPLOAD_BLOCKED_NO_AUTOMATIC_RETRY',error=str(exc),package=str(folder),training_untouched=True))
                        last_key=key
                except Exception as exc:
                    upload_enabled=False;upload_failed=True
                    record_event(args,dict(status='SNAPSHOT_INTEGRITY_OR_COLLECTION_FAILURE',error=str(exc),training_untouched=True))
            next_at=time.monotonic()+args.interval_seconds
        if done:
            return rc if rc not in [None,0] else (3 if upload_failed else 0)
        time.sleep(1)


def main():
    argv=sys.argv[1:]
    command=[]
    if '--' in argv:
        cut=argv.index('--');command=argv[cut+1:];argv=argv[:cut]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['snapshot','publish','watch','run'])
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--arm',choices=['baseline','ssf'],required=True)
    ap.add_argument('--world-size',type=int,required=True)
    ap.add_argument('--state-dir',type=Path)
    ap.add_argument('--eval-root',type=Path)
    ap.add_argument('--remote',default='git@github.com:helloworld-maker-create/SSF.git')
    ap.add_argument('--branch')
    ap.add_argument('--interval-seconds',type=int,default=1800)
    args=ap.parse_args(argv);args.command=command
    if not args.run_root.expanduser().is_absolute():ap.error('Use an absolute experiment directory')
    args.run_root=args.run_root.expanduser().resolve()
    if not args.run_root.is_absolute() or args.run_root==Path('/'):
        ap.error('Specify an explicit experiment directory')
    args.state_dir=(args.state_dir or args.run_root.parent/'.ssf_result_uploads'/args.run_root.name/args.arm).resolve()
    if args.state_dir==args.run_root or args.state_dir.is_relative_to(args.run_root):
        ap.error('Keep uploader state outside the training run directory')
    args.branch=args.branch or 'results/'+args.run_root.name+'-'+args.arm
    remote_name(args.remote);validate_branch(args.branch)
    if args.interval_seconds<60:ap.error('Publication interval must be >=60 seconds')
    if args.world_size<1 or 64%args.world_size:ap.error('world-size must divide B64')
    args.state_dir.mkdir(parents=True,exist_ok=True)
    # Serialize this run/arm's publishers only; never lock or pause training.
    with (args.state_dir/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode in ['snapshot','publish']:
            folder,summary=snapshot(args)
            record_event(args,dict(status='LOCAL_SNAPSHOT',folder=str(folder),counts=summary['counts']))
            if args.mode=='publish':record_event(args,publish(folder,args.remote,args.branch,args.state_dir))
            return 0
        if args.mode=='watch':return automate(args)
        command=args.command
        if not command:ap.error('run requires -- followed by the existing experiment command')
        # This wrapper does not allocate a GPU, select devices, change the
        # training recipe, signal processes or perform optimizer operations.
        # The existing launcher/Slurm job keeps its own lifecycle and safeguards.
        child=subprocess.Popen(command,env=dict(os.environ,PYTHONNOUSERSITE='1'))
        try:return automate(args,child)
        except KeyboardInterrupt:
            record_event(args,dict(status='PUBLISHER_INTERRUPTED',child_may_still_be_running=True,
                                   no_process_killed=True))
            return 130


if __name__=='__main__':
    sys.exit(main())
