"""Read-only, bounded Phase G result snapshots. No model, CUDA or weight reads."""
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

FILE_LIMIT = 8 * 1024**2
PACKAGE_LIMIT = 32 * 1024**2
SECRET_PATTERNS = [
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
    re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}'),
    re.compile(rb'AKIA[0-9A-Z]{16}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'(?im)(?:authorization[\"\x27]?\s*:\s*[\"\x27]?(?:bearer|basic)\s+|(?:api[_-]?key|access[_-]?token|github[_-]?token|password|secret|token)[\"\x27]?\s*[=:]\s*[\"\x27]?)[A-Za-z0-9_+/=.-]{12,}'),
    re.compile(rb'https?://[^\s/:]+:[^\s/@]+@'),
]
CONFIG_KEYS = {'independent_first_frame','warp_denoising_step','weight_decay','same_step_across_blocks',
    'last_step_only','num_training_frames','context_noise','causal','real_name','denoising_step_list',
    'num_train_timestep','timestep_shift','guidance_scale','denoising_loss_type','mixed_precision',
    'seed','sharding_strategy','lr','lr_critic','beta1','beta2','beta1_critic','beta2_critic','batch_size',
    'ema_weight','ema_start_step','total_batch_size','dfake_gen_update_ratio','image_or_video_shape',
    'distribution_loss','trainer','gradient_checkpointing','num_frame_per_block','persistent_state',
    'model_kwargs','phaseG_accumulation_sync_policy','generator_fsdp_wrap_strategy',
    'real_score_fsdp_wrap_strategy','fake_score_fsdp_wrap_strategy','text_encoder_fsdp_wrap_strategy'}


def js(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_public_bytes(name, data):
    if len(data) > FILE_LIMIT:
        raise ValueError('RESULT_FILE_TOO_LARGE: '+name)
    if any(p.search(data) for p in SECRET_PATTERNS):
        # Deliberately never include the matching secret/value in an exception.
        raise ValueError('PUBLICATION_BLOCKED_CREDENTIAL_PATTERN: '+name)


def safe_read(root, path, limit=FILE_LIMIT):
    root, path = Path(root).resolve(), Path(path).absolute()
    if not path.is_relative_to(root) or path.resolve() != path:
        raise ValueError('Refusing symlink/outside result path')
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Not a bounded regular result file: '+path.name)
    with path.open('rb') as f:
        data = f.read(limit+1)
    if len(data)>limit:
        raise ValueError('Growing result exceeded read limit')
    return data


def read_json(root, path):
    return json.loads(safe_read(root,path))


def accepted_log(root, path):
    data = safe_read(root,path)
    # Writers append one locked JSON line at a time. Ignore an incomplete last
    # line, never fabricate it or accept a malformed completed record.
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    rows = [json.loads(line) for line in lines]
    if any(r.get('accepted_optimizer_update') is not True for r in rows):
        raise ValueError('Non-accepted record in accepted-update log')
    return rows, lines


def validate_row(r, world):
    c = r['counts_after']
    if r['kind'] not in ['G','C'] or c['G']<0 or c['C']<0 or c['outer']!=c['C']:
        raise ValueError('Invalid update counters')
    for k in ['loss_global_mean','preclip_global_norm','seconds']:
        if not math.isfinite(float(r[k])):
            raise ValueError('NONFINITE_ACCEPTED_METRIC: '+k)
    if any(not math.isfinite(float(v)) for v in r.get('gradient_groups_preclip',{}).values()):
        raise ValueError('NONFINITE_GROUP_GRADIENT')
    if r['global_samples']!=64 or r['local_micros']!=64//world:
        raise ValueError('B64 sample/accumulation contract mismatch')
    if r['clip_calls']!=1 or r['optimizer_steps']!=1:
        raise ValueError('Expected exactly one clip and optimizer step')
    if r['accepted_global_data_cursor']!=64*(c['G']+c['C']):
        raise ValueError('Accepted sample cursor mismatch')
    if r['kind']=='C' and (c['outer']!=r['outer_before']+1 or c['G']!=(c['outer']+4)//5):
        raise ValueError('Native completed-outer schedule mismatch')


def validate_sequence(rows, initial):
    previous=initial.copy()
    for r in rows:
        if r['outer_before']!=previous['outer']:raise ValueError('Update sequence outer mismatch')
        expected=previous.copy()
        due=previous['outer']%5==0 and previous['G']==previous['outer']//5
        if (r['kind']=='G')!=due:raise ValueError('Native G/C update order mismatch')
        if r['kind']=='G':
            expected['G']+=1
            if expected['EMA_initialized']:expected['EMA_updates']+=1
        else:
            expected['outer']+=1;expected['C']+=1
            if expected['outer']==200:expected['EMA_initialized']=True
        if expected!=r['counts_after']:raise ValueError('Accepted counter restoration/sequence mismatch')
        previous=expected


def loss_svg(rows):
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480">',
           '<rect width="900" height="480" fill="white"/>']
    for panel,kind in enumerate(['G','C']):
        data=[r for r in rows if r['kind']==kind]
        if not data: continue
        values=[float(r['loss_global_mean']) for r in data]
        xs=[r['counts_after']['G' if kind=='G' else 'C'] for r in data]
        lo,hi=min(values),max(values); y0=panel*230+45
        points=['%.2f,%.2f'%(70+790*(x-min(xs))/max(1,max(xs)-min(xs)),
                  y0+160-160*(v-lo)/max(1e-12,hi-lo)) for x,v in zip(xs,values)]
        parts += ['<text x="70" y="%d" font-size="17">%s native loss; own update index</text>'%(y0-15,kind),
                  '<polyline fill="none" stroke="#2458b5" stroke-width="1.5" points="%s"/>'%' '.join(points),
                  '<text x="8" y="%d" font-size="12">%.6f</text>'%(y0+10,hi),
                  '<text x="8" y="%d" font-size="12">%.6f</text>'%(y0+160,lo),
                  '<text x="70" y="%d" font-size="12">%d</text>'%(y0+179,min(xs)),
                  '<text x="830" y="%d" font-size="12">%d</text>'%(y0+179,max(xs))]
    return ('\n'.join(parts+['</svg>'])+'\n').encode()


def collect(root, arm, world, observed_at=None, command_returncode=None, eval_root=None):
    root = Path(root).resolve()
    if arm not in ['baseline','ssf'] or world<1 or 64%world:
        raise ValueError('Choose baseline/ssf and a world size dividing B64')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',root.name):
        raise ValueError('Unsafe run_id')
    now=observed_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
    files={}; skipped=[]; armroot=root/'training'/arm
    attempts=sorted([p for p in armroot.glob('attempt_*') if re.fullmatch(r'attempt_\d+',p.name)],
                    key=lambda p:int(p.name.split('_')[1]))
    attempt=attempts[-1] if attempts else None
    rows=[]; common=0; observed=[]; start=[]; sources=[]
    all_rows=[]; all_lines=[]; missing=[]
    for rank in range(world):
        folder=attempt/f'rank{rank}' if attempt else None
        path=folder/'updates.jsonl' if folder else None
        initial=folder/'accepted_start.json' if folder else None
        if initial and initial.exists():start.append(read_json(root,initial))
        else:start.append(None)
        if not path or not path.exists():
            missing.append(rank); all_rows.append([]);all_lines.append([]);observed.append(0);continue
        rs,ls=accepted_log(root,path)
        for r in rs: validate_row(r,world)
        if rs and start[-1] is not None:validate_sequence(rs,start[-1]['counts'])
        if any(int(r['rank'])!=rank for r in rs):raise ValueError('Rank identity mismatch')
        all_rows.append(rs);all_lines.append(ls);observed.append(len(rs))
        sources.append(dict(rank=rank,path=str(path.relative_to(root)),bytes_read=sum(map(len,ls)),sha256=sha(b''.join(ls))))
    common=min(observed) if observed else 0
    fields=['kind','outer_before','counts_after','loss_global_mean','preclip_global_norm',
            'clip_calls','optimizer_steps','global_samples','local_micros','accepted_global_data_cursor']
    if common:
        if not all(start):raise ValueError('Accepted updates without complete accepted_start provenance')
        if any(s['counts']!=start[0]['counts'] for s in start):raise ValueError('Rank start counters differ')
        for i in range(common):
            if any(any(r[i][k]!=all_rows[0][i][k] for k in fields) for r in all_rows[1:]):
                raise ValueError('Rank accepted-prefix mismatch')
        cut=max([i+1 for i,r in enumerate(all_rows[0][:common]) if r['kind']=='C'] or [0])
        rows=all_rows[0][:cut]
        for rank in range(world):
            raw=b''.join(all_lines[rank][:cut]);check_public_bytes('raw rank%d'%rank,raw)
            files['raw/updates_rank%d.jsonl.gz'%rank]=gzip.compress(raw,mtime=0)
    else:cut=0
    counts=rows[-1]['counts_after'] if rows else (start[0]['counts'] if start and all(start) else None)
    config=start[0].get('config',{}) if start and start[0] else {}
    public_config={k:v for k,v in config.items() if k in CONFIG_KEYS}
    for key in ['generator_ckpt','real_model_path','data_path']:
        if config.get(key):public_config[key+'_basename']=Path(config[key]).name
    files['training_config.json']=js(public_config)
    for name in ['source_files.json','formal_code_manifest.json']:
        p=root/'source_manifest'/name
        if p.exists(): files['provenance/'+name]=safe_read(root,p)
    checkpoints=[]
    for p in sorted((armroot/'checkpoints').glob('*.COMMITTED.json')):
        m=read_json(root,p)
        keys=['status','SHA256','bytes','counts','cursor','arm','accumulation_position',
              'generator_raw_tensor_SHA256','generator_EMA_tensor_SHA256','readback_all_components_exact']
        checkpoints.append(dict(filename=p.name,**{k:m[k] for k in keys if k in m}))
    files['checkpoint_manifest.json']=js(checkpoints)
    finished=[]
    if attempt:
        for rank in range(world):
            p=attempt/f'rank{rank}'/'worker_finished.json'
            if p.exists():
                m=read_json(root,p);finished.append(dict(rank=rank,**{k:m.get(k) for k in ['success','process_group_cleanup_complete','at']}))
    complete_path=armroot/'TRAINING_AND_GENERATION_COMPLETE.json'
    if complete_path.exists():
        completed=read_json(root,complete_path)
        if not counts or (counts['outer'],counts['G'],counts['C'],counts['EMA_updates'])!=(600,120,600,80):
            raise ValueError('Completion marker contradicts accepted counters')
        files['training_completion.json']=js(completed)
        status='TRAINING_COMPLETE_HEALTH_REVIEW_STILL_REQUIRED'
    elif attempt and any((attempt/n).exists() for n in ['STOPPED_RECOVERABLE.json','STOPPED_STORAGE_RECOVERABLE.json']):
        status='STOPPED_RECOVERABLE_MARKER_PRESENT'
    elif finished and any(not x['success'] for x in finished):status='WORKER_STOPPED_OR_FAILED'
    elif missing:status='WAITING_FOR_RANK_LOGS'
    else:status='OBSERVED_TRAINING_PROGRESS' if rows else 'NO_COMPLETED_OUTER_OBSERVED'
    if command_returncode is not None and command_returncode!=0:
        status='COMMAND_FAILED_OR_STOPPED_CHECK_MARKERS'
    elif command_returncode==0 and not complete_path.exists():
        status='COMMAND_EXITED_NO_TRAINING_COMPLETION_MARKER'
    losses={}
    for kind in ['G','C']:
        data=[r for r in rows if r['kind']==kind];vals=[float(r['loss_global_mean']) for r in data]
        if vals:losses[kind]=dict(accepted_this_attempt=len(vals),first=vals[0],latest=vals[-1],mean=statistics.mean(vals),
            first10_mean=statistics.mean(vals[:10]),last10_mean=statistics.mean(vals[-10:]),
            mean_seconds=statistics.mean(float(r['seconds']) for r in data))
    group_grads={}
    for name in ['Reader','Writer','backbone']:
        vals=[float(r['gradient_groups_preclip'][name]) for r in rows
              if r['kind']=='G' and name in r.get('gradient_groups_preclip',{})]
        if vals:group_grads[name]=dict(mean=statistics.mean(vals),latest=vals[-1],maximum=max(vals))
    memory=[]
    for rank in range(world):
        rs=all_rows[rank][:cut]
        memory.append(dict(rank=rank,peak_allocated_bytes=max([r.get('memory',{}).get('peak_allocated',0) for r in rs] or [0]),
                          peak_reserved_bytes=max([r.get('memory',{}).get('peak_reserved',0) for r in rs] or [0])))
    if rows:
        buf=io.StringIO(newline=''); writer=csv.writer(buf)
        writer.writerow(['rank','kind','outer_before','outer','G','C','loss_global_mean','preclip_global_norm','seconds','global_samples','local_micros','clip_calls','optimizer_steps'])
        for rank in range(world):
            for r in all_rows[rank][:cut]:
                c=r['counts_after'];writer.writerow([rank,r['kind'],r['outer_before'],c['outer'],c['G'],c['C'],
                    *[r[k] for k in ['loss_global_mean','preclip_global_norm','seconds','global_samples','local_micros','clip_calls','optimizer_steps']]])
        files['updates.csv']=buf.getvalue().encode();files['loss.svg']=loss_svg(rows)
    # Only bounded selected log tails, never environment dumps or arbitrary logs.
    for rank in range(world):
        p=root/'logs'/('train_%s_rank%d.log'%(arm,rank))
        if p.exists():
            if p.resolve()!=p.absolute() or not p.resolve().is_relative_to(root):raise ValueError('Unsafe log path')
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-16384));data=f.read(16384)
            if len(data)==16384:data=data.split(b'\n',1)[-1]
            files['logs/rank%d_tail.log'%rank]=data
    eval_status='NOT_COLLECTED'
    if eval_root:
        eroot=Path(eval_root).resolve();reports=sorted((eroot/'reports').glob('*/summary.json'))
        if reports:
            latest=reports[-1].parent
            for name in ['summary.json','per_video.csv','report.md']:
                p=latest/name
                if p.exists():files['evaluation/'+name]=safe_read(eroot,p)
            eval_status=read_json(eroot,latest/'summary.json').get('status','UNKNOWN')
    summary=dict(schema_version=1,run_id=root.name,arm=arm,observed_at=now,status=status,
        world_size=world,counts=counts,attempt=None if attempt is None else attempt.name,
        accepted_updates_this_attempt=cut,rank_observed_update_counts=observed,
        missing_rank_logs=missing,rank_common_prefix_consistent=True if cut else None,
        snapshot_cut='Common accepted prefix through a completed critic/outer boundary; later in-flight work excluded',
        historical_attempts_not_merged=True,losses=losses,G_gradient_groups_preclip=group_grads,memory=memory,
        full_checkpoint_markers=checkpoints,evaluation_status=eval_status,workers_finished=finished,
        command_returncode=command_returncode,health_review='NOT_INFERRED_OR_UPLOADED',
        checkpoint_weights_read=False,weights_uploaded=False,videos_uploaded=False,
        limitations=['Snapshot is not live progress.','Loss changes are not video-quality evidence.',
                     'Ranks repeat one global update, not independent experiments.',
                     'Checkpoint hashes are copied from markers; weight bytes are not re-read.'])
    files['summary.json']=js(summary)
    c=counts or {}
    readme=['# SSF automatic result snapshot','',status,'',
        'Observed UTC: '+now,'',
        '| Field | Value |','|---|---|',
        '| Arm | %s |'%arm,'| Outer / target | %s / 600 |'%c.get('outer','unknown'),
        '| G / C | %s / %s |'%(c.get('G','unknown'),c.get('C','unknown')),
        '| World size | %d |'%world,'| EMA initialized / updates | %s / %s |'%(c.get('EMA_initialized','unknown'),c.get('EMA_updates','unknown')),
        '| Checkpoint markers | %d |'%len(checkpoints),'| Evaluation status | %s |'%eval_status,'',
        'No weight tensors, generated videos, training prompts or user health ratings are uploaded.',
        'See summary.json, updates.csv, checkpoint_manifest.json and logs/.']
    if rows:readme += ['','![Native losses](loss.svg)','', 'G and C use separate y axes; neither loss alone measures video quality.']
    files['README.md']=('\n'.join(readme)+'\n').encode()
    for name,data in files.items():
        checked=gzip.decompress(data) if name.endswith('.gz') else data
        check_public_bytes(name,checked)
    if sum(map(len,files.values()))>PACKAGE_LIMIT:raise ValueError('RESULT_PACKAGE_TOO_LARGE')
    manifest=dict(schema_version=1,run_id=root.name,arm=arm,observed_at=now,sources=sources,
                  files=[dict(path=k,bytes=len(v),sha256=sha(v)) for k,v in sorted(files.items())],
                  skipped=skipped,production_files_modified=False)
    files['manifest.json']=js(manifest)
    return summary,files
