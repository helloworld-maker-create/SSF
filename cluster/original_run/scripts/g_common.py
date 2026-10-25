"""Phase G isolated execution: budgets, exclusive resources and immutable provenance."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source'
PRODUCTION=Path('/data/wangyuqi_intern26/repos/Self-Forcing')

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def append(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
def storage_guard(required=2*1024**3):
    s=os.statvfs(ROOT);free=s.f_bavail*s.f_frsize
    assert free-required>=32*1024**3,f'BLOCKED_STORAGE_DYNAMIC free={free}, next={required}'
    return free
def check_source():
    m=read(ROOT/'source_manifest/source_files.json')
    for f in m['files']:assert sha(SOURCE/f['path'])==f['SHA256'],'SOURCE_CHANGED '+f['path']
    parent=read(ROOT/'contracts/parent_version_manifest.json')
    for f in parent['files']:assert sha(ROOT/'contracts'/f['name'])==f['SHA256'],'CONTRACT_CHANGED '+f['name']
    return m
def resources(before_model=False):
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];assert ',' not in uuid
    raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    v=[x.strip() for x in raw.split(',')]
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[x for x in procs.splitlines() if v[1] in x and int(x.split(',')[1])!=os.getpid()]
    assert not foreign,'BLOCKED_RESOURCE_OTHER_PROCESS '+str(foreign)
    if before_model:assert float(v[3])<=5 and int(v[4])>=120*1024,'BLOCKED_RESOURCE '+raw
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(mem['MemAvailable'].split()[0])*1024
    assert available>=128*1024**3,'BLOCKED_HOST_RAM'
    free=storage_guard()
    return dict(at=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),gpu=raw,foreign=foreign,
                filesystem_available_bytes=free,host_MemAvailable_bytes=available)
def acquire_gpu():
    folder=ROOT.parent/'.gpu_leases';folder.mkdir(exist_ok=True)
    uuid=os.environ['CUDA_VISIBLE_DEVICES'];p=folder/(uuid+'.lock');f=p.open('a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    append(ROOT/'resource_snapshots/leases.jsonl',dict(pid=os.getpid(),GPU=uuid,at=time.time(),action='acquire',lock_path=str(p)))
    return f
def matrix(stage='A'):
    with (ROOT/'contracts/matched_mvp_matrix.csv').open() as f:return [x for x in csv.DictReader(f) if x['stage']==stage]

def approved_stage_A():
    a=read(ROOT/'authorizations/stage_A_user_start.json')
    assert a['origin']=='actual user turn' and a['stage_A_budget_GPUh']==dict(qualification=48,reference_generation_scoring=8)
    assert a['parent_contract_bundle_SHA256']==read(ROOT/'contracts/parent_version_manifest.json')['bundle_SHA256']
    return a

def budget_used(category):
    # Sum wall-clock GPU time of terminal AND still-running worker handles.
    total=0.0
    for p in (ROOT/'state').glob('*_started.json'):
        x=read(p)
        if x.get('budget_category')!=category:continue
        finish=p.with_name(p.name.replace('_started.json','_finished.json'))
        if finish.exists():end=read(finish)['ended_at_epoch']
        else:end=time.time()
        total+=(end-x['started_at_epoch'])*x.get('GPU_count',1)/3600
    return total

def budget_guard(category,cap):
    used=budget_used(category);assert used<cap-.05,f'BLOCKED_BUDGET {category} used={used} cap={cap}'
    return used
