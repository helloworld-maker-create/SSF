"""Goal B orchestration utilities; no model imports or production mutation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/data/wangyuqi_intern26/repos/Self-Forcing')
A=ROOT.parent/'causal_attribution_5s_20261012_101223'
GENPY='/data/tuyijing/anaconda3/envs/self/bin/python'
EVALPY='/data/tuyijing/anaconda3/envs/vbench2/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def check_source():
 expected=read(ROOT/'source_manifest/manifest.json')
 files=sorted(set(subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()))
 records=[dict(path=n,sha256=sha(REPO/n)) for n in files if (REPO/n).suffix in ('.py','.yaml','.yml','.sh') and (REPO/n).is_file()]
 digest=hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 assert digest==expected['SHA256'],'SOURCE_STATE_CHANGED'
 assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=REPO)).hexdigest()==expected['git_diff_SHA256'],'SOURCE_STATE_CHANGED git diff'
 for n,h in read(ROOT/'approved_plan/package_manifest.json').items():assert sha(ROOT/'approved_plan'/n)==h,'APPROVED_CONTRACT_CHANGED '+n
 return digest
def resources():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
 proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 result=[]
 for line in raw.splitlines():
  i,u,n,v,f,t=[s.strip() for s in line.split(',')];p=[s for s in proc.splitlines() if u in s]
  result.append(dict(index=int(i),uuid=u,name=n,util=float(v),free_MiB=int(f),total_MiB=int(t),processes=p,idle=not p and float(v)<=5 and int(f)>=120*1024 and n=='NVIDIA H20-3e'))
 return dict(time=time.time(),gpus=result,processes=proc)
def preflight(training=False):
 assert os.environ.get('PYTHONNOUSERSITE')=='1'
 visible=os.environ['CUDA_VISIBLE_DEVICES'];assert ',' not in visible
 snap=resources();g=next(x for x in snap['gpus'] if x['uuid']==visible or str(x['index'])==visible)
 foreign=[p for p in g['processes'] if int(p.split(',')[1])!=os.getpid()]
 assert not foreign and g['util']<=5 and g['free_MiB']>=120*1024,'BLOCKED_RESOURCE before model load '+str(g)
 available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
 if training:assert available>=350*2**30,'BLOCKED_RESOURCE host RAM'
 return dict(time=time.time(),gpu=g,host_available_bytes=available)
def all_prompts():
 suite=read(ROOT/'contracts/original_prompt_suite.json');panel=read(ROOT/'contracts/prompt_panel.json');bytext={p['prompt']:p for p in panel['prompts']};rows=[]
 for i,item in enumerate(suite['prompts']):
  pid=f'p{i:04d}-'+hashlib.sha256(item['prompt'].encode()).hexdigest()[:12]
  if item['prompt'] in bytext:assert pid==bytext[item['prompt']]['prompt_id']
  rows.append({**item,'prompt_id':pid,'source_index':i})
 assert len(rows)==944 and len({p['prompt'] for p in rows})==944
 return rows
def selected_prompts(scope):return read(ROOT/'contracts/prompt_panel.json')['prompts'] if scope=='panel' else all_prompts()
def gen_dir(rep,condition,seed):return ROOT/'generation'/rep/condition/str(seed)
def eval_dir(rep,condition,seed,scope):return ROOT/('panel_eval' if scope=='panel' else 'full_vbench')/rep/condition/str(seed)
def registry(rep,condition):return read(ROOT/'checkpoint_registry.json')[rep]['baseline' if condition=='baseline' else 'ssf']
def loaded_sources(*names):return {n:sha(ROOT/'scripts'/n) for n in names}
def block_downloads():
 import urllib.request,requests
 def blocked(*a,**kw):raise RuntimeError('Dependency download forbidden')
 urllib.request.urlretrieve=blocked;urllib.request.urlopen=blocked;requests.sessions.Session.request=blocked
 original=subprocess.run
 def guarded(command,*a,**kw):
  if isinstance(command,(list,tuple)) and command:
   name=Path(str(command[0])).name
   if name in ('wget','curl','rm','unlink') or (name=='git' and 'clone' in command):raise RuntimeError('Forbidden command '+repr(command))
  return original(command,*a,**kw)
 subprocess.run=guarded
