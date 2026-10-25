"""Hash only assets actually used by Phase G; stdlib, no model/GPU imports."""
from g_common import *
import traceback

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    prior=ROOT.parent/'phaseF_reader_credit_20261023_143148/source_manifest/input_freeze_end.json'
    assert sha(prior)=='10cca03f842601455aef2ceee6b7d5a2d8fe627b46dd27c3966c370bfffb2c43'
    old=read(prior);assert old['verdict']=='PASS_ALL55_INPUT_BYTES_UNCHANGED'
    base='/data/wangyuqi_intern26/repos/Self-Forcing/wan_models/Wan2.1-T2V-1.3B/'
    teacher='/data/tuyijing/IP2V/model_weights/model_weights/Wan2.1-T2V-14B/'
    prompts='/data/wangyuqi_intern26/repos/Self_Gradient_Forcing/prompts/vidprom_filtered_extended.txt'
    selected=[]
    for x in old['files']:
        path=x['path'];name=Path(path).name
        if path.startswith(base) or path==prompts or (path.startswith(teacher) and (name=='config.json' or name.startswith('diffusion_pytorch_model'))):selected.append(x)
    assert len(selected)>=17
    completed=[]
    for x in selected:
        p=Path(x['path']);before=p.stat();start=time.time();actual=sha(p);after=p.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'ASSET_CHANGED_DURING_HASH'
        assert before.st_size==x['bytes'] and actual==x['sha256'],'USED_ASSET_IDENTITY_MISMATCH '+str(p)
        rec=dict(path=str(p),resolved=str(p.resolve()),SHA256=actual,bytes=before.st_size,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns,seconds=time.time()-start)
        completed.append(rec);append(ROOT/'logs/used_asset_hash_progress.jsonl',rec);print('USED_ASSET_VERIFIED',p.name,x['bytes'],rec['seconds'],flush=True)
    dump(ROOT/'source_manifest/used_assets_verified.json',dict(status='PASS_USED_TRAINING_AND_INFERENCE_ASSET_BYTES',checked_at=time.time(),records=completed,bytes_read=sum(x['bytes'] for x in completed),
        prior_audit_manifest=str(prior),prior_manifest_SHA256=sha(prior),
        scope='used input assets only; no old model snapshots or old videos rescanned',
        initial_reference_models_loaded_before_this_independent_byte_scan=True,
        before_real_DMD_qualification=True,GPU_used=False))

if __name__=='__main__':
    try:main()
    except BaseException:append(ROOT/'logs/asset_freeze_failure.jsonl',dict(at=time.time(),traceback=traceback.format_exc()));raise
