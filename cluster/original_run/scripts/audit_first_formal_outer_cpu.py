"""Audit the first completed native G64/C64 outer; no GPU or optimizer calls."""
from g_common import *
import math, statistics, sys


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_AUDIT_ONLY'
    sys.path.insert(0,str(SOURCE))
    from utils.dataset import TextDataset
    from phaseg_formal_control import local_order,validate_boundary
    from g_accumulation import EventCounters
    check_source()
    results=[];content=[];indices=[];means={'G':[],'C':[]}
    for rank in [0,1]:
        folder=ROOT/f'training/baseline/attempt_00/rank{rank}'
        start=read(folder/'accepted_start.json')
        dataset=TextDataset(start['config']['data_path'])
        order=local_order(len(dataset),rank,2,0)
        raw=[json.loads(x) for x in (folder/'micro_forward.jsonl').read_text().splitlines()]
        updates=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
        for offset,kind in enumerate(['G','C']):
            u=next(x for x in updates if x['kind']==kind and x['outer_before']==0)
            assert u['accepted_optimizer_update'] and u['global_samples']==64 and u['local_micros']==32
            assert u['clip_calls']==u['optimizer_steps']==1
            assert u['micro_gradient_dtypes']==[['torch.float32']]*32
            assert math.isfinite(u['preclip_global_norm']) and u['preclip_global_norm']>0
            assert u['dataset_indices']==order[32*offset:32*(offset+1)]
            indices.extend(u['dataset_indices'])
            rows=[x for x in raw if x['kind']==kind and x['outer']==0]
            assert [x['micro'] for x in rows]==list(range(32))
            assert u['actual_exit_choice']==u['shared_exit_indices'][0]
            for x in rows:
                assert x['dataset_index']==order[32*offset+x['micro']]
                assert x['prompt_SHA256']==hashlib.sha256(dataset[x['dataset_index']]['prompts'].encode()).hexdigest()
                assert math.isfinite(x['loss']) and all(math.isfinite(v) for v in x['loss_aux'].values())
                assert x['writer_calls']==0
                assert sum(d['function']=='randn_like' for d in x['actual_stochastic_tensors'])==7*u['actual_exit_choice']+8
                means[kind].append(x['loss'])
                tape={k:x[k] for k in ['rank','outer','kind','micro','dataset_index','prompt_SHA256','actual_stochastic_tensors']}
                digest=hashlib.sha256(json.dumps(tape,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                content.append(dict(rank=rank,kind=kind,outer=0,micro=x['micro'],exogenous_content_SHA256=digest))
            if kind=='C':
                counter=EventCounters()
                for key,value in u['counts_after'].items():setattr(counter,key,value)
                validate_boundary(counter,64,2)
                assert u['counts_after']==dict(outer=1,G=1,C=1,EMA_initialized=False,EMA_updates=0)
                assert u['accepted_global_data_cursor']==128
            results.append(u)
    assert len(set(indices))==len(indices)==128
    for kind in ['G','C']:
        group=[x for x in results if x['kind']==kind]
        assert len(means[kind])==64
        assert group[0]['shared_exit_indices']==group[1]['shared_exit_indices']
        assert group[0]['preclip_global_norm']==group[1]['preclip_global_norm']
        assert all(math.isclose(statistics.mean(means[kind]),x['loss_global_mean'],abs_tol=1e-12,rel_tol=1e-12) for x in group)
    # Comparable across arms: deliberately exclude provenance paths and all
    # endogenous losses/forward predictions from the exogenous tape digest.
    content.sort(key=lambda x:(x['rank'],x['kind'],x['micro']))
    digest=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    tape_path=ROOT/'seed_manifests/baseline_first_outer_exogenous.json'
    dump(tape_path,dict(training_seed=2027,sampler_seed=0,world_size=2,microbatch=1,accumulation=32,
        content=content,comparable_content_SHA256=digest,
        exclusion='no paths, losses, predictions or parameter values in content digest',
        raw_log_sources=[str(ROOT/f'training/baseline/attempt_00/rank{r}/micro_forward.jsonl') for r in [0,1]],
        source_manifest_SHA256=sha(ROOT/'source_manifest/source_files.json')))
    output=dict(status='PASS_FIRST_COMPLETED_OUTER_G64_C64',at=datetime.datetime.now().astimezone().isoformat(),
        update_results=results,accepted_counts=dict(outer=1,G=1,C=1,EMA_initialized=False,EMA_updates=0),
        accepted_global_data_cursor=128,distinct_samples=128,global_means={k:statistics.mean(v) for k,v in means.items()},
        tape_manifest=str(tape_path),tape_manifest_SHA256=sha(tape_path),code_SHA256=sha(__file__),
        GPU_used_by_audit=False,optimizer_operation_by_audit=False,
        scope='first outer only; no health verdict, EMA event, complete formal snapshot or SSF match claimed')
    path=ROOT/'training/baseline/first_outer_integrity.json';dump(path,output)
    print(json.dumps(dict(status=output['status'],counts=output['accepted_counts'],global_means=output['global_means'],path=str(path))))


if __name__=='__main__':main()
