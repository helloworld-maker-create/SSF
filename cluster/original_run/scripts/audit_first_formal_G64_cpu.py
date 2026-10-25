"""Audit recorded first Baseline G64; CPU only, never changes training state."""
from g_common import *
import math, statistics, sys


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_AUDIT_ONLY'
    sys.path.insert(0, str(SOURCE))
    from utils.dataset import TextDataset
    from phaseg_formal_control import local_order
    check_source()
    target = ROOT/'training/baseline/first_G64_integrity.json'
    assert not target.exists(), 'PRESERVE_EXISTING_AUDIT'
    byrank = []; tapes = []; per_sample_losses = []; indices = []
    for rank in [0, 1]:
        folder = ROOT/f'training/baseline/attempt_00/rank{rank}'
        start = read(folder/'accepted_start.json')
        dataset = TextDataset(start['config']['data_path'])
        order = local_order(len(dataset), rank, 2, 0)
        lines = (folder/'micro_forward.jsonl').read_text().splitlines()
        rows = [(i+1, json.loads(line)) for i, line in enumerate(lines)
                if json.loads(line)['kind'] == 'G' and json.loads(line)['outer'] == 0]
        assert [x['micro'] for _, x in rows] == list(range(32))
        updates = [json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
        u = next(x for x in updates if x['kind'] == 'G' and x['outer_before'] == 0)
        assert u['accepted_optimizer_update'] and u['global_samples'] == 64 and u['local_micros'] == 32
        assert u['clip_calls'] == u['optimizer_steps'] == 1
        assert u['counts_after'] == dict(outer=0,G=1,C=0,EMA_initialized=False,EMA_updates=0)
        assert u['accepted_global_data_cursor'] == 64
        assert u['micro_gradient_dtypes'] == [['torch.float32']]*32
        assert math.isfinite(u['preclip_global_norm']) and u['preclip_global_norm'] > 0
        assert u['gradient_groups_preclip']['Reader'] == u['gradient_groups_preclip']['Writer'] == 0
        assert math.isclose(u['preclip_global_norm'],u['gradient_groups_preclip']['backbone'],rel_tol=1e-5,abs_tol=1e-10)
        exit_choice = u['shared_exit_indices'][0]
        assert u['actual_exit_choice'] == exit_choice and len(u['shared_exit_indices']) == 7
        assert u['dataset_indices'] == order[:32]
        for line_number, x in rows:
            assert x['rank'] == rank and x['dataset_index'] == order[x['micro']]
            text = dataset[x['dataset_index']]['prompts']
            assert x['prompt_SHA256'] == hashlib.sha256(text.encode()).hexdigest()
            assert math.isfinite(x['loss']) and all(math.isfinite(v) for v in x['loss_aux'].values())
            assert x['writer_calls'] == 0
            # Native context refresh still draws noise at t=0: seven draws.
            # Plus seven per non-exit denoise and one DMD score-noise draw.
            assert sum(d['function']=='randn_like' for d in x['actual_stochastic_tensors']) == 7*exit_choice+7+1
            tape = {k:x[k] for k in ['rank','outer','kind','micro','dataset_index','prompt_SHA256','actual_stochastic_tensors']}
            encoded = json.dumps(tape,sort_keys=True,separators=(',',':')).encode()
            tapes.append(dict(rank=rank,micro=x['micro'],source=str(folder/'micro_forward.jsonl'),line=line_number,exogenous_tape_SHA256=hashlib.sha256(encoded).hexdigest()))
            per_sample_losses.append(x['loss']);indices.append(x['dataset_index'])
        byrank.append(dict(rank=rank,update=u,raw_first32_lines_SHA256=hashlib.sha256(('\n'.join(lines[:32])+'\n').encode()).hexdigest()))
    assert len(set(indices)) == len(indices) == 64
    assert byrank[0]['update']['shared_exit_indices'] == byrank[1]['update']['shared_exit_indices']
    assert byrank[0]['update']['preclip_global_norm'] == byrank[1]['update']['preclip_global_norm']
    mean = statistics.mean(per_sample_losses)
    assert all(math.isclose(mean,x['update']['loss_global_mean'],rel_tol=1e-12,abs_tol=1e-12) for x in byrank)
    tape_hash = hashlib.sha256(json.dumps(tapes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result = dict(status='PASS_FIRST_FORMAL_G64_NATIVE_DATA_AND_COUNTER_AUDIT',
        at=datetime.datetime.now().astimezone().isoformat(),native_DMD_64_sample_mean=mean,rank_results=byrank,
        exogenous_tapes=tapes,exogenous_tape_bundle_SHA256=tape_hash,actual_first_exit_applied_to_all_chunks=True,
        proof='native first flag plus actual per-micro renoise counts; no seventh-list-entry substitution',
        sampler_seed=0,training_seed=2027,distinct_training_samples=64,State_parameters=0,
        CUDA_used=False,optimizer_operation_performed_by_audit=False,code_SHA256=sha(__file__),
        scope='first G update only; C/600outer/EMA/health/SSF/MVP remain unverified')
    dump(target,result)
    print(json.dumps(dict(status=result['status'],mean_loss=mean,global_grad_norm=byrank[0]['update']['preclip_global_norm'],shared_exit=exit_choice,distinct_samples=64,path=str(target))))


if __name__ == '__main__': main()
