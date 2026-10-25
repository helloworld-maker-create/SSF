"""Export frozen full lossless videos and a genuinely blank user score table.

No model is loaded. No AI judgement or user score is generated. Original
filenames, model labels and quality results stay out of the blind package.
"""
from g_common import *
import argparse, secrets


def anonymize(items, salt):
    assert len(items) == 64
    assert len({x['source_video_id'] for x in items}) == 64
    ordered = sorted(items, key=lambda x: hashlib.sha256(
        salt + x['source_video_id'].encode()).hexdigest())
    return [{**x, 'clip_id': f'clip_{i+1:04d}'} for i, x in enumerate(ordered)]


def blank_row(item):
    return dict(clip_id=item['clip_id'], prompt=item['prompt'],
                inference_seed=item['seed'], O='',
                A='' if item['action_applicable'] else 'NA', T='',
                severe_deformation='', severe_identity_or_object_failure='',
                action_freeze='' if item['action_applicable'] else 'NA',
                viewed_version='lossless_source', notes='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    check_source()
    gate = read(ROOT/'contracts/baseline_health_gate.json')
    complete = read(ROOT/'training/baseline/TRAINING_AND_GENERATION_COMPLETE.json')
    assert complete['status'] == 'PASS_600_OUTER_120G_600C_EMA80'
    assert all(complete['counts'][k] == v for k,v in
               {'outer':600,'G':120,'C':600,'EMA_updates':80}.items())
    committed = read(ROOT/'training/baseline/checkpoints/outer_000600.COMMITTED.json')
    assert committed['status'] == 'COMMITTED' and committed['readback_all_components_exact']
    for rank in [0,1]:
        attempts = sorted((ROOT/'training/baseline').glob(f'attempt_*/rank{rank}/worker_finished.json'))
        assert attempts and read(attempts[-1])['success'], 'WAIT_FOR_BOTH_WORKERS_CLEANUP'
    ids = set(gate['monitor_prompt_ids']); items = []
    action_ids = set(gate['action_prompt_ids'])
    for path in sorted((ROOT/'reference/records').glob('*.json')):
        r = read(path)
        purpose = r['matrix_row']['purpose']
        if r['prompt_id'] not in ids: continue
        if (r['arm'] == 'ODE' and purpose == 'ODE_original_panel') or (
                r['arm'] == 'released_DMD' and purpose == 'reference_health_original'):
            items.append(dict(source_video_id=r['matrix_row']['video_id'],
                model_identity=r['arm'], weight=r['matrix_row']['weight'],
                prompt_id=r['prompt_id'], prompt=r['matrix_row']['prompt'], seed=r['seed'],
                video=r['video'], action_applicable=r['prompt_id'] in action_ids))
    from phaseg_eval_inputs import selected_rows, load_generation_records
    for weight in ['raw','EMA']:
        for seed in gate['seeds']:
            rows = [r for r in selected_rows('baseline',600,weight,seed) if r['prompt_id'] in ids]
            records = load_generation_records(rows)
            for row in rows:
                r = records[row['prompt']]
                assert r['checkpoint_SHA256'] == committed['SHA256']
                items.append(dict(source_video_id=row['video_id'], model_identity='baseline',
                    weight=weight, prompt_id=row['prompt_id'], prompt=row['prompt'],seed=seed,
                    video=r['video'], action_applicable=row['prompt_id'] in action_ids))
    assert len(items) == gate['blind_pack_unique_model_videos'] == 64
    assert all(sum(x['prompt_id']==pid and x['seed']==seed for x in items)==4
               for pid in ids for seed in gate['seeds'])
    for item in items:
        video=item['video']; assert video['lossless']
        assert sha(video['path']) == video['SHA256']
    base=ROOT/'health_review'; assert not (base/'BLIND_PACKAGE_READY.json').exists(), 'PRESERVE_EXISTING_PACKAGE'
    assert not (base/'blind').exists() and not (base/'private').exists(), 'PRESERVE_PARTIAL_PACKAGE_FOR_REVIEW'
    blind=base/'blind';private=base/'private'
    blind.mkdir(parents=True);private.mkdir()
    salt=secrets.token_bytes(32);coded=anonymize(items,salt)
    dump(private/'mapping.json',dict(salt_hex=salt.hex(),items=coded,
        hidden_until_user_scores_frozen=True,AI_scores=None,user_scores=None))
    public=[];scores=[]
    for item in coded:
        target=blind/'videos'/(item['clip_id']+'.mp4');target.parent.mkdir(exist_ok=True)
        original=Path(item['video']['path']);os.link(original,target)
        assert not target.is_symlink() and os.path.samefile(original,target)
        public.append(dict(clip_id=item['clip_id'],prompt=item['prompt'],seed=item['seed'],
                           file='videos/'+target.name,SHA256=item['video']['SHA256'],
                           action_applicable=item['action_applicable']))
        scores.append(blank_row(item))
    dump(blind/'video_manifest.json',public)
    with (blind/'blank_user_scores.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    instructions='''# Phase G 用户盲审\n\n请观看所有64条完整视频并填写blank_user_scores.csv。每条均为原lossless科学输入，没有观看转码。\n\nO对象: 0错误/缺失，1可辨但明显问题，2清楚合理。\nA动作: 0缺失/错误，1部分/含糊，2完成；不适用保持NA。\nT时序: 0严重不可用，1局部问题但可用，2连贯。T=0计入严重失败；合理静态场景不能仅因静态记0。\nsevere_deformation、severe_identity_or_object_failure、适用的action_freeze请填写0或1。要求运动却广泛冻结/重复时标记相应问题。\n\n此表全部待用户亲自评分；不存在预填AI结论。请完整评分后提交，冻结评分后才揭示模型映射并判定健康门。未获真实健康PASS和明确阶段C批准前不启动SSF。\n\n盲法有局限：您可能认识prompt或旧视频；同用户文件系统内的硬链接可被技术手段关联。请依据当前匿名包评审，不查看private映射目录。\n'''
    with (blind/'README.md').open('x') as f:f.write(instructions)
    dump(base/'BLIND_PACKAGE_READY.json',dict(status='AWAITING_USER_HEALTH_REVIEW',
        clips=64,unique_prompts=8,inference_seeds=gate['seeds'],source_lossless=True,
        hardlinks_no_duplicate_video_storage=True,user_scores_frozen=False,health_PASS=None,
        stage_C_execution='NOT_AUTHORIZED',AI_judgement='NOT_PERFORMED',
        blind_directory=str(blind),blank_scores=str(blind/'blank_user_scores.csv'),
        no_GPU_job_started=True))
    print('AWAITING_USER_HEALTH_REVIEW',str(blind),flush=True)


if __name__=='__main__': main()
