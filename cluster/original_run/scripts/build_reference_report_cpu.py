"""Summarize frozen G0 references, with exact same-prompt membership counts."""
from g_common import *
import statistics

def main():
    c=read(ROOT/'contracts/evaluation_manifest.json');monitor=c['monitor']['prompts'];rows=[];sources=[]
    targets={d:{p['prompt_id'] for p in monitor if d in p['dims']} for d in c['metrics']}
    for d,ids in targets.items():
        mapped={}
        for arm in ['ODE','released_DMD']:
            for seed in [2026,2027]:
                folder=ROOT/'reference/scoring'/arm/str(seed)
                path=folder/(d+'_eval_results.json');prov=read(folder/(d+'_provenance.json'))
                assert prov['result_SHA256']==sha(path)
                aliases={}
                if d=='human_action':aliases={x['alias']:x['prompt_id'] for x in read(folder/'human_action_aliases/alias_manifest.json')}
                purpose='ODE_original_panel' if arm=='ODE' else 'reference_health_original'
                lookup={x['video']['path']:x['prompt_id'] for p in (ROOT/'reference/records').glob('*.json')
                    if (x:=read(p))['arm']==arm and x['seed']==seed and x['matrix_row']['purpose']==purpose}
                out={}
                for item in read(path)[d][1]:
                    pid=aliases[item['video_path']] if d=='human_action' else lookup[item['video_path']]
                    if pid in ids:out[pid]=float(item['video_results'])*(.01 if d=='imaging_quality' else 1.0)
                mapped[arm,seed]=out
                sources.append(dict(arm=arm,seed=seed,dimension=d,path=str(path),SHA256=sha(path)))
        kept=set(ids)
        for values in mapped.values():kept&=set(values)
        means={arm:None if not kept else statistics.mean(statistics.mean(mapped[arm,seed][pid] for seed in [2026,2027]) for pid in sorted(kept))*100 for arm in ['ODE','released_DMD']}
        rows.append(dict(dimension=d,eligible_same_health_prompts=len(ids),complete_same_prompts=len(kept),
            ODE_raw_points=means['ODE'],released_EMA_points=means['released_DMD'],
            units='official raw score x100; NOT official Total',inference_seeds='2026,2027',same_prompts_and_Local18=True))
    with (ROOT/'reference/ability_levels_same_health_prompts_v2.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gen=read(ROOT/'reference/generation_audit.json');legacy=read(ROOT/'reference/historical_overlap_v2_units/audit.json')
    assets=read(ROOT/'source_manifest/used_assets_verified.json');metric_assets=read(ROOT/'source_manifest/current_metric_assets.json')
    report=dict(status='G0_REFERENCE_GENERATION_SCORING_AND_IDENTITY_COMPLETE',reference_videos=136,
        ability_levels=rows,metric_sources=sources,registered_MVP_panel_size=48,health_monitor_size=8,
        actual_health_memberships={d:sorted(x) for d,x in targets.items()},
        ODE_SHA256=gen['ODE_SHA256'],released_EMA_SHA256=gen['released_EMA_SHA256'],
        unbounded_vs_Local18_first6_chunks_bit_exact_for_all8=True,final_chunk_changes_observed=True,
        used_training_inference_assets_matched=len(assets['records']),metric_weight_assets_matched=len(metric_assets['records']),
        initial_reference_asset_byte_check_timing=assets['initial_reference_models_loaded_before_this_independent_byte_scan'],
        historical_existing_video_references=len(legacy['existing_video_references']),historical_prompt_overlap=42,
        historical_stats_path=str(ROOT/'reference/historical_overlap_v2_units/absolute_levels.csv'),
        no_old_weight_or_video_bytes_rescanned_for_historical_reuse=True,no_retraining_or_new_old_scores=True,
        health_PASS=None,human_scores=None,AI_task_completion_rating='NOT_PERFORMED_NOT_FABRICATED',
        published_reference_is_not_budget_matched=True,paper84_31_reproduction_not_required=True,
        no_official_Total_or_partial_Quality_Semantic=True,
        interpretive_limits=['tiny health memberships; report N, do not infer equivalence or paper-score reproduction',
            'published ability reference does not establish the fresh Baseline healthy',
            'legacy B1/raw/20-75 and proposed B64/EMA/600outer differ in multiple ways; no single-factor attribution',
            'future Baseline gate needs frozen user O/A/T review, not only VBench or finite output'])
    report['unit_adapter']='imaging per-video MUSIQ /100 before common raw-dimension x100; other registered dimensions unchanged'
    dump(ROOT/'reference/G0_report_v2.json',report)
    lines=['# G0 能力参照与资产报告','',
        '136条参考视频、对应官方membership评分和输入身份核验已完成。正式Baseline尚未训练，未生成任何用户健康评分或健康PASS。',
        '',f"ODE SHA256：`{gen['ODE_SHA256']}`",f"发布EMA SHA256：`{gen['released_EMA_SHA256']}`",'',
        '发布模型原窗口与Local18在8条详细prompt上，初始/renoise相同，前6chunk raw-flow逐位相同，第7chunk发生预期变化。',
        '', '以下只比较健康门固定8条原始prompt、两个seed；每个指标仅用自身官方成员。先在prompt内平均两seed。不同指标实际N很小，不能当成完整48面板或官方Total。', '',
        '|维度|合格prompt数|完整配对数|ODE raw×100|发布EMA×100|','|---|---:|---:|---:|---:|']
    for x in rows:
        fmt=lambda v:'NA' if v is None else f'{v:.3f}'
        lines.append(f"|{x['dimension']}|{x['eligible_same_health_prompts']}|{x['complete_same_prompts']}|{fmt(x['ODE_raw_points'])}|{fmt(x['released_EMA_points'])}|")
    lines+=['','发布EMA是能力参照，并非同预算竞争者，也不替代ODE训练初始化。Baseline不必达到发布分数；同任务参照表现差时应标任务适用性，不能改健康阈值。',
        '', '已复用旧R1/R2、20/50/75、Baseline/SSF Normal的42条相同prompt及两个seed：120项绝对水平见historical_overlap_v2_units/absolute_levels.csv。旧视频和旧权重没有重新扫描，未重跑旧VBench。',
        '', '旧能力表与本表的cohort口径不同，不能直接把表间分数差解释为训练退化。旧B1/raw与新B64/EMA/600outer存在多项配方差异，不作单一因素因果归因，不宣布旧R1/R2训崩。',
        '', '初始参考模型先加载，独立大资产字节复验随后完成；ODE/发布checkpoint本身在各自加载前已核验SHA。该时间顺序已如实保留。当前所有实际训练/推理资产及9维所用8个评分权重均匹配既有冻结清单。',
        '', '单位修正：官方imaging_quality视频层MUSIQ为0–100，维度总分才除100。本表先转换为维度raw，再统一×100；旧版仅本次新汇总误放大100倍，原始评分及Phase C+D既有统计没有改动。旧版保留并标为SUPERSEDED。',
        '', 'G1真实DMD累积与完整恢复仍以运行结果为准。尚不进入正式训练或人工健康PASS判定。']
    with (ROOT/'reference/G0_report_v2.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('G0_REFERENCE_REPORT_WRITTEN_NO_HEALTH_PASS')

if __name__=='__main__':main()
