"""CPU-only postflight accounting. Never launches or loads a model.

Writes a new report after all G1 checks have passed. It does not replace
the frozen plan, original reports, authorization records, or worker results.
"""
import datetime
import json
import os
from pathlib import Path

from g_common import ROOT, budget_used, check_source, read, sha


def main():
    check_source()
    q = read(ROOT / 'qualification/real_dmd_accumulation/complete.json')
    assert q['status'] in ['PASS_G1_B64_NATIVE_DMD_ACCUMULATION_EMA_RESUME', 'PASS_G1_B64_SYNC_CANDIDATE_NATIVE_DMD_EMA_RESUME']
    candidate = bool(q.get('formal_policy_change_pending_approval',False))
    folder = Path(q['attempt_directory'])
    world = q['world_size']
    assert world == 2
    records = []
    loads, saves, emas = [], [], []
    for rank in range(world):
        for kind in ('G', 'C'):
            row = read(folder / f'{kind}_rank{rank}.json')
            assert row['status'] == 'PASS_NATIVE_DMD_B64_GRADIENT_MEAN'
            assert row['exact_forward_loss_and_rng_replay']
            assert row['comparison']['relative_L2_error'] <= .01
            assert not any(row['comparison']['gradient_presence_mismatches_by_rank'])
            assert row['accumulation']['clip_calls'] == 1
            assert row['qualification_optimizer_updates'] == 1 and row['formal_updates'] == 0
            records.append(row)
        restored = read(folder / f'restore_rank{rank}.json')
        assert restored['status'] == 'PASS_COMPLETE_FSDP_OPTIMIZER_RNG_MAPPING'
        assert restored['full_named_optimizer_state_matches_saved']
        assert restored['RNG_deliberately_advanced_then_exactly_restored']
        loads.append(read(folder / f'model_load_rank{rank}.json'))
        saves.append(read(folder / f'checkpoint_cost_rank{rank}.json'))
        emas.append(read(folder / f'EMA_profile_rank{rank}.json'))
    committed = read(folder / 'qualification_COMMITTED.json')
    assert committed['status'] == 'COMMITTED' and committed['all_components_readback_equal']
    assert Path(committed['snapshot']).stat().st_size == committed['bytes']
    g_seconds = max(x['accumulation_seconds'] for x in records if x['kind'] == 'G')
    c_seconds = max(x['accumulation_seconds'] for x in records if x['kind'] == 'C')
    overhead_seconds = dict(
        model_initialization=max(x['seconds'] for x in loads),
        three_full_saves_and_readbacks=3 * max(x['save_and_readback_seconds'] for x in saves),
        EMA_initialization=max(x['init_seconds'] for x in emas),
        eighty_EMA_refreshes=80 * max(x['one_function_refresh_seconds'] for x in emas),
        conditioning_46080_examples=max(x['conditioning128_examples_wall_seconds'] for x in saves) * 46080 / 128,
    )
    compute_GPUh = (120 * g_seconds + 600 * c_seconds) * world / 3600
    overhead_GPUh = sum(overhead_seconds.values()) * world / 3600
    sizes = [p.stat().st_size for p in (ROOT / 'reference/videos').glob('*.mp4')]
    assert len(sizes) == 136
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize / 2**30
    snapshot = committed['bytes'] / 2**30
    # This is a conservative planning reserve, NOT an execution grant or a
    # claim that future lossless output sizes are bounded by observed files.
    video_B = max(60 * 256 / 664, max(sizes) * 256 / 2**30) + 2
    storage_B = dict(
        three_routine_full_snapshots_GiB=3 * snapshot,
        one_retained_emergency_GiB=snapshot,
        one_temporary_full_write_GiB=snapshot,
        video_scoring_log_projection_GiB=video_B,
        proposed_operational_floor_GiB=32,
        startup_conservative_projection_GiB=5 * snapshot + video_B + 32,
        available_GiB=free,
        projected_shortfall_GiB=max(0, 5 * snapshot + video_B + 32 - free),
        floor_and_formal_stage_storage_contract_require_approval=True,
        quota='UNKNOWN',
        already_existing_qualification_not_counted_twice=True,
        future_formal_snapshot_size_not_yet_measured=True,
    )
    costs = dict(
        fixed_world_size=world,
        measured_G64_seconds_max_rank=g_seconds,
        measured_C64_seconds_max_rank=c_seconds,
        compute_only_GPUh_per_arm=compute_GPUh,
        measured_overhead_projection_seconds=overhead_seconds,
        overhead_GPUh_per_arm=overhead_GPUh,
        total_instrumented_point_projection_GPUh_per_arm=compute_GPUh + overhead_GPUh,
        proposed_training_cap_GPUh_per_arm=500,
        estimate_within_proposed_cap=compute_GPUh + overhead_GPUh <= 500,
        generation_scoring_budget_separate=True,
        qualification_instrumentation_included_not_clean_speed_benchmark=True,
        sampled_exit_plan_by_kind={k: next(x['shared_exit_indices'] for x in records if x['kind'] == k) for k in ('G', 'C')},
        limitation='One G and one C exit plan were measured. This is not a bound over all later stochastic exits, shared-host contention, interruptions or all save sizes.',
    )
    report = dict(
        at=datetime.datetime.now().astimezone().isoformat(),
        status='QUALIFICATION_CANDIDATE_PASSED_FORMAL_ADOPTION_PENDING' if candidate else 'STAGE_A_PASSED_FORMAL_EXECUTION_NOT_AUTHORIZED',
        G0=read(ROOT / 'reference/generation_audit.json')['status'],
        G0_current_report=str(ROOT / 'reference/G0_report_v2.md'),
        G1=q['status'],
        accumulation_sync_policy=q.get('accumulation_sync_policy','no_sync_first_A_minus1'),
        formal_accumulation_policy_change_pending_approval=candidate,
        original_no_sync_G1_result='FAIL_NEW_ACCUMULATION_QUALIFICATION' if candidate else None,
        original_failure_evidence=str(ROOT/'qualification/real_dmd_accumulation/attempt_00/G_comparison_failed_rank0.json') if candidate else None,
        costs=costs,
        stage_B_storage_forecast=storage_B,
        resource_measurements=[dict(rank=x['rank'],kind=x['kind'],reference=x['reference_memory'],accumulation=x['accumulation_memory'],relative_gradient_error=x['comparison']['relative_L2_error'],group_norms=x['comparison'].get('preclip_global_group_norms')) for x in records],
        qualification_checkpoint=committed,
        budget_actual_GPUh=dict(reference=budget_used('stage_A_reference_GPUh'),qualification=budget_used('stage_A_qualification_GPUh')),
        qualification_checkpoint_is_not_formal_initialization=True,
        formal_baseline_started=False,
        formal_ssf_started=False,
        user_health_scores=None,
        health_PASS=None,
        stage_B_approval='PENDING',
        stage_C_approval='PENDING_USER_HEALTH_REVIEW_AND_EXPLICIT_APPROVAL',
        actual_capacity_exception_is_stage_A_only=True,
        relevant_input_file_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'qualification/real_dmd_accumulation/complete.json',folder/'qualification_COMMITTED.json',ROOT/'reference/current_report_manifest.json',ROOT/'authorizations/approval_channels_current.json']},
    )
    path = ROOT / 'state/stage_A_measured_summary_v1.json'
    with path.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('saved:',path)


if __name__ == '__main__':
    main()
