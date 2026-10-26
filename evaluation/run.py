"""Evaluate local lossless videos; never loads a Wan checkpoint or generates video."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
from common import (HERE, digest, fingerprint, generation_record, gpu_preflight,
                    inspect_video, load_config, load_panel, official_entries,
                    paired_summary, read, stat_identity, write_once)


def video_paths(cfg, arm, seed, row):
    root = Path(cfg['video_root']).resolve()
    values = dict(arm=arm['name'], seed=seed, prompt_id=row['prompt_id'])
    path = (root / cfg['video_pattern'].format(**values)).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Video path escaped video_root')
    record = cfg.get('record_pattern')
    if record:
        record = (root / record.format(**values)).resolve()
        if not record.is_relative_to(root):
            raise ValueError('Record path escaped video_root')
    return path, record


def plan(cfg):
    panel = load_panel(cfg['panel'])
    counts = {d: sum(d in r['dims'] for r in panel['prompts']) for d in cfg['dimensions']}
    return dict(panel=panel['name'], n_prompts=len(panel['prompts']), seeds=cfg['seeds'],
                arms=[a['name'] for a in cfg['arms']],
                expected_existing_videos=len(panel['prompts'])*len(cfg['seeds'])*len(cfg['arms']),
                official_membership_counts=counts, output=cfg['output'],
                GPU_execution=False, model_loading=False, video_generation=False,
                official_Total_computed=False)


def prepare(cfg):
    panel = load_panel(cfg['panel'])
    out = Path(cfg['output'])
    records = []
    jobs = []
    missing = []
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            paths = {}
            inputs = []
            for row in panel['prompts']:
                path, record_path = video_paths(cfg, arm, seed, row)
                if not path.is_file() or (cfg.get('require_generation_records', True) and (not record_path or not record_path.is_file())):
                    missing.append(dict(arm=arm['name'], seed=seed, prompt_id=row['prompt_id'],
                                        video=str(path), generation_record=str(record_path)))
                    continue
                identity = dict(prompt_id=row['prompt_id'], prompt=row['prompt'], path=str(path),
                                sha256=digest(path), stat=stat_identity(path), stream=inspect_video(path))
                provenance = None
                if record_path and record_path.is_file():
                    provenance = generation_record(record_path, row['prompt'], seed, identity['sha256'], arm)
                identity['generation_provenance'] = provenance
                inputs.append(identity)
                paths[row['prompt_id']] = path
            if len(inputs) != len(panel['prompts']):
                continue
            records.append(dict(arm=arm['name'], seed=seed, videos=inputs))
            for dim in cfg['dimensions']:
                entries = official_entries(panel['prompts'], dim, paths)
                if not entries:
                    continue
                folder = out / 'jobs' / arm['name'] / str(seed) / dim
                inp = dict(schema_version=1, arm=arm, seed=seed, dimension=dim,
                           contract_sha256=digest(HERE/'contract.json'),
                           metric_assets_sha256=digest(HERE/'metric_assets.json'),
                           panel_sha256=digest(cfg['panel']), config_hash=fingerprint(cfg),
                           vbench_cache=cfg['vbench_cache'], entries=entries,
                           inputs=[x for x in inputs if any(e['prompt_en']==x['prompt'] for e in entries)])
                jobs.append(dict(folder=str(folder), input=inp))
    if missing:
        raise ValueError('MISSING_INPUTS (%d): %s' % (len(missing), json.dumps(missing[:12], ensure_ascii=False)))
    pairing = []
    for seed in cfg['seeds']:
        rows_by_arm = {r['arm']: {v['prompt_id']: v for v in r['videos']} for r in records if r['seed']==seed}
        for row in panel['prompts']:
            traces = [rows_by_arm[a['name']][row['prompt_id']]['generation_provenance'] for a in cfg['arms']]
            verified = all(traces)
            if verified:
                if len({(r['initial_noise_sha256'], r['renoise_fingerprint']) for r in traces}) != 1:
                    raise ValueError('PAIRED_NOISE_MISMATCH: ' + row['prompt_id'])
            pairing.append(dict(seed=seed, prompt_id=row['prompt_id'],
                                status='VERIFIED_INITIAL_AND_ALL_RENOISE' if verified else 'UNVERIFIED'))
    manifest = dict(schema_version=1, config=cfg, config_hash=fingerprint(cfg), plan=plan(cfg),
                    contract_sha256=digest(HERE/'contract.json'), panel_sha256=digest(cfg['panel']),
                    records=records, pairing=pairing,
                    jobs=[dict(folder=j['folder'], input_hash=fingerprint(j['input'])) for j in jobs],
                    checkpoint_loading=False, generated_videos=0)
    # Validate every input first; no partial manifest is promoted to a ready run.
    write_once(out/'manifest.json', manifest)
    for job in jobs:
        write_once(Path(job['folder'])/'input.json', job['input'])
    return manifest


def summarize(manifest):
    cfg = manifest['config']
    out = Path(cfg['output'])
    per_video = []
    missing_jobs = []
    aggregates = []
    for job in manifest['jobs']:
        folder = Path(job['folder'])
        done = folder/'COMPLETED.json'
        if not done.exists():
            missing_jobs.append(str(folder))
            continue
        prov = read(done)
        inp = read(folder/'input.json')
        if prov['input_hash'] != job['input_hash'] or prov['input_hash'] != fingerprint(inp):
            raise ValueError('Completed result uses another input manifest')
        if digest(folder/'result.json') != prov['result_sha256']:
            raise ValueError('Completed score hash mismatch')
        res = read(folder/'result.json')
        dim = inp['dimension']
        aggregates.append(dict(arm=inp['arm']['name'], seed=inp['seed'], dimension=dim,
                               native_aggregate=res['aggregate_native'], points=res['aggregate_native']*100,
                               eligible=res['eligible'], retained=len(res['per_video']),
                               missing_prompt_ids=res['missing_prompt_ids']))
        per_video.extend(dict(arm=inp['arm']['name'], seed=inp['seed'], dimension=dim, **r) for r in res['per_video'])
    lookup = {(r['arm'], r['seed'], r['dimension'], r['prompt_id']): r['score_points'] for r in per_video}
    panel = load_panel(cfg['panel'])
    comparisons = []
    ref = cfg['arms'][0]['name']
    for arm in cfg['arms'][1:]:
        for dim in cfg['dimensions']:
            eligible = [p for p in panel['prompts'] if dim in p['dims']]
            differences = []
            per_seed = {}
            for seed in cfg['seeds']:
                pairs = [(lookup[(ref,seed,dim,p['prompt_id'])], lookup[(arm['name'],seed,dim,p['prompt_id'])])
                         for p in eligible if (ref,seed,dim,p['prompt_id']) in lookup
                         and (arm['name'],seed,dim,p['prompt_id']) in lookup]
                per_seed[str(seed)] = dict(paired_n=len(pairs), reference_mean=statistics.mean(a for a,b in pairs) if pairs else None,
                    arm_mean=statistics.mean(b for a,b in pairs) if pairs else None,
                    mean_delta=statistics.mean(b-a for a,b in pairs) if pairs else None)
            for p in eligible:
                keys = [(name,seed,dim,p['prompt_id']) for name in [ref,arm['name']] for seed in cfg['seeds']]
                if all(k in lookup for k in keys):
                    differences.append(statistics.mean(lookup[(arm['name'],s,dim,p['prompt_id'])]-lookup[(ref,s,dim,p['prompt_id'])] for s in cfg['seeds']))
            comparisons.append(dict(reference=ref, arm=arm['name'], dimension=dim, eligible_prompts=len(eligible),
                all_seed_paired_prompts=len(differences), per_seed=per_seed,
                statistics=paired_summary(differences, cfg.get('win_tie_tolerance',1e-8), cfg.get('bootstrap_resamples',10000))))
    result = dict(status='COMPLETE' if not missing_jobs else 'PARTIAL_NOT_A_COMPLETE_EVALUATION',
                  input_provenance='VERIFIED_RECORDS_AND_NOISE' if all(p['status'].startswith('VERIFIED') for p in manifest['pairing']) else 'UNVERIFIED',
                  missing_jobs=missing_jobs, aggregates=aggregates, comparisons=comparisons,
                  official_Total_computed=False,
                  notes=['Dimension scores only; no subset Quality/Semantic/Total.',
                         'Prompt is the analysis unit; seeds averaged within prompt.',
                         'Sign tests/bootstrap are descriptive, uncorrected for multiplicity; no success gate.',
                         'Object detectors may omit videos; missing entries remain missing, never zero-filled.',
                         'One checkpoint pair cannot establish training-seed robustness.'])
    return result, per_video


def export_report(manifest):
    result, rows = summarize(manifest)
    # Reports are versioned: a partial report never overwrites prior evidence.
    import datetime
    folder = Path(manifest['config']['output'])/'reports'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    write_once(folder/'summary.json', result)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'per_video.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arm','seed','dimension','prompt_id','native_score','score_points'])
        writer.writeheader(); writer.writerows(rows)
    text = ['# Local video evaluation', '', result['status'], '',
            'Per-dimension raw score ×100. No Official VBench Total.', '',
            '| Dimension | Comparison | Paired prompts | Mean delta | Bootstrap 95% CI |',
            '|---|---|---:|---:|---|']
    for c in result['comparisons']:
        s = c['statistics']
        text.append('| %s | %s − %s | %d | %s | %s |' % (c['dimension'],c['arm'],c['reference'],
                    s['n_prompts'],str(s.get('mean')),str(s.get('bootstrap_95_ci'))))
    text += ['', 'Input provenance: '+result['input_provenance'], '', *result['notes']]
    (folder/'report.md').write_text('\n'.join(text)+'\n')
    print(str(folder/'report.md'))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['plan','prepare','run','report'])
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gpu', help='Physical GPU index or full UUID; required only for run')
    a = ap.parse_args()
    cfg = load_config(a.config, require_paths=a.command != 'plan')
    if a.command == 'plan':
        print(json.dumps(plan(cfg), indent=2)); return
    if a.command == 'report':
        manifest = read(Path(cfg['output'])/'manifest.json')
        if manifest['config_hash'] != fingerprint(cfg):
            raise ValueError('Config changed; preserve run and use another output directory')
        export_report(manifest); return
    manifest = prepare(cfg)
    if a.command == 'prepare':
        print(json.dumps(manifest['plan'], indent=2)); return
    if not a.gpu:
        ap.error('run requires --gpu; no GPU is auto-selected or assumed idle')
    for job in manifest['jobs']:
        folder = Path(job['folder'])
        if (folder/'COMPLETED.json').exists():
            previous = read(folder/'COMPLETED.json')
            if previous['input_hash'] != job['input_hash'] or digest(folder/'result.json') != previous['result_sha256']:
                raise ValueError('Completed output changed')
            continue
        if (folder/'result.json').exists() or (folder/'native_result.json').exists():
            raise ValueError('Preserve partial metric output for review; use a new output directory')
        preflight = gpu_preflight(a.gpu)
        env = dict(os.environ, PYTHONNOUSERSITE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=preflight['uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # A new process per dimension releases all metric VRAM. No controller,
        # background waiting, model reload loop, training, or checkpoint access.
        with (folder/'stdout.log').open('a') as log:
            completed = subprocess.run([cfg['metric_python'],'-B','-X','faulthandler','-u',
                str(HERE/'score_worker.py'),'--input',str(folder/'input.json')], env=env,
                stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Metric failed; full traceback in '+str(folder/'stdout.log'))
    export_report(manifest)


if __name__ == '__main__':
    main()
