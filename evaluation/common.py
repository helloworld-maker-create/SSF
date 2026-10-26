"""Portable evaluation metadata and reporting. Standard library only."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_once(path, value):
    """No overwrite of different evidence, including partial/failed runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing evidence differs; use a new output directory: ' + str(path))
        return
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path, require_paths=True):
    path = Path(path).resolve()
    cfg = read(path)
    cfg['panel'] = str((HERE / cfg['panel']).resolve()) if not Path(cfg['panel']).is_absolute() else cfg['panel']
    for key in ['video_root', 'output', 'metric_python', 'vbench_cache']:
        value = cfg[key]
        if not value or (require_paths and '/REPLACE/' in value) or not Path(value).is_absolute():
            raise ValueError('Set an absolute local path for ' + key)
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    names = [a['name'] for a in cfg['arms']]
    if len(set(names)) != len(names) or len(names) < 1:
        raise ValueError('Unique arm names required')
    if any(not n or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in n) for n in names):
        raise ValueError('Unsafe arm name')
    if len({a['weight_kind'] for a in cfg['arms']}) != 1:
        raise ValueError('Compare raw with raw or EMA with EMA')
    for a in cfg['arms']:
        sha = a.get('checkpoint_sha256')
        if sha is not None and (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Invalid checkpoint SHA256')
    if not cfg['dimensions'] or len(set(cfg['dimensions'])) != len(cfg['dimensions']):
        raise ValueError('Dimensions must be nonempty and unique')
    if not set(cfg['dimensions']) <= set(read(HERE / 'metric_assets.json')['submodules']):
        raise ValueError('Unknown VBench dimension')
    return cfg


def load_panel(path):
    data = read(path)
    rows = data['prompts']
    if len({r['prompt'] for r in rows}) != len(rows) or len({r['prompt_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate panel prompt/id')
    for r in rows:
        expected = 'p%04d-' % r['source_index'] + hashlib.sha256(r['prompt'].encode()).hexdigest()[:12]
        if r['prompt_id'] != expected:
            raise ValueError('Prompt ID mismatch')
    return data


def official_entries(prompts, dimension, paths):
    contract = read(HERE / 'contract.json')
    meta = HERE / contract['official_metadata']
    if digest(meta) != contract['official_metadata_sha256']:
        raise ValueError('Official metadata changed')
    by_text = {r['prompt']: r for r in prompts}
    result = []
    for entry in read(meta):
        text = entry['prompt_en']
        if text in by_text and dimension in entry['dimension']:
            result.append({**entry, 'video_list': [str(paths[by_text[text]['prompt_id']])]})
    expected = {r['prompt_id'] for r in prompts if dimension in r['dims']}
    seen = [by_text[e['prompt_en']]['prompt_id'] for e in result]
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError('Official dimension membership mismatch: ' + dimension)
    return result


def stat_identity(path):
    st = Path(path).stat()
    return dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)


def inspect_video(path):
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames',
           '-of', 'json', str(path)]
    info = json.loads(subprocess.check_output(cmd, text=True))['streams']
    if len(info) != 1:
        raise ValueError('Expected one video stream: ' + str(path))
    info = info[0]
    a, b = info['avg_frame_rate'].split('/')
    actual = [int(info['width']), int(info['height']), int(info['nb_read_frames']), float(a) / float(b)]
    if actual != [832, 480, 81, 16.0]:
        raise ValueError('Video contract mismatch: ' + str((path, actual)))
    if info['pix_fmt'] not in ['gbrp', 'rgb24', 'bgr0', 'rgba', 'bgra']:
        raise ValueError('Use lossless RGB scoring originals, not YUV preview: ' + str(path))
    return info


def generation_record(path, prompt, seed, video_sha, arm):
    r = read(path)
    row = r.get('matrix_row', r)
    if r.get('prompt', row.get('prompt')) != prompt or int(r.get('seed', row.get('seed', -1))) != seed:
        raise ValueError('Generation record prompt/seed mismatch')
    video = r['video']
    if not video.get('lossless') or video.get('SHA256', video.get('sha256')) != video_sha:
        raise ValueError('Generation record video identity/lossless mismatch')
    if video.get('RGB_SHA256') is None:
        raise ValueError('Lossless roundtrip RGB hash missing')
    ck = r.get('checkpoint_SHA256', r.get('checkpoint', {}).get('SHA256'))
    if arm.get('checkpoint_sha256') is None or ck != arm['checkpoint_sha256']:
        raise ValueError('Declare the generation checkpoint SHA256 in config')
    if str(r.get('weight', row.get('weight', ''))).lower() != arm['weight_kind'].lower():
        raise ValueError('Generation record raw/EMA choice mismatch')
    initial = r.get('initial_noise_SHA256')
    renoise = r.get('renoise_trace')
    if not initial or not isinstance(renoise, list) or len(renoise) != 21:
        raise ValueError('Initial noise/all 21 renoise traces required')
    if not r.get('latent_finite', r.get('finite', False)):
        raise ValueError('Latent finite evidence missing')
    return dict(record_sha256=digest(path), initial_noise_sha256=initial,
                renoise_fingerprint=fingerprint(renoise), checkpoint_sha256=ck,
                config_sha256=r.get('config_SHA256'), lossless_rgb_sha256=video['RGB_SHA256'])


def gpu_preflight(selection, min_free_mib=12288):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                     '--format=csv,noheader,nounits'], text=True)
    found = []
    for line in raw.splitlines():
        index, uuid, name, util, free, total = [s.strip() for s in line.split(',')]
        if selection not in [index, uuid]:
            continue
        foreign = [s for s in procs.splitlines() if s.split(',')[0].strip() == uuid
                   and int(s.split(',')[1].strip()) != os.getpid()]
        if foreign or float(util) > 5 or float(free) < min_free_mib:
            raise RuntimeError('BLOCKED_RESOURCE: chosen GPU is not idle/free: ' + index)
        found.append(dict(index=int(index), uuid=uuid, name=name, utilization=float(util),
                          free_mib=float(free), total_mib=float(total), foreign_processes=foreign))
    if len(found) != 1:
        raise RuntimeError('Choose exactly one physical GPU index or full UUID')
    return found[0]


def score_points(dimension, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite metric')
    # MUSIQ per-video records are 0..100, its aggregate is already /100.
    return value if dimension == 'imaging_quality' else value * 100.0


def paired_summary(differences, tolerance=1e-8, resamples=10000):
    if not differences:
        return dict(status='NO_PAIRED_DATA', n_prompts=0)
    if tolerance < 0 or resamples < 10000:
        raise ValueError('Use nonnegative tie tolerance and >=10000 bootstrap draws')
    x = list(differences)
    if not all(math.isfinite(v) for v in x):
        raise ValueError('Nonfinite paired differences')
    n = len(x)
    wins = sum(v > tolerance for v in x)
    losses = sum(v < -tolerance for v in x)
    k = wins + losses
    p = min(1.0, 2 * sum(math.comb(k, i) for i in range(min(wins, losses) + 1)) / 2**k) if k else 1.0
    rng = random.Random(2026)
    means = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return dict(n_prompts=n, mean=statistics.mean(x), median=statistics.median(x),
                standard_error=statistics.stdev(x)/math.sqrt(n) if n > 1 else None,
                bootstrap_95_ci=[means[int(.025*resamples)], means[int(.975*resamples)]],
                wins=wins, ties=n-wins-losses, losses=losses, sign_test_two_sided=p,
                tie_tolerance_points=tolerance, bootstrap_resamples=resamples,
                estimand='mean prompt-level difference after averaging inference seeds',
                limitation='Descriptive prompt bootstrap; no training-seed inference or noninferiority claim')


def expand_assets(value, cache, vbench):
    if isinstance(value, str):
        return value.replace('${CACHE}', str(cache)).replace('${VBENCH}', str(vbench))
    if isinstance(value, dict):
        return {k: expand_assets(v, cache, vbench) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_assets(v, cache, vbench) for v in value]
    return value


def make_action_aliases(folder, entries):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = []
    for e in entries:
        prompt = e['prompt_en']
        name = prompt + '-0.mp4'
        if not prompt.startswith('A person is ') or Path(name).name != name or len(name.encode()) > 255:
            raise ValueError('Unsupported official human-action filename')
        source = Path(e['video_list'][0]).resolve()
        target = folder / name
        if not target.exists():
            # If filesystems differ, stop; no silent copy/re-encoding fallback.
            os.link(source, target)
        if target.is_symlink() or not os.path.samefile(source, target):
            raise ValueError('Human-action alias is not the exact input inode')
        parsed = target.name.lower().split('-')[0].split('person is ')[-1].split('_')[0]
        if parsed != prompt.lower().split('person is ')[-1]:
            raise ValueError('Human-action filename label mismatch')
        result.append({**e, 'video_list': [str(target)]})
    return result
