"""Standard-library checks only: never imports a model or starts GPU work."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_code():
    source = read(REPO / 'cluster/qualified_source_manifest.json')
    overrides = {x['path']: x for x in source.get('packaging_overrides', [])}
    checked = []
    for item in source['files']:
        path = REPO / item['path']
        if not path.resolve().is_relative_to(REPO) or path.is_symlink():
            raise ValueError('Source escaped checkout: ' + item['path'])
        expected = overrides.get(item['path'], {}).get('handoff_SHA256', item['SHA256'])
        if sha(path) != expected:
            raise ValueError('Source mismatch: ' + item['path'])
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
        checked.append(item['path'])
    archived = read(REPO / 'cluster/original_run_manifest.json')
    for item in archived['files']:
        path = REPO / item['path']
        if path.is_symlink() or not path.resolve().is_relative_to(REPO):
            raise ValueError('Archive escaped checkout: ' + item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['SHA256']:
            raise ValueError('Archive mismatch: ' + item['path'])
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
    return dict(qualified_source_files=len(checked), archived_files=len(archived['files']),
                packaging_only_overrides=list(overrides), model_code_changed=False)


def check_assets(config, verify_bytes):
    rows = []
    for item in read(REPO / 'cluster/assets_manifest.json')['assets']:
        base = config.get(item['group'])
        if item.get('optional') and not base:
            rows.append(dict(group=item['group'], status='OPTIONAL_NOT_SELECTED'))
            continue
        if not base or '/REPLACE/' in str(base):
            raise ValueError('Fill the asset path for: ' + item['group'])
        root = Path(base).expanduser()
        path = root / item['relative_path'] if item['relative_path'] else root
        if not path.is_file():
            raise ValueError('Missing asset: ' + str(path))
        if path.stat().st_size != item['bytes']:
            raise ValueError('Asset size mismatch: ' + str(path))
        if verify_bytes and sha(path) != item['SHA256']:
            raise ValueError('Asset SHA256 mismatch: ' + str(path))
        rows.append(dict(group=item['group'], path=str(path), bytes=item['bytes'],
                         SHA256_verified=verify_bytes))
    return rows


def resources():
    result = {'GPU_execution': False, 'model_loaded': False}
    if shutil.which('nvidia-smi'):
        result['GPU_inventory'] = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total',
            '--format=csv,noheader,nounits'], text=True)
        result['compute_processes'] = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True)
    else:
        result['GPU_inventory'] = 'nvidia-smi unavailable'
    mem = Path('/proc/meminfo')
    if mem.exists():
        result['host_memory'] = [line for line in mem.read_text().splitlines()
                                 if line.startswith(('MemTotal:', 'MemAvailable:'))]
    fs = os.statvfs(REPO)
    result['checkout_filesystem_available_bytes'] = fs.f_bavail * fs.f_frsize
    result['user_quota'] = 'UNKNOWN'
    result['experiment_allocated_storage'] = 'NOT_INFERRED_FROM_FILESYSTEM_FREE_SPACE'
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path)
    ap.add_argument('--verify-weight-sha256', action='store_true')
    ap.add_argument('--resources', action='store_true')
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    if args.verify_weight_sha256 and not args.assets:
        ap.error('--verify-weight-sha256 requires --assets')
    report = dict(status='PASS_CODE_INTEGRITY_ONLY', code=check_code(),
                  python=sys.version, interpreter=sys.executable,
                  PYTHONNOUSERSITE=os.environ.get('PYTHONNOUSERSITE'),
                  cluster_runtime_qualified=False, training_started=False,
                  assets_checked=False)
    observed = read(REPO / 'cluster/environment_observed.json')['packages']
    report['environment'] = {}
    for name, expected in observed.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report['environment'][name] = dict(observed_original=expected, installed=actual)
    if args.assets:
        report['assets'] = check_assets(read(args.assets), args.verify_weight_sha256)
        report['assets_checked'] = True
        report['status'] = 'PASS_STATIC_INPUTS_ONLY'
    if args.resources:
        report['resources'] = resources()
    if args.json_out:
        with args.json_out.open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
