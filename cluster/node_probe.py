"""Self-contained read-only SSH node probe; no models are imported."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def probe(cfg, asset_manifest, packages, full=False):
    for key, path in cfg['assets'].items():
        if '/REPLACE/' in path:
            raise ValueError('Fill asset path: ' + key)
    assets = []
    for item in asset_manifest['assets']:
        if item.get('optional'):
            continue
        base = Path(cfg['assets'][item['group']])
        path = base / item['relative_path'] if item['relative_path'] else base
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('Missing asset or size mismatch: ' + str(path))
        if full and sha(path) != item['SHA256']:
            raise ValueError('Asset SHA256 mismatch: ' + str(path))
        assets.append(dict(path=str(path)))
    observed = packages
    versions = {}
    for name in observed:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    required = ['torch', 'torchvision', 'transformers', 'diffusers', 'flash_attn', 'omegaconf',
                'numpy', 'av', 'safetensors', 'einops', 'sentencepiece', 'lmdb', 'wandb', 'ftfy', 'easydict']
    missing = [n for n in required if versions[n] is None]
    if missing:
        raise ValueError('Missing training packages: ' + ', '.join(missing))
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=15)
    proc = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {row[0].strip() for row in csv.reader(proc.splitlines()) if row}
    gpus = []
    for row in csv.reader(raw.splitlines()):
        index, identifier, name, util, free, total = [v.strip() for v in row]
        if identifier not in busy and float(util) <= 5 and float(free) >= cfg['min_gpu_free_gib'] * 1024:
            gpus.append(dict(index=int(index), uuid=identifier, name=name, free_mib=int(free), total_mib=int(total)))
    if len(gpus) < cfg['gpus_per_node']:
        raise ValueError('Not enough idle GPUs with requested free VRAM')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024
    # Original offload working set was ~300 GiB/rank, before a reserve.
    # This conservative admission estimate is replaced by measured qualification.
    projected = cfg['gpus_per_node'] * 300 + cfg['host_reserve_gib']
    if available < projected * 1024**3:
        raise ValueError(f'Host RAM: projected {projected:g} GiB required, {available / 1024**3:.1f} GiB available; reduce gpus_per_node and increase nodes')
    output = Path(cfg['output'])
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK):
        raise ValueError('Output parent is not writable: ' + str(ancestor))
    fs = os.statvfs(ancestor)
    free = fs.f_bavail * fs.f_frsize
    if free < cfg['min_output_free_gib'] * 1024**3:
        raise ValueError(f'Persistent output needs {cfg["min_output_free_gib"]:g} GiB; available {free / 1024**3:.1f} GiB')
    # Do not accept tmpfs as the only full-checkpoint destination.
    filesystem = subprocess.check_output(['stat', '-f', '-c', '%T', str(ancestor)], text=True, timeout=10).strip()
    if filesystem in ['tmpfs', 'ramfs']:
        raise ValueError('Output must be persistent, not ' + filesystem)
    records = []
    for item in assets:
        if 'path' not in item:
            continue
        path = Path(item['path'])
        st = path.stat()
        records.append(dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns))
    return dict(hostname=socket.gethostname(), python_version=sys.version, gpus=gpus[:cfg['gpus_per_node']],
                host_available_bytes=available, output_free_bytes=free, filesystem=filesystem,
                versions=versions, driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=15).strip(),
                assets=records, full_asset_sha256=full)


if __name__ == '__main__':
    try:
        cfg,assets,packages=json.loads(sys.argv[1])
        print(json.dumps(probe(cfg,assets,packages)))
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
