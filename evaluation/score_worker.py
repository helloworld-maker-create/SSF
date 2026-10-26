"""One offline official VBench dimension. No Wan/T5/teacher/checkpoint loading."""
import argparse
import faulthandler
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from common import (HERE, digest, expand_assets, fingerprint, gpu_preflight,
                    make_action_aliases, read, score_points, stat_identity, write_once)


def block_network_and_downloads():
    """Fail closed rather than let a metric auto-fetch missing model assets."""
    import urllib.request
    def blocked(*a, **k):
        raise RuntimeError('Offline evaluation: downloads/network access are forbidden')
    urllib.request.urlopen = blocked
    urllib.request.urlretrieve = blocked
    try:
        import requests
        requests.sessions.Session.request = blocked
    except ImportError:
        pass
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    def checked_connect(self, address):
        if self.family in [socket.AF_INET, socket.AF_INET6]:
            return blocked()
        return connect(self, address)
    def checked_connect_ex(self, address):
        if self.family in [socket.AF_INET, socket.AF_INET6]:
            return blocked()
        return connect_ex(self, address)
    socket.socket.connect = checked_connect
    socket.socket.connect_ex = checked_connect_ex
    run = subprocess.run
    def checked_run(command, *a, **k):
        if isinstance(command, str) or k.get('shell'):
            raise RuntimeError('Offline metric subprocess requires an explicit argv list')
        name = Path(str(command[0])).name
        if name in ['wget','curl','rm','unlink'] or (name == 'git' and 'clone' in command):
            raise RuntimeError('Forbidden metric subprocess: '+name)
        return run(command, *a, **k)
    subprocess.run = checked_run


def validate_assets(dim, cache, vbench_dir):
    spec = read(HERE/'metric_assets.json')
    raw = spec['submodules'][dim]
    templates = []
    def visit(x):
        if isinstance(x, str) and ('${CACHE}' in x or '${VBENCH}' in x):
            templates.append(x)
        elif isinstance(x, dict):
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(raw)
    checked = []
    for template in templates:
        path = Path(expand_assets(template, cache, vbench_dir))
        if not path.exists():
            raise FileNotFoundError('Provide local metric asset; no download: '+str(path))
        registered = [k for k in spec['verified_assets'] if k == template or k.startswith(template.rstrip('/')+'/')]
        if not registered:
            raise ValueError('Unregistered metric asset: '+str(path))
        for key in registered:
            expected = spec['verified_assets'][key]
            p = Path(expand_assets(key, cache, vbench_dir))
            if not p.is_file() or p.stat().st_size != expected['bytes'] or digest(p) != expected['sha256']:
                raise ValueError('Metric asset identity changed: '+str(p))
            checked.append(dict(path=str(p), **expected))
    return expand_assets(raw, cache, vbench_dir), checked


def canonical_result(aggregate, rows, inp, entries):
    bypath = {str(Path(e['video_list'][0]).resolve()):
              next(v for v in inp['inputs'] if v['prompt']==e['prompt_en'])['prompt_id'] for e in entries}
    expected = set(bypath)
    seen = set()
    values = []
    for row in rows:
        path = str(Path(row['video_path']).resolve())
        if path not in expected or path in seen:
            raise ValueError('Duplicate/unregistered scored video')
        seen.add(path)
        native = float(row['video_results'])
        values.append(dict(prompt_id=bypath[path], native_score=native,
                           score_points=score_points(inp['dimension'], native)))
    if inp['dimension'] not in ['object_class','multiple_objects'] and seen != expected:
        raise ValueError('Metric omitted eligible videos')
    if not math.isfinite(float(aggregate)):
        raise ValueError('Nonfinite aggregate')
    return dict(aggregate_native=float(aggregate), per_video=values, eligible=len(expected),
                missing_prompt_ids=sorted(bypath[p] for p in expected-seen),
                units='unmodified native aggregate; per-video unit score times100',
                official_Total_computed=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    a = ap.parse_args()
    folder = a.input.resolve().parent
    inp = read(a.input)
    if (folder/'COMPLETED.json').exists() or (folder/'result.json').exists():
        raise ValueError('Preserve existing result; resume with run.py or review partial output')
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise ValueError('Set PYTHONNOUSERSITE=1')
    for file, key in [('contract.json','contract_sha256'),('metric_assets.json','metric_assets_sha256')]:
        if digest(HERE/file) != inp[key]:
            raise ValueError('Evaluation contract/assets changed')
    contract = read(HERE/'contract.json')
    for item in inp['inputs']:
        if stat_identity(item['path']) != item['stat']:
            raise ValueError('Video changed after prepare: '+item['path'])
    selected = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected.startswith('GPU-') or ',' in selected:
        raise ValueError('Launch through run.py with one preflighted physical GPU/UUID')
    resources = gpu_preflight(selected)
    import torch
    import numpy as np
    block_network_and_downloads()
    try:
        import vbench
        if importlib.metadata.version('vbench') != contract['expected_vbench_version']:
            raise ValueError('VBench version differs from frozen evaluation; no automatic upgrade')
        dim = inp['dimension']
        vbench_dir = Path(vbench.__file__).resolve().parent
        source = vbench_dir/(dim+'.py')
        if digest(source) != contract['metric_module_sha256'][dim+'.py']:
            raise ValueError('VBench metric source differs from the validated version')
        sub, verified = validate_assets(dim, inp['vbench_cache'], vbench_dir)
        # Recheck immediately before metric model loading; never touch another job.
        gpu_preflight(selected)
        torch.cuda.set_device(0)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one visible CUDA GPU')
        torch.set_grad_enabled(False)
        random.seed(contract['metric_seed']); np.random.seed(contract['metric_seed'])
        torch.manual_seed(contract['metric_seed']); torch.cuda.manual_seed_all(contract['metric_seed'])
        entries = inp['entries']
        if dim == 'human_action':
            entries = make_action_aliases(folder/'action_aliases', entries)
        metadata_path = folder/'eligible_full_info.json'
        write_once(metadata_path, entries)
        module = importlib.import_module('vbench.'+dim)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        aggregate, rows = getattr(module, 'compute_'+dim)(str(metadata_path), torch.device('cuda:0'), sub,
            imaging_quality_preprocessing_mode=contract['imaging_quality_preprocessing_mode'])
        torch.cuda.synchronize()
        result = canonical_result(aggregate, rows, inp, entries)
        # Save original scorer records as well as normalized per-video records.
        def plain(x):
            if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
            if isinstance(x, (list,tuple)): return [plain(v) for v in x]
            if isinstance(x, np.generic): return x.item()
            return x
        write_once(folder/'native_result.json', {dim:plain([aggregate,rows])})
        write_once(folder/'result.json', result)
        write_once(folder/'COMPLETED.json', dict(status='COMPLETE_DIMENSION_ONLY',
            input_hash=fingerprint(inp), result_sha256=digest(folder/'result.json'),
            native_result_sha256=digest(folder/'native_result.json'),
            metric_source_sha256=digest(source), verified_metric_assets=verified,
            evaluation_script_sha256={p.name:digest(p) for p in [HERE/'run.py',HERE/'common.py',HERE/'score_worker.py']},
            seconds=time.perf_counter()-start, GPU=resources, python=sys.version,
            runtime=dict(torch=str(torch.__version__), numpy=np.__version__, vbench=importlib.metadata.version('vbench')),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_checkpoint_loaded=True, no_training=True, no_network=True))
        print('COMPLETE_DIMENSION', inp['arm']['name'], inp['seed'], dim, float(aggregate), flush=True)
    finally:
        # Metrics normally run without a process group. Clean up if one exists.
        import torch.distributed as dist
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
