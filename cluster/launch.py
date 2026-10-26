"""One command: select an idle node, prepare source, and start baseline training.

The controller uses only the standard library. --dry-run never contacts nodes.
SSH nodes need Python and asset paths; source is synchronized automatically. Multi-node jobs also
need a shared persistent output directory; the controller verifies it live.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'cluster'))
from preflight import check_code, sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config(path):
    default = read(REPO / 'cluster/site.example.json')
    supplied = read(path)
    unknown = set(supplied) - set(default)
    if unknown:
        raise ValueError('Unknown site settings: ' + ', '.join(sorted(unknown)))
    cfg = {**default, **supplied}
    for key in ['nodes', 'gpus_per_node', 'master_port']:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    world = cfg['nodes'] * cfg['gpus_per_node']
    if world < 2 or world > 64 or 64 % world:
        raise ValueError('B64 requires 2/4/8/16/32/64 ranks; nodes * gpus_per_node must divide 64')
    if cfg['master_port'] > 65535:
        raise ValueError('Invalid master_port')
    if cfg['transport'] not in ['ssh', 'local']:
        raise ValueError('transport must be ssh or local')
    if cfg['transport'] == 'local' and cfg['nodes'] != 1:
        raise ValueError('local transport supports one node')
    if cfg['ssh_user'] is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', cfg['ssh_user']):
        raise ValueError('Invalid ssh_user')
    for key in ['min_gpu_free_gib', 'host_reserve_gib', 'min_output_free_gib', 'wall_hours']:
        if type(cfg[key]) not in [int, float] or not 0 < cfg[key] < float('inf'):
            raise ValueError(key + ' must be finite and positive')
    if not isinstance(cfg['environment'], dict) or not all(
            re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k) and isinstance(v, str)
            and k.startswith(('NCCL_', 'GLOO_', 'TORCH_', 'OMP_', 'OPENBLAS_'))
            for k, v in cfg['environment'].items()):
        raise ValueError('environment only accepts explicit NCCL/GLOO/TORCH/OMP/OPENBLAS string settings')
    if not isinstance(cfg['assets'], dict) or set(cfg['assets']) != {'wan_1_3b', 'teacher_14b', 'ode_init', 'training_prompts'}:
        raise ValueError('Provide the four required asset paths; legacy full_resume is not a new-cluster initializer')
    for key in ['python', 'repo', 'output', 'inventory', 'ssh_identity', 'ssf_review']:
        if cfg[key] is not None:
            cfg[key] = os.path.expanduser(cfg[key])
    cfg['assets'] = {k: os.path.expanduser(v) for k, v in cfg['assets'].items()}
    if not all(Path(v).is_absolute() for v in cfg['assets'].values()):
        raise ValueError('Asset paths must be absolute on every selected node')
    for key in ['python', 'repo', 'output']:
        if not Path(cfg[key]).is_absolute():
            raise ValueError(key + ' must be an absolute path')
    return cfg


def init_config(path):
    cfg = read(REPO / 'cluster/site.example.json')
    cfg['repo'] = str(REPO)
    cfg['ssh_user'] = getpass.getuser()
    key = Path.home() / '.ssh/id_ed25519_h200'
    if key.is_file():
        cfg['ssh_identity'] = str(key)
    candidates = [Path('/home/zkln/miniconda3/envs/fastvideo/bin/python'), Path(sys.executable)]
    cfg['python'] = str(next(p for p in candidates if p.is_file()))
    # Discover paths only. Byte verification still runs on every selected node.
    roots = [Path('/mnt/nvme3n1/wubin/t2v_original_baseline_assets')]
    for root in roots:
        if (root / 'ode_init.pt').is_file():
            cfg['assets'] = dict(wan_1_3b=str(root / 'Wan2.1-T2V-1.3B'),
                                 teacher_14b=str(root / 'Wan2.1-T2V-14B'),
                                 ode_init=str(root / 'ode_init.pt'),
                                 training_prompts=str(root / 'prompts/vidprom_filtered_extended.txt'))
            break
    cfg['output'] = str(Path.home() / 'ssf_runs')
    write(path, cfg)
    print('Created ' + str(path) + '; review persistent output and asset/Python paths before launch.')


def hosts(cfg):
    if cfg['transport'] == 'local':
        return ['localhost']
    values = cfg['hosts']
    if not values:
        path = Path(cfg['inventory'])
        if not path.is_file():
            raise ValueError('Inventory absent; run gpuall --list or set hosts in site config')
        values = [line.split('|')[-1].strip() for line in path.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
    if not isinstance(values, list) or not all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_.:-]+', v) and not v.startswith('-') for v in values):
        raise ValueError('Invalid host list')
    if len(values) != len(set(values)):
        raise ValueError('Duplicate hosts')
    if len(values) < cfg['nodes']:
        raise ValueError('Inventory has fewer hosts than requested nodes')
    return values


def remote(cfg, host, argv):
    if cfg['transport'] == 'local':
        return argv
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
    if cfg['ssh_identity']:
        command += ['-o', 'IdentitiesOnly=yes', '-i', cfg['ssh_identity']]
    target = host if not cfg['ssh_user'] else cfg['ssh_user'] + '@' + host
    return command + [target, 'exec ' + shlex.join(argv)]


def capture(cfg, host, argv, timeout=30, input_text=None):
    result = subprocess.run(remote(cfg, host, argv), input=input_text, text=True, capture_output=True, timeout=timeout, check=True)
    return json.loads(result.stdout)


def code_manifest():
    paths = set(p['path'] for p in read(REPO / 'cluster/qualified_source_manifest.json')['files'])
    paths.update(p['path'] for p in read(REPO / 'cluster/original_run_manifest.json')['files'])
    for folder in ['cluster', 'evaluation']:
        paths.update(str(p.relative_to(REPO)) for p in (REPO / folder).rglob('*')
                     if p.is_file() and p.suffix in ['.py', '.json', '.csv', '.yaml']
                     and '__pycache__' not in p.parts and 'run_receipts' not in p.parts
                     and '.local.' not in p.name)
    paths.update(['launch.sh', 'cluster/supervise.py'])
    # Local path configs and run outputs never enter scientific provenance.
    return {p: sha(REPO / p) for p in sorted(paths)}


def deploy(cfg, selected, code):
    """Copy only reviewed small source files into a versioned deployment.

    No --delete, weights, credentials, Git directory or site.local.json. Each
    source version has its own directory, so other worktrees are preserved.
    """
    if cfg['transport'] == 'local':
        if Path(cfg['repo']).resolve() != REPO:
            raise ValueError('local repo must be this checkout')
        return
    if not shutil.which('rsync'):
        raise ValueError('SSH source preparation needs rsync on controller and nodes')
    with tempfile.TemporaryDirectory(prefix='ssf-source-') as folder:
        filelist = Path(folder) / 'files'
        filelist.write_text('\n'.join(code) + '\n')
        for host, _ in selected:
            mkdir = [cfg['python'], '-c',
                     'from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)', cfg['repo']]
            subprocess.run(remote(cfg, host, mkdir), check=True, timeout=30)
            # Obtain the SSH options without the target/remote command.
            ssh = remote(cfg, host, ['true'])[:-2]
            target = (cfg['ssh_user'] + '@' if cfg['ssh_user'] else '') + host
            subprocess.run(['rsync', '-a', '--protect-args', '--files-from=' + str(filelist),
                            '-e', shlex.join(ssh), str(REPO) + '/', target + ':' + cfg['repo'] + '/'],
                           check=True, timeout=120)


def node_probe(cfg, full=False):
    from node_probe import probe
    check_code()
    report = probe(cfg, read(REPO / 'cluster/assets_manifest.json'),
                   read(REPO / 'cluster/environment_observed.json')['packages'], full)
    report['code'] = code_manifest()
    return report


def probe_command(cfg, full=False):
    if full:
        return [cfg['python'], '-B', cfg['repo'] + '/cluster/launch.py', '--node-probe',
                json.dumps(cfg), '--verify-assets']
    payload = [cfg, read(REPO / 'cluster/assets_manifest.json'),
               read(REPO / 'cluster/environment_observed.json')['packages']]
    return [cfg['python'], '-B', '-c', (REPO / 'cluster/node_probe.py').read_text(), json.dumps(payload)]


def select_nodes(cfg):
    def probe(host):
        try:
            report = capture(cfg, host, probe_command(cfg))
            return host, report, None
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, 'stderr', None) or str(exc)
            return host, None, detail.strip()[-1200:]
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(probe, hosts(cfg)))
    ready = []
    rejected = []
    for host, report, error in results:
        if error:
            rejected.append(host + ': ' + error)
        else:
            ready.append((host, report))
    ready.sort(key=lambda row: row[1]['output_free_bytes'], reverse=True)
    chosen = ready[:cfg['nodes']]
    if len(chosen) < cfg['nodes']:
        for error in rejected:
            print(error, file=sys.stderr)
        raise ValueError('Not enough prepared idle nodes; no GPU workers launched. Check the per-node diagnostics above.')
    print('Selected ' + ', '.join(h for h, _ in chosen) + f'; skipped {len(rejected)} unavailable nodes', file=sys.stderr)
    if len({r['hostname'] for _, r in chosen}) != len(chosen):
        raise ValueError('Inventory aliases resolve to the same node')
    signatures = {digest(dict(versions=r['versions'], python=r['python_version'], driver=r['driver'],
                              gpu_models=[g['name'] for g in r['gpus']])) for _, r in chosen}
    if len(signatures) != 1:
        raise ValueError('Selected node environments/GPU models differ')
    return chosen


def worker_env(cfg, run, node):
    return dict(PYTHONNOUSERSITE='1', PYTHONOPTIMIZE='0', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in node['report']['gpus']),
                SSF_RUN_ROOT=run['root'], SSF_SITE_JSON=json.dumps(cfg),
                SSF_WAN_1_3B=cfg['assets']['wan_1_3b'], SSF_RUN_FINGERPRINT=run['fingerprint'],
                PYTHONPATH=cfg['repo'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='4',
                **cfg['environment'])


def training_command(cfg, run, node_rank, mode, arm, resume=None):
    node = run['nodes'][node_rank]
    env = worker_env(cfg, run, node)
    argv = ['env'] + [key + '=' + value for key, value in env.items()]
    argv += [cfg['python'], '-B', cfg['repo'] + '/cluster/supervise.py',
             cfg['python'], '-B', '-m', 'torch.distributed.run', '--nnodes=' + str(cfg['nodes']),
             '--nproc_per_node=' + str(cfg['gpus_per_node']), '--node_rank=' + str(node_rank),
             '--master_addr=' + (run['nodes'][0]['host'] if cfg['transport'] == 'ssh' else '127.0.0.1'),
             '--master_port=' + str(cfg['master_port']), '--max_restarts=0',
             cfg['repo'] + '/cluster/runtime/worker.py', '--mode', mode, '--arm', arm]
    if resume:
        argv += ['--resume', resume]
    if mode == 'train':
        argv += ['--attempt', str(run['attempt'])]
    return remote(cfg, node['host'], argv)


def run_group(cfg, run, mode, arm, resume=None):
    children = []
    failure = None
    try:
        for rank, node in enumerate(run['nodes']):
            # SSH exec keeps the session attached; worker handles SIGHUP as stop.
            child = subprocess.Popen(training_command(cfg, run, rank, mode, arm, resume), stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
        while any(c.poll() is None for c in children):
            if any(c.poll() not in [None, 0] for c in children):
                exits = [(run['nodes'][i]['host'], c.poll()) for i, c in enumerate(children) if c.poll() not in [None, 0]]
                raise RuntimeError(mode + ' worker failed: ' + str(exits))
            time.sleep(1)
        if any(c.returncode != 0 for c in children):
            raise RuntimeError(mode + ' failed')
    except (RuntimeError, OSError) as exc:
        failure = exc
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    if failure:
        report = dict(run_id=run['run_id'], output=run['root'], mode=mode, arm=arm,
                      error=str(failure), logs=[], time=time.time())
        try:
            logs = remote_action(cfg, run['nodes'][0]['host'], 'error_logs',
                                 dict(config=cfg, run=run, mode=mode, arm=arm))
            report['logs'] = logs['logs']
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report['log_fetch_error'] = str(exc)
        local = REPO / 'cluster/run_receipts' / (run['run_id'] + f'.error-{time.time_ns()}.json')
        write(local, report)
        for index, log in enumerate(report['logs']):
            print('\n' + log['path'], file=sys.stderr)
            if index == 0:
                print('\n'.join(log['tail'].splitlines()[-12:]), file=sys.stderr)
        raise RuntimeError(str(failure) + '\nError report: ' + str(local)) from failure


def setup_run(cfg, run):
    root = Path(run['root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'run.json', run)
    for directory in ['logs', 'state', 'source_manifest', 'reference']:
        (root / directory).mkdir(exist_ok=True)
    write(root / 'source_manifest/source_files.json', {'files': [{'path': p, 'SHA256': v} for p, v in run['code'].items()]})
    write(root / 'source_manifest/formal_code_manifest.json', {'files': [{'name': p, 'SHA256': v} for p, v in run['code'].items()]})
    ode = next(a for a in read(REPO / 'cluster/assets_manifest.json')['assets'] if a['group'] == 'ode_init')
    write(root / 'reference/ODE_input_checkpoint.json', {'SHA256': ode['SHA256']})
    if run['context']['arm'] == 'ssf':
        write(root / 'ssf_review.json', run['context']['review'])


def node_setup(cfg, run, host, existing=False):
    root = Path(run['root'])
    if host == run['nodes'][0]['host'] and not existing:
        setup_run(cfg, run)
    # Shared-output handshake: no assumptions from matching path strings.
    if read(root / 'run.json')['fingerprint'] != run['fingerprint']:
        raise ValueError('Output is not shared with the leader')
    full = run['context'].get('qualification_required', False)
    check = node_probe(cfg, full=full)
    expected = next(n['report'] for n in run['nodes'] if n['host'] == host)
    expected_gpus = [(g['uuid'], g['name'], g['total_mib']) for g in expected['gpus']]
    actual_gpus = [(g['uuid'], g['name'], g['total_mib']) for g in check['gpus']]
    if (check['code'] != run['code'] or check['versions'] != expected['versions']
            or check['driver'] != expected['driver'] or actual_gpus != expected_gpus
            or check['python_version'] != expected['python_version']):
        raise ValueError('Node resources/source/environment changed before launch')
    asset_record = dict(status='PASS_USED_TRAINING_AND_INFERENCE_ASSET_BYTES' if full else 'PASS_ASSET_SIZE_AND_METADATA_ONLY',
                        records=check['assets'])
    asset_path = root / 'node_checks' / check['hostname'] / 'assets.json'
    if existing:
        if read(asset_path) != asset_record:
            raise ValueError('Asset metadata changed since qualification; use a fresh run')
    else:
        write(asset_path, asset_record)
        write(root / 'node_checks' / check['hostname'] / 'ready.json', dict(fingerprint=run['fingerprint'], report=check))
    return {'ready': True, 'hostname': check['hostname']}


def remote_action(cfg, host, action, value):
    return capture(cfg, host, [cfg['python'], '-B', cfg['repo'] + '/cluster/launch.py',
                               '--node-action', action, '--payload-stdin'], timeout=3600,
                   input_text=json.dumps(value))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=REPO / 'cluster/site.local.json')
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--init', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--qualify-only', action='store_true')
    ap.add_argument('--with-qualification', action='store_true', help='Run the optional native DMD qualification before training')
    ap.add_argument('--arm', choices=['baseline', 'ssf'], default='baseline')
    ap.add_argument('--run-id')
    ap.add_argument('--resume', help='Full checkpoint from this portable run and identical topology')
    ap.add_argument('--node-probe', help=argparse.SUPPRESS)
    ap.add_argument('--verify-assets', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--node-action', help=argparse.SUPPRESS)
    ap.add_argument('--payload', help=argparse.SUPPRESS)
    ap.add_argument('--payload-stdin', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.node_probe:
        print(json.dumps(node_probe(json.loads(args.node_probe), args.verify_assets)))
        return
    if args.node_action:
        payload = json.loads(sys.stdin.read() if args.payload_stdin else args.payload)
        cfg, run = payload['config'], payload['run']
        if args.node_action == 'setup':
            result = node_setup(cfg, run, payload['host'], payload.get('existing', False))
        elif args.node_action == 'read':
            path = Path(payload['path']).resolve()
            if not path.is_relative_to(Path(run['root']).resolve()):
                raise ValueError('Read outside run output')
            result = read(path)
        elif args.node_action == 'prepare_train':
            root = Path(run['root'])
            folder = root / 'training' / run['context']['arm']
            attempts = [int(p.name.split('_')[1]) for p in folder.glob('attempt_*') if p.is_dir()]
            if (folder / 'TRAINING_AND_GENERATION_COMPLETE.json').exists():
                raise ValueError('This arm already completed 600 outer iterations')
            if not payload.get('resume') and list((folder / 'checkpoints').glob('*.COMMITTED.json')):
                raise ValueError('Existing trajectory requires an explicit full --resume checkpoint')
            if attempts and not payload.get('resume'):
                raise ValueError('Existing training attempt: resume a committed checkpoint or use a new run-id')
            if payload.get('resume'):
                path = Path(payload['resume'])
                marker = read(path.with_suffix('.COMMITTED.json'))
                if (marker['status'] != 'COMMITTED' or marker['path'] != str(path)
                        or marker['arm'] != run['context']['arm'] or not marker['readback_all_components_exact']
                        or sha(path) != marker['SHA256']):
                    raise ValueError('Invalid or modified complete checkpoint')
            result = {'attempt': max(attempts, default=-1) + 1}
        elif args.node_action == 'handshake':
            token = payload['token']
            if not re.fullmatch(r'[a-f0-9]{32}', token):
                raise ValueError('Invalid handshake token')
            path = Path(run['root']) / 'handshakes' / token / (socket.gethostname() + '.json')
            write(path, dict(token=token, hostname=socket.gethostname()))
            result = {'path': str(path), 'token': token}
        elif args.node_action == 'error_logs':
            mode, arm = payload['mode'], payload['arm']
            if mode not in ['train', 'qualify'] or arm not in ['baseline', 'ssf']:
                raise ValueError('Invalid worker log selection')
            logs = []
            for path in sorted((Path(run['root']) / 'logs').glob(f'{mode}_{arm}_rank*.log')):
                from collections import deque
                with path.open(errors='replace') as stream:
                    tail = ''.join(deque(stream, maxlen=60))
                logs.append(dict(path=str(path), tail=tail))
            result = dict(logs=logs)
        else:
            raise ValueError('Unknown internal action')
        print(json.dumps(result))
        return
    if args.init:
        init_config(args.config)
        return
    if not args.config.is_file():
        raise ValueError('Site config absent. Run ./launch.sh --init, then review cluster/site.local.json')
    cfg = config(args.config)
    qualification_required = args.with_qualification or args.qualify_only
    run_id = args.run_id or time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_id):
        raise ValueError('Invalid run-id')
    check_code()
    code = code_manifest()
    if args.dry_run:
        print(json.dumps(dict(status='PLAN_ONLY', inventory=hosts(cfg), nodes=cfg['nodes'],
                              world_size=cfg['nodes'] * cfg['gpus_per_node'], accumulation_per_rank=64 // (cfg['nodes'] * cfg['gpus_per_node']),
                              output=cfg['output'], arm=args.arm, native_qualification_first=qualification_required,
                              training_started=False, node_contacted=False), indent=2))
        return
    receipt_path = REPO / 'cluster/run_receipts' / (run_id + '.json')
    prior = read(receipt_path) if args.run_id and receipt_path.is_file() else None
    if args.resume and not prior:
        raise ValueError('--resume requires --run-id and its local run receipt')
    if prior:
        cfg['hosts'] = [n['host'] for n in prior['run']['nodes']]
    selected = select_nodes(cfg)
    if args.check:
        print(json.dumps(dict(status='PASS_NODE_PREFLIGHT_ONLY', selected=[h for h, _ in selected],
                              training_started=False, runtime_qualified=False), indent=2))
        return
    review = None
    if args.arm == 'ssf':
        if not cfg['ssf_review']:
            raise ValueError('SSF requires ssf_review with a real frozen baseline review and explicit SSF approval; see cluster/README.md')
        review = read(cfg['ssf_review'])
        if (review.get('status') != 'PASS' or review.get('ssf_authorized') is not True
                or review.get('origin') != 'actual user review' or not review.get('baseline_run')
                or not review.get('scores_file') or not review.get('scores_sha256')):
            raise ValueError('Incomplete SSF review record')
        if sha(review['scores_file']) != review['scores_sha256']:
            raise ValueError('Frozen user score SHA256 mismatch')
    if cfg['transport'] == 'ssh':
        cfg['repo'] = str(Path(cfg['repo']) / '.ssf_deploy' / digest(code)[:20])
    deploy(cfg, selected, code)
    nodes = [dict(host=h, report=r) for h, r in selected]
    context = dict(code=code, config=cfg, nodes=nodes, arm=args.arm, review=review,
                   qualification_required=qualification_required)
    run = dict(root=str(Path(cfg['output']) / run_id), nodes=nodes, code=code,
               fingerprint=digest(context), context=context, run_id=run_id, created_at=time.time())
    if prior:
        old_cfg = dict(prior['config'])
        # hosts may originally have come from gpuall's larger pool.
        old_cfg['hosts'] = cfg['hosts']
        if prior['run']['code'] != code or old_cfg != cfg or prior['run']['context']['arm'] != args.arm:
            raise ValueError('Existing run source/config/arm differs; use a new run-id')
        run = prior['run']
        if run['context'].get('qualification_required', True) != qualification_required:
            raise ValueError('Use the same qualification option as the original run')
    if args.resume:
        path = Path(args.resume)
        expected = Path(run['root']) / 'training' / args.arm / 'checkpoints'
        if not path.is_absolute() or path.parent != expected or path.suffix != '.pt':
            raise ValueError('--resume must be a full checkpoint in this run/arm checkpoints directory')
    payload = dict(config=cfg, run=run, host=nodes[0]['host'], existing=bool(prior))
    print('Preparing ' + run['root'] + ' on ' + ', '.join(n['host'] for n in nodes), flush=True)
    remote_action(cfg, nodes[0]['host'], 'setup', payload)
    if not prior:
        write(receipt_path, dict(config=cfg, run=run))
    for node in nodes[1:]:
        remote_action(cfg, node['host'], 'setup', dict(config=cfg, run=run, host=node['host'], existing=bool(prior)))
    token = uuid.uuid4().hex
    for node in nodes:
        handshake = remote_action(cfg, node['host'], 'handshake', dict(config=cfg, run=run, token=token))
        checked = remote_action(cfg, nodes[0]['host'], 'read', dict(config=cfg, run=run, path=handshake['path']))
        if checked.get('token') != token:
            raise ValueError('Output directory failed the cross-node write/read handshake')
    if qualification_required:
        if not prior:
            run_group(cfg, run, 'qualify', args.arm)
        result = remote_action(cfg, nodes[0]['host'], 'read', dict(config=cfg, run=run, path=run['root'] + '/qualification/real_dmd_accumulation/complete.json'))
        if result.get('run_fingerprint') != run['fingerprint'] or result.get('status') != 'PASS_CLUSTER_NATIVE_DMD_B64_EMA_RESUME':
            raise ValueError('Current topology qualification did not pass')
        if result['projected_training_hours'] > cfg['wall_hours']:
            raise ValueError('Measured training projection exceeds wall_hours; qualification preserved, training not started')
    if not args.qualify_only:
        attempt = remote_action(cfg, nodes[0]['host'], 'prepare_train',
                                dict(config=cfg, run=run, resume=args.resume))
        run['attempt'] = attempt['attempt']
        print('Starting ' + args.arm + '; worker logs: ' + run['root'] + '/logs/train_' + args.arm + '_rank*.log', flush=True)
        run_group(cfg, run, 'train', args.arm, args.resume)
        complete = remote_action(cfg, nodes[0]['host'], 'read', dict(config=cfg, run=run,
                  path=run['root'] + '/training/' + args.arm + '/TRAINING_AND_GENERATION_COMPLETE.json'))
        if complete.get('status') != 'PASS_600_OUTER_120G_600C_EMA80':
            raise ValueError('Training stopped before 600 outer iterations; inspect checkpoints/logs and use --resume')
    print('Completed ' + ('qualification' if args.qualify_only else args.arm) + ': ' + run['root'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, 'stderr', None) or str(exc)
        print('SSF: ' + detail.strip(), file=sys.stderr)
        sys.exit(1)
