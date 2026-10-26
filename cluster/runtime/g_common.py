"""Portable run context. Historical host grants are never read or fabricated."""
from pathlib import Path
import csv, datetime, fcntl, hashlib, json, os, socket, subprocess, time
import sys

SOURCE = Path(__file__).resolve().parents[2]
PRODUCTION = SOURCE
ROOT = Path(os.environ['SSF_RUN_ROOT'])
SITE = json.loads(os.environ['SSF_SITE_JSON'])
sys.path.insert(0, str(SOURCE / 'cluster/original_run/scripts'))
# Keep this directory first so archives import the portable context.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()


def check_source():
    manifest = read(ROOT / 'source_manifest/source_files.json')
    for item in manifest['files']:
        assert sha(SOURCE / item['path']) == item['SHA256'], 'SOURCE_CHANGED ' + item['path']
    return manifest


def storage_guard(required=2 * 1024**3):
    fs = os.statvfs(ROOT)
    free = fs.f_bavail * fs.f_frsize
    assert free - required >= 32 * 1024**3, f'BLOCKED_STORAGE free={free}, next={required}'
    return free


def resources(before_model=False):
    identifier = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in identifier
    raw = subprocess.check_output(['nvidia-smi', '-i', identifier,
          '--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total',
          '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    values = [v.strip() for v in raw.split(',')]
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    foreign = [row for row in csv.reader(procs.splitlines())
               if row and row[0].strip() == values[1] and int(row[1]) != os.getpid()]
    assert not foreign, 'BLOCKED_RESOURCE_OTHER_PROCESS ' + str(foreign)
    if before_model:
        assert float(values[3]) <= 5 and int(values[4]) >= SITE['min_gpu_free_gib'] * 1024, 'BLOCKED_GPU_VRAM'
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024
    assert available >= SITE['host_reserve_gib'] * 1024**3, 'BLOCKED_HOST_RAM_RESERVE'
    return dict(at=time.time(), pid=os.getpid(), gpu=raw, foreign=foreign,
                filesystem_available_bytes=storage_guard(), host_MemAvailable_bytes=available)


def acquire_gpu():
    folder = Path.home() / '.cache/ssf_gpu_leases'
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / (os.environ['CUDA_VISIBLE_DEVICES'] + '.lock')).open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def matrix(stage='A'):
    with (SOURCE / 'cluster/original_run/contracts/matched_mvp_matrix.csv').open() as stream:
        return [row for row in csv.DictReader(stream) if row['stage'] == stage]


def budget_used(category):
    total = 0.0
    for path in (ROOT / 'state').glob('*_started.json'):
        state = read(path)
        if state.get('budget_category') != category:
            continue
        finished = path.with_name(path.name.replace('_started.json', '_finished.json'))
        end = read(finished)['ended_at_epoch'] if finished.exists() else time.time()
        total += (end - state['started_at_epoch']) * state.get('GPU_count', 1) / 3600
    return total


def budget_guard(category, cap):
    used = budget_used(category)
    assert used < cap - .05, f'BLOCKED_BUDGET {category} used={used} cap={cap}'
    return used
