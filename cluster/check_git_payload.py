"""Inspect publish candidates without printing matched credential contents."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.mp4', '.mov', '.avi', '.npy', '.npz'}
PATTERNS = [
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
    re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}'),
    re.compile(rb'AKIA[0-9A-Z]{16}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]
ASSIGNMENT = re.compile(
    rb"(?:api[_-]?key|access[_-]?token|password|secret)\s*=\s*['\"]([^'\"\r\n]{12,})['\"]",
    re.IGNORECASE)


def main():
    raw = subprocess.check_output([
        'git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=REPO)
    names = sorted(set(x.decode() for x in raw.split(b'\0') if x))
    errors = []
    total = 0
    for name in names:
        path = REPO / name
        if path.is_symlink() or not path.resolve().is_relative_to(REPO):
            errors.append(dict(path=name, reason='symlink/outside checkout'))
            continue
        if path.suffix.lower() in FORBIDDEN or path.name.startswith('.env'):
            errors.append(dict(path=name, reason='model/data/video/private environment payload'))
            continue
        size = path.stat().st_size
        if size > 5 * 1024**2:
            errors.append(dict(path=name, reason='larger than5MiB; review separately', bytes=size))
            continue
        data = path.read_bytes()
        total += size
        for index, pattern in enumerate(PATTERNS):
            if pattern.search(data):
                errors.append(dict(path=name, reason='credential-like content', pattern=index))
        for match in ASSIGNMENT.finditer(data):
            value = match.group(1).lower()
            if not any(word in value for word in [b'example', b'placeholder', b'replace', b'your_', b'dummy']):
                errors.append(dict(path=name, reason='literal credential assignment; value withheld',
                                   line=data[:match.start()].count(b'\n')+1))
    print(json.dumps(dict(status='PASS_CODE_ONLY_PAYLOAD' if not errors else 'FAIL_REVIEW_REQUIRED',
                          files=len(names), bytes=total, issues=errors), indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(main())
