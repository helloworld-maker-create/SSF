#!/usr/bin/env bash
set -euo pipefail
SSF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SSF_PYTHON:-python3}" -B "$SSF_ROOT/cluster/launch.py" "$@"
