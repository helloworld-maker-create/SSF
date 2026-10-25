#!/usr/bin/env bash
#SBATCH --job-name=ssf-code-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
# Submit partition/account/GPU/memory requests explicitly for your cluster.
set -euo pipefail
: "${SSF_CHECKOUT:?Set the cloned repository directory}"
: "${SSF_PYTHON:?Set the already-qualified Python executable}"
cd "$SSF_CHECKOUT"
export PYTHONNOUSERSITE=1
"$SSF_PYTHON" -B cluster/preflight.py --resources
# This template never starts a model, qualification backward, or training.
