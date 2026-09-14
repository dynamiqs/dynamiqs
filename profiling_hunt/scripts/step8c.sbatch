#!/bin/bash
#SBATCH --job-name=mh_step8c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step8c_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step8c_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step8c_verify.py
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
echo; echo "=== A) N=2000 ntsave=101: baseline vs patched ==="
"$PY" "$S" --N 2000 --ntsave 101 --patch 0
"$PY" "$S" --N 2000 --ntsave 101 --patch 1
echo; echo "=== B) N=2000 ntsave=303: baseline OOMs today ==="
"$PY" "$S" --N 2000 --ntsave 303 --patch 0 || true
"$PY" "$S" --N 2000 --ntsave 303 --patch 1 || true
echo; echo "=== C) N=2000 ntsave=606: reachable only with the fix ==="
"$PY" "$S" --N 2000 --ntsave 606 --patch 1 || true
echo; echo "=== D) safety at production size: max_steps=1 must raise ==="
"$PY" "$S" --N 2000 --ntsave 101 --max-steps 1 --patch 1 || true
echo; echo "### STEP 8c COMPLETE ###"
