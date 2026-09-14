#!/bin/bash
#SBATCH --job-name=mh_step10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step10_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step10_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step10_exact.py
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"

echo; echo "=== A) N=2000, ntsave=101: memory and run time ==="
for v in baseline result_code plus_nan; do "$PY" "$S" --variant $v --N 2000 --ntsave 101 || true; done

echo; echo "=== B) N=2000, ntsave=303: baseline OOMs today ==="
for v in baseline result_code plus_nan; do "$PY" "$S" --variant $v --N 2000 --ntsave 303 || true; done

echo; echo "=== C) run time at a save-heavy config, N=1000 ntsave=1000 ==="
for v in baseline result_code plus_nan; do "$PY" "$S" --variant $v --N 1000 --ntsave 1000 || true; done

echo; echo "=== D) does the failure still surface? max_steps=1 ==="
for v in baseline result_code plus_nan; do "$PY" "$S" --variant $v --N 64 --ntsave 11 --max-steps 1 || true; done

echo; echo "### STEP 10 COMPLETE ###"
