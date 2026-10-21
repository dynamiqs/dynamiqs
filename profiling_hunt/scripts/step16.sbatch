#!/bin/bash
#SBATCH --job-name=mh_step16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step16_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step16_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step16_exactpatch_sweep.py
J=$HUNT/data/step16_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
for v in baseline patched; do
  echo; echo "=== $v ==="
  for nt in 20 50 100 200 400; do
    "$PY" "$S" --variant $v --ntsave $nt --json "$J" || echo "(exit $?)"
  done
done
echo; echo "### STEP 16 COMPLETE  json=$J ###"
