#!/bin/bash
#SBATCH --job-name=mh_step18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step18_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step18_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step18_tracedt0.py
J=$HUNT/data/step18_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
for tr in 0 1; do
  echo; echo "=== traced=$tr ==="
  for nt in 20 50 100 200 400; do
    "$PY" "$S" --traced $tr --ntsave $nt --json "$J" || echo "(exit $?)"
  done
done
echo; echo "### STEP 18 COMPLETE  json=$J ###"
