#!/bin/bash
#SBATCH --job-name=mh_step8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step8_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step8_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
for v in v0_baseline v1_marker v2_barrier v3_one_callback; do
  echo; echo "================================================================"
  "$PY" "$HUNT/scripts/step8_tokenfix.py" --variant $v 2>&1 | tail -25 || echo "(exit $?)"
done
echo; echo "### STEP 8 COMPLETE ###"
