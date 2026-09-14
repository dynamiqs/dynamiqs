#!/bin/bash
#SBATCH --job-name=mh_step7b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7b_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7b_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
for v in RAISE off nan; do
  echo; echo "############### EQX_ON_ERROR=$v ###############"
  if [ "$v" = "RAISE" ]; then
    "$PY" "$HUNT/scripts/step7b_errorif.py"
  else
    EQX_ON_ERROR=$v "$PY" "$HUNT/scripts/step7b_errorif.py"
  fi
done
echo; echo "### STEP 7b COMPLETE ###"
