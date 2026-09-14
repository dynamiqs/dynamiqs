#!/bin/bash
#SBATCH --job-name=mh_step2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2d_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2d_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID on $(hostname) $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
"$PY" "$HUNT/scripts/step2d_pattern.py"
echo "### STEP 2d COMPLETE ###"
