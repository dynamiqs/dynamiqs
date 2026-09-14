#!/bin/bash
#SBATCH --job-name=mh_step7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
DUMP=$HUNT/data/step7_hlo_${SLURM_JOB_ID}
mkdir -p "$DUMP"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
"$PY" "$HUNT/scripts/step7_hlo.py" "$DUMP"
echo; echo "### dumped files (buffer assignment) ###"
ls "$DUMP" | grep -i "buffer" | head
echo "### total files: $(ls "$DUMP" | wc -l) ###"
