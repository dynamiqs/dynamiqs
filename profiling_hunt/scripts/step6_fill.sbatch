#!/bin/bash
#SBATCH --job-name=mh_s6fill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6fill_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6fill_%j.err
# Re-run the 2 grid cells whose worker was SIGTERMed (exit 143) before it could
# append its record. Written to its own shard file so the main run stays untouched.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step6_grid.py
J=$HUNT/data/step6_69401_fill.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# fill job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
"$PY" "$S" --N 1024 --ntsave 670 --save-states 1 --method rk2 --gradient direct --json "$J"
"$PY" "$S" --N 64   --ntsave 1000 --save-states 0 --method rk5 --gradient direct --json "$J"
echo "### FILL COMPLETE ###"
