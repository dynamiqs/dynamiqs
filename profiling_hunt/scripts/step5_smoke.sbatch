#!/bin/bash
#SBATCH --job-name=mh_s5smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step5smoke_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step5smoke_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step5_dq_save.py
J=$HUNT/data/step5smoke_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
"$PY" "$S" --N 400 --ntsave 11 --save-states 1 --json "$J" || echo "FAILED A"
"$PY" "$S" --N 400 --ntsave 11 --save-states 0 --json "$J" || echo "FAILED B"
"$PY" "$S" --N 400 --ntsave 11 --save-states 0 --n-exp-ops 2 --json "$J" || echo "FAILED C"
"$PY" "$S" --N 400 --ntsave 11 --gradient forward --json "$J" || echo "FAILED D"
"$PY" "$S" --N 400 --ntsave 11 --gradient direct --json "$J" || echo "FAILED E"
echo "### SMOKE DONE ###"
