#!/bin/bash
#SBATCH --job-name=mh_s4smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4smoke_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4smoke_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step4_lindblad.py
J=$HUNT/data/step4smoke_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
for r in bare lindblad dq_dense dq_dia; do
  echo "########## $r ##########"
  "$PY" "$S" --row $r --N 400 --json "$J" || echo "FAILED $r"
done
echo "### SMOKE DONE ###"
