#!/bin/bash
#SBATCH --job-name=mh_s3smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step3smoke_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step3smoke_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step3_diffrax.py
J=$HUNT/data/step3smoke_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
"$PY" "$S" --part 3a --N 500 --n-steps 10 --save t1    --json "$J" || echo "FAILED 3a-t1"
"$PY" "$S" --part 3a --N 500 --n-steps 10 --save steps --json "$J" || echo "FAILED 3a-steps"
"$PY" "$S" --part 3b_manual --N 500 --solver Tsit5 --n-steps 5 --json "$J" || echo "FAILED 3b_manual"
"$PY" "$S" --part 3b_solve  --N 500 --solver Tsit5 --n-steps 5 --json "$J" || echo "FAILED 3b_solve"
echo "### SMOKE DONE ###"
