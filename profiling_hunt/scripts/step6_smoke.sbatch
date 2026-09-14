#!/bin/bash
#SBATCH --job-name=mh_s6smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6smoke_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6smoke_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step6_grid.py
J=$HUNT/data/step6smoke_${SLURM_JOB_ID}.jsonl

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true

echo "# job $SLURM_JOB_ID on $(hostname) $(date -Is) GPU=$CUDA_VISIBLE_DEVICES"
echo "# inherited XLA env (contamination check):"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

# worst-case timing corners: biggest N x most saves x lowest order (most steps)
for cfg in \
  "2048 1000 0 rk2 none" \
  "2048 1000 0 rk8 none" \
  "2048 10   0 rk5 none" \
  "2048 120  1 rk5 none" \
  "2048 1000 1 rk5 none" \
  "1024 1000 0 rk2 none" \
  "64   1000 1 rk2 direct" \
  "64   10   1 rk2 none" ; do
  set -- $cfg
  echo "=== N=$1 ntsave=$2 save=$3 method=$4 grad=$5 ==="
  /usr/bin/time -f "WALL %e s  MAXRSS %M KB" \
    "$PY" "$S" --N $1 --ntsave $2 --save-states $3 --method $4 --gradient $5 --json "$J" \
    || echo "(exit $?)"
  echo
done
echo "### SMOKE COMPLETE  json=$J ###"
