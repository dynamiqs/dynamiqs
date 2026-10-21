#!/bin/bash
#SBATCH --job-name=mh_s20smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20smoke_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20smoke_%j.err
# Worst-case corners of the Step 20 grid: timing and headroom before the full run.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step20_grid.py
J=$HUNT/data/step20smoke_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true     # MUST be explicit - login env sets false
echo "# job $SLURM_JOB_ID on $(hostname) $(date -Is) GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
for cfg in \
  "B 16   20  0 0  none" \
  "A 1536 180 1 0  none" \
  "B 3072 180 0 0  none" \
  "B 3072 180 0 16 dense" \
  "B 3072 180 0 16 dia" ; do
  set -- $cfg
  /usr/bin/time -f "WALL %e s" "$PY" "$S" --study $1 --N $2 --ntsave $3 \
    --save-states $4 --n-obs $5 --obs-layout $6 --json "$J" || echo "(exit $?)"
done
echo "### SMOKE COMPLETE json=$J ###"
