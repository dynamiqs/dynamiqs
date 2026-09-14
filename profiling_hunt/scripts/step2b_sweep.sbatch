#!/bin/bash
#SBATCH --job-name=mh_step2b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2b_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2b_%j.err

# Step 2b - the full sweep, n = 2..40, k upward until OOM.
#
# ONE PROCESS PER n. peak_bytes_in_use is a high-water mark that never resets, so
# running all n in a single process would let the peak from n=2 mask every later
# configuration. A fresh process per n keeps peak_bytes_in_use meaningful.
set -u

HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
SCRIPT=$HUNT/scripts/step2_kron.py
CSV=$HUNT/data/step2b_sweep_${SLURM_JOB_ID}.csv

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # MUST be explicit: login env sets false

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)   $(date -Is)"
echo "# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?}"
echo "# XLA env actually in force:"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
echo "############################################################################"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo

for n in $(seq 2 40); do
  echo
  echo "######################## SWEEP n=$n ########################"
  "$PY" "$SCRIPT" --n "$n" --kmax 17 --csv "$CSV" || echo "(process exited nonzero: $?)"
done

echo
echo "### STEP 2b COMPLETE - csv at $CSV ###"
