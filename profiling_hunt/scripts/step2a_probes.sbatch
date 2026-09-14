#!/bin/bash
#SBATCH --job-name=mh_step2a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2a_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step2a_%j.err

# Step 2a - the two special probes from the brief, run early and repeated for
# determinism. Each repeat is a SEPARATE PROCESS so peak_bytes_in_use and the
# allocator state start clean every time.
#
#   n=36,k=11 -> 40.500 GiB resident. Brief calls this the "known good" anchor.
#                H1 (resident only) says it fits with 3.97 GiB headroom.
#                H2 (kron holds its input) says it needs 50.625 GiB -> OOM.
#   n=37,k=11 -> 42.781 GiB resident, +1.436 % over the MEASURED ceiling.
#                Predicted to OOM under every hypothesis.
set -u

HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
SCRIPT=$HUNT/scripts/step2_kron.py
CSV=$HUNT/data/step2a_probes_${SLURM_JOB_ID}.csv

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

for rep in 1 2 3 4 5; do
  echo
  echo "############################ n=36 k<=11  repeat $rep ########################"
  "$PY" "$SCRIPT" --n 36 --kmax 11 --csv "$CSV" || echo "(process exited nonzero: $?)"
done

for rep in 1 2 3 4 5; do
  echo
  echo "############################ n=37 k<=11  repeat $rep ########################"
  "$PY" "$SCRIPT" --n 37 --kmax 11 --csv "$CSV" || echo "(process exited nonzero: $?)"
done

echo
echo "### STEP 2a COMPLETE - csv at $CSV ###"
