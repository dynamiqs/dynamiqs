#!/bin/bash
#SBATCH --job-name=mh_step4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4_%j.err

# Step 4 - Attributing the overhead. One process per row (peak_bytes_in_use never
# resets). Same N, same solver (Tsit5), trajectory storage off, for every row.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step4_lindblad.py
J=$HUNT/data/step4_${SLURM_JOB_ID}.jsonl
N=2000

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)  $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
echo "# N=$N   R = 16*N^2 = $((16*N*N)) B   (complex128 = 16 bytes/element)"
echo "############################################################################"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

for r in bare lindblad dq_dense dq_dia; do
  echo
  echo "######################## ROW: $r ########################"
  "$PY" "$S" --row $r --N $N --json "$J" || echo "(exited nonzero: $?)"
done

echo
echo "### STEP 4 COMPLETE - json at $J ###"
