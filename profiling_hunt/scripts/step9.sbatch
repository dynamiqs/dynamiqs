#!/bin/bash
#SBATCH --job-name=mh_step9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step9_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step9_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step9_marker.py
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"
echo; echo "=== A) memory, N=2000 ntsave=101 ==="
for v in A_baseline B_marker C_marker_jit D_baseline_jit; do
  "$PY" "$S" --variant $v --N 2000 --ntsave 101 || true
done
echo; echo "=== B) does each still report a genuine failure? max_steps=1 ==="
for v in A_baseline B_marker C_marker_jit D_baseline_jit; do
  "$PY" "$S" --variant $v --N 64 --ntsave 11 --max-steps 1 || true
done
echo; echo "=== C) how far can the best variant go? ==="
"$PY" "$S" --variant C_marker_jit --N 2000 --ntsave 303 || true
"$PY" "$S" --variant C_marker_jit --N 2000 --ntsave 606 || true
echo; echo "### STEP 9 COMPLETE ###"
