#!/bin/bash
#SBATCH --job-name=mh_step3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step3_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step3_%j.err

# Step 3 - Isolating diffrax.
# ONE PROCESS PER CONFIGURATION: peak_bytes_in_use never resets, so a shared
# process would let the first configuration's peak mask every later one.
set -u

HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step3_diffrax.py
J=$HUNT/data/step3_${SLURM_JOB_ID}.jsonl
N=4000

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)   $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
echo "# N=$N  state_bytes = 8*N^2 = $((8*N*N)) B = $((8*N*N/1048576)) MiB"
echo "############################################################################"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

echo
echo "============================================================================"
echo "3a - MEMORY vs NUMBER OF TIME STEPS   (Euler, fixed dt)"
echo "  PREDICTION: SaveAt(t1=True)    -> FLAT in step count"
echo "  PREDICTION: SaveAt(steps=True) -> LINEAR, n_stored x state_bytes"
echo "============================================================================"
for ns in 10 100; do
  "$PY" "$S" --part 3a --N $N --n-steps $ns --save t1 --json "$J" \
    || echo "(exited nonzero: $?)"
done
for ns in 10 100; do
  "$PY" "$S" --part 3a --N $N --n-steps $ns --save steps --json "$J" \
    || echo "(exited nonzero: $?)"
done

echo
echo "----------------------------------------------------------------------------"
echo "3a-extra: does trajectory storage follow n_steps, or max_steps?"
echo "  same n_steps=10, max_steps x4"
echo "----------------------------------------------------------------------------"
"$PY" "$S" --part 3a --N $N --n-steps 10 --save steps --max-steps-mult 4 --json "$J" \
  || echo "(exited nonzero: $?)"

echo
echo "----------------------------------------------------------------------------"
echo "3a-extra: does closing over M in the vector field retain it after the solve?"
echo "----------------------------------------------------------------------------"
for mode in closure args; do
  "$PY" "$S" --part 3a --N $N --n-steps 10 --save t1 --vf-mode $mode --json "$J" \
    || echo "(exited nonzero: $?)"
done

echo
echo "============================================================================"
echo "3b - MEMORY vs NUMBER OF SOLVER STAGES   (10 steps held fixed)"
echo "  PREDICTION: memory ~ a + b*n_stages with b ~ state_bytes"
echo "============================================================================"
echo "--- 3b via solver.step() in a Python loop (un-jitted at the call level) ---"
for sv in Euler Ralston Bosh3 Tsit5 Dopri8; do
  "$PY" "$S" --part 3b_manual --N $N --solver $sv --n-steps 10 --json "$J" \
    || echo "(exited nonzero: $?)"
done
echo "--- 3b via the real diffeqsolve API, SaveAt(t1=True) ---"
for sv in Euler Ralston Bosh3 Tsit5 Dopri8; do
  "$PY" "$S" --part 3b_solve --N $N --solver $sv --n-steps 10 --json "$J" \
    || echo "(exited nonzero: $?)"
done

echo
echo "### STEP 3 COMPLETE - json at $J ###"
