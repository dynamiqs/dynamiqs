#!/bin/bash
#SBATCH --job-name=mh_step5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step5_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step5_%j.err

# Step 5 - does dynamiqs inherit the 3x trajectory peak? One process per config.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step5_dq_save.py
J=$HUNT/data/step5_${SLURM_JOB_ID}.jsonl
N=2000

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)  $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
echo "# N=$N  R = 16*N^2 = $((16*N*N)) B"
echo "############################################################################"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

echo "============================================================================"
echo "A) ntsave sweep, save_states=True  -> fit peak = a + b * ntsave * R"
echo "   H-B: b = 3.000 means dynamiqs inherits the raw-diffrax 3x; b = 1.000 means not"
echo "============================================================================"
for nt in 2 6 11 26 51 101; do
  "$PY" "$S" --N $N --ntsave $nt --save-states 1 --json "$J" || echo "(nonzero: $?)"
done

echo
echo "============================================================================"
echo "B) same ntsave sweep, save_states=False  -> must be FLAT"
echo "============================================================================"
for nt in 2 11 51 101; do
  "$PY" "$S" --N $N --ntsave $nt --save-states 0 --json "$J" || echo "(nonzero: $?)"
done

echo
echo "============================================================================"
echo "C) exp_ops instead of states: save_states=False + 2 exp_ops -> must be FLAT"
echo "============================================================================"
for nt in 11 101; do
  "$PY" "$S" --N $N --ntsave $nt --save-states 0 --n-exp-ops 2 --json "$J" \
    || echo "(nonzero: $?)"
done

echo
echo "============================================================================"
echo "D) max_steps sweep at fixed ntsave=11  -> H-C: memory must NOT depend on it"
echo "   (dynamiqs hard-codes max_steps=100000 for fixed-step methods)"
echo "============================================================================"
for ms in 1000 10000 100000 1000000; do
  "$PY" "$S" --N $N --ntsave 11 --save-states 1 --max-steps $ms --json "$J" \
    || echo "(nonzero: $?)"
done

echo
echo "============================================================================"
echo "E) gradient/adjoint sweep at fixed ntsave=11, forward-only solve"
echo "   gradient=None already selects RecursiveCheckpointAdjoint (checkpointed loop)"
echo "============================================================================"
for g in none forward direct checkpointed; do
  "$PY" "$S" --N $N --ntsave 11 --save-states 1 --gradient $g --json "$J" \
    || echo "(nonzero: $?)"
done

echo
echo "============================================================================"
echo "F) dense-operator cross-check of the ntsave slope (layout must not matter)"
echo "============================================================================"
for nt in 11 51; do
  "$PY" "$S" --N $N --ntsave $nt --save-states 1 --layout dense --json "$J" \
    || echo "(nonzero: $?)"
done

echo
echo "### STEP 5 COMPLETE - json at $J ###"
