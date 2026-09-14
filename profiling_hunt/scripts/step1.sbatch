#!/bin/bash
#SBATCH --job-name=mh_step1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step1_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step1_%j.err

# Step 1 - Grounding. One whole L40S, no MPS, no sharing.
set -u

HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
SCRIPT=$HUNT/scripts/step1_limits.py
JSON=$HUNT/data/step1_${SLURM_JOB_ID}.jsonl

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)"
echo "# date         : $(date -Is)"
echo "# CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "# SLURM_JOB_GPUS       = ${SLURM_JOB_GPUS:-<unset>}"
echo "############################################################################"
echo
echo "### full-node nvidia-smi view of the reservation ###"
nvidia-smi
echo

# ---------------------------------------------------------------------------
# Config A: JAX defaults. Expect bytes_limit ~= 75% of physical.
# ---------------------------------------------------------------------------
env -u XLA_PYTHON_CLIENT_MEM_FRACTION -u XLA_PYTHON_CLIENT_PREALLOCATE \
    "$PY" "$SCRIPT" "A_jax_defaults" "$JSON"
echo

# ---------------------------------------------------------------------------
# Config B: the operating configuration for the whole hunt. Expect 45.89e9.
# ---------------------------------------------------------------------------
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 \
    "$PY" "$SCRIPT" "B_mem_fraction_095" "$JSON"
echo

# ---------------------------------------------------------------------------
# Config C: on-demand allocation. Recorded for comparison, NOT adopted -
# it changes OOM timing and would confound Step 2.
# ---------------------------------------------------------------------------
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" "$SCRIPT" "C_preallocate_false" "$JSON"
echo

# ---------------------------------------------------------------------------
# Config D: .95 + no preallocation, to separate the two knobs.
# ---------------------------------------------------------------------------
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PY" "$SCRIPT" "D_095_and_no_prealloc" "$JSON"

echo
echo "### STEP 1 COMPLETE - json at $JSON ###"
