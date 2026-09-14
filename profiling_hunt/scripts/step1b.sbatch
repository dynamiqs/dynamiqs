#!/bin/bash
#SBATCH --job-name=mh_step1b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step1b_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step1b_%j.err

# Step 1b - Grounding, corrected.
#
# The first pass (job 69370) was contaminated: XLA_PYTHON_CLIENT_PREALLOCATE=false
# is set in the AMBIENT login environment and SLURM --export=ALL propagated it into
# the job, so the intended operating configuration (.95 WITH preallocation) was
# never actually measured. Here both variables are controlled explicitly for every
# configuration - nothing is inherited.
set -u

HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
SCRIPT=$HUNT/scripts/step1_limits.py
JSON=$HUNT/data/step1b_${SLURM_JOB_ID}.jsonl

echo "############################################################################"
echo "# SLURM job $SLURM_JOB_ID on $(hostname)   $(date -Is)"
echo "# CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "# SLURM_JOB_GPUS       = ${SLURM_JOB_GPUS:-<unset>}"
echo "#"
echo "# INHERITED environment as seen by the job (the contamination check):"
env | grep -iE '^(XLA|JAX)' | sort | sed 's/^/#   /'
echo "#   (nothing printed above == nothing inherited)"
echo "############################################################################"
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo

run_cfg () {  # $1=label  $2=mem_fraction|"-"  $3=prealloc|"-"  $4=probe_ceiling 0/1
  local label=$1 frac=$2 prealloc=$3 probe=$4
  local -a pre=(env)
  if [ "$frac" = "-" ]; then pre+=(-u XLA_PYTHON_CLIENT_MEM_FRACTION)
  else pre+=("XLA_PYTHON_CLIENT_MEM_FRACTION=$frac"); fi
  if [ "$prealloc" = "-" ]; then pre+=(-u XLA_PYTHON_CLIENT_PREALLOCATE)
  else pre+=("XLA_PYTHON_CLIENT_PREALLOCATE=$prealloc"); fi
  pre+=("PROBE_CEILING=$probe")
  echo
  echo "############### $label  (frac=$frac prealloc=$prealloc probe=$probe) ######"
  "${pre[@]}" "$PY" "$SCRIPT" "$label" "$JSON"
}

# A: true JAX defaults - both vars genuinely unset.
run_cfg "A_true_defaults"            "-"   "-"      0

# B: THE OPERATING CONFIGURATION for the whole hunt: .95 + preallocation ON.
#    This is the one the brief specifies and the one job 69370 failed to measure.
run_cfg "B_095_prealloc_TRUE"        ".95" "true"   1

# C: preallocation off, default fraction (what the ambient env silently gave us).
run_cfg "C_defaultfrac_prealloc_off" "-"   "false"  0

# D: .95 + preallocation off - recorded for comparison, NOT adopted.
run_cfg "D_095_prealloc_off"         ".95" "false"  1

echo
echo "### STEP 1b COMPLETE - json at $JSON ###"
