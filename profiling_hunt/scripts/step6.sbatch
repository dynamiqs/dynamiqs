#!/bin/bash
#SBATCH --job-name=mh_step6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6_%A_%a.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step6_%A_%a.err

# Step 6 - grid-search validation of the two Step-5 memory formulas.
# 960 cells: save_states{0,1} x N{64..2048} x method{rk2,rk3,rk5,rk8}
#            x ntsave{linspace(10,1000,10)} x gradient{none,direct}
# ONE PROCESS PER CELL (peak_bytes_in_use is a high-water mark that never resets).
# Sharded over 8 array tasks, 4 concurrent (4 of the 8 L40S are held by other jobs).
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step6_grid.py
G=$HUNT/scripts/step6_configs.py
NSHARDS=8
SID=$SLURM_ARRAY_TASK_ID
J=$HUNT/data/step6_${SLURM_ARRAY_JOB_ID}_${SID}.jsonl

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true     # MUST be explicit - login env sets false

echo "############################################################################"
echo "# STEP 6 shard $SID/$NSHARDS  job ${SLURM_ARRAY_JOB_ID}_${SID}  $(hostname)"
echo "# $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
echo "# inherited XLA*/JAX* environment (contamination check, Finding 1.A):"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
echo "############################################################################"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo

NCFG=$("$PY" "$G" "$SID" "$NSHARDS" | wc -l)
echo "shard $SID has $NCFG configs"
echo

i=0
"$PY" "$G" "$SID" "$NSHARDS" | while read -r N NT SV M GR; do
  i=$((i+1))
  echo "-------- [$SID $i/$NCFG] N=$N ntsave=$NT save=$SV method=$M gradient=$GR --------"
  "$PY" "$S" --N "$N" --ntsave "$NT" --save-states "$SV" --method "$M" \
       --gradient "$GR" --json "$J" || echo "(worker exit $?)"
done

echo
echo "### STEP 6 SHARD $SID COMPLETE  json=$J  $(date -Is) ###"
