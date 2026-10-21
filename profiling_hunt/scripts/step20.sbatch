#!/bin/bash
#SBATCH --job-name=mh_step20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20_%A_%a.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20_%A_%a.err

# Step 20 - dynamiqs save-path memory laws.
#   Study A: save_states=True, 5 N x 5 ntsave.
#   Study B: save_states=False + exp_ops, 5 N x 5 ntsave x n_obs{0,1,2,4,8,16} x {dense,dia}.
#   + N=16 floor controls. 303 cells, ONE PROCESS PER CELL.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step20_grid.py
G=$HUNT/scripts/step20_configs.py
NSHARDS=6
SID=$SLURM_ARRAY_TASK_ID
J=$HUNT/data/step20_${SLURM_ARRAY_JOB_ID}_${SID}.jsonl

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true     # MUST be explicit - login env sets false

echo "# STEP 20 shard $SID/$NSHARDS  job ${SLURM_ARRAY_JOB_ID}_${SID}  $(hostname)  $(date -Is)  GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

NCFG=$("$PY" "$G" "$SID" "$NSHARDS" | wc -l)
i=0
"$PY" "$G" "$SID" "$NSHARDS" | while read -r ST N NT SV NE LY; do
  i=$((i+1))
  echo "-------- [$SID $i/$NCFG] study=$ST N=$N ntsave=$NT save=$SV n_obs=$NE layout=$LY --------"
  "$PY" "$S" --study "$ST" --N "$N" --ntsave "$NT" --save-states "$SV" \
       --n-obs "$NE" --obs-layout "$LY" --json "$J" || echo "(worker exit $?)"
done
echo "### STEP 20 SHARD $SID COMPLETE  json=$J  $(date -Is) ###"
