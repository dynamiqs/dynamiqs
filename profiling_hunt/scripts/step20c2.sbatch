#!/bin/bash
#SBATCH --job-name=mh_step20c2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20c2_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step20c2_%j.err
# Step 20 study C - boundary scan of the save_states=False working-set step
# Follow-up: pin the step between N=2496 and N=2560, and check the hi regime persists at large N.
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step20_grid.py
J=$HUNT/data/step20c2_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true     # MUST be explicit - login env sets false
echo "# job $SLURM_JOB_ID on $(hostname) $(date -Is) GPU=$CUDA_VISIBLE_DEVICES"
env | grep -iE '^XLA|^JAX' | sort | sed 's/^/#   /'
for N in 2512 2528 2544 3584 4096 5120; do
  for obs in "0 none" "1 dia" "1 dense"; do
    set -- $obs
    "$PY" "$S" --study C --N $N --ntsave 20 --save-states 0 --n-obs $1 \
      --obs-layout $2 --json "$J" || echo "(worker exit $?)"
  done
done
echo "### STEP 20C COMPLETE json=$J ###"
