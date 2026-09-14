#!/bin/bash
#SBATCH --job-name=mh_step4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4b_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step4b_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step4_lindblad.py
J=$HUNT/data/step4b_${SLURM_JOB_ID}.jsonl
N=2000
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID on $(hostname) $(date -Is) GPU=$CUDA_VISIBLE_DEVICES  N=$N"
env | grep -iE '^XLA' | sort | sed 's/^/#   /'
# The layout-only control pair: same cat qubit physics, DIA vs forced-dense operators.
for r in dq_dia dq_cat_dense; do
  echo; echo "######################## ROW: $r ########################"
  "$PY" "$S" --row $r --N $N --json "$J" || echo "(exited nonzero: $?)"
done
echo "### STEP 4b COMPLETE ###"
