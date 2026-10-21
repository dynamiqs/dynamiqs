#!/bin/bash
#SBATCH --job-name=mh_step19
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step19_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt/logs/step19_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step19_floatfix.py
J=$HUNT/data/step19_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"

echo; echo "=== A) bare diffrax, float(ts[0])/float(ts[-1]) applied pre-jit ==="
for nt in 20 50 100 200 400; do
  "$PY" "$S" --variant bare_float_fix --ntsave $nt --json "$J" || echo "(exit $?)"
done

echo; echo "=== B) the naive fix dropped directly into DiffraxIntegrator.run() ==="
"$PY" "$S" --variant dynamiqs_naive_property_fix --ntsave 100 --json "$J" || echo "(exit $?)"

echo; echo "### STEP 19 COMPLETE  json=$J ###"
