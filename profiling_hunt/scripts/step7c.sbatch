#!/bin/bash
#SBATCH --job-name=mh_step7c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7c_%j.out
#SBATCH --error=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/logs/step7c_%j.err
set -u
HUNT=/srv/data/users/nlepage/new_dynamiqs/profiling_hunt
PY=/srv/data/users/nlepage/new_dynamiqs/dynamiqs/.venv/bin/python
S=$HUNT/scripts/step7c_verify.py
J=$HUNT/data/step7c_${SLURM_JOB_ID}.jsonl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export XLA_PYTHON_CLIENT_PREALLOCATE=true
echo "# job $SLURM_JOB_ID $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date -Is)"

echo; echo "=========== A) peak memory and answer, N=2000 ntsave=101 ==========="
for v in RAISE nan off; do
  echo "--- EQX_ON_ERROR=$v ---"
  if [ "$v" = "RAISE" ]; then "$PY" "$S" --N 2000 --ntsave 101 --json "$J"
  else EQX_ON_ERROR=$v "$PY" "$S" --N 2000 --ntsave 101 --json "$J"; fi
done

echo; echo "=========== B) how far can we now go? ntsave=303 (3x), only with off ==="
echo "--- default (expected to OOM: 3R*304 = 58 GiB) ---"
"$PY" "$S" --N 2000 --ntsave 303 --json "$J" || true
echo "--- EQX_ON_ERROR=off (expected ~1R*303 = 18 GiB) ---"
EQX_ON_ERROR=off "$PY" "$S" --N 2000 --ntsave 303 --json "$J" || true

echo; echo "=========== C) safety: a solve that genuinely fails (max_steps=1) ======"
for v in RAISE nan off; do
  echo "--- EQX_ON_ERROR=$v, max_steps=1 ---"
  if [ "$v" = "RAISE" ]; then "$PY" "$S" --N 64 --ntsave 11 --max-steps 1 --json "$J" || true
  else EQX_ON_ERROR=$v "$PY" "$S" --N 64 --ntsave 11 --max-steps 1 --json "$J" || true; fi
done
echo; echo "### STEP 7c COMPLETE  json=$J ###"
