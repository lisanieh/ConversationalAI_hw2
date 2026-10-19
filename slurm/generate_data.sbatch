#!/bin/bash
# Usage: sbatch slurm/generate_data.sbatch

#SBATCH --job-name=a2-data
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate ~/scratch/conda/envs/CAI_A2

export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

python -u - <<'PY'
import contextlib
import io
import json

from generate_preference_data import synthesize_pairs

with contextlib.redirect_stdout(io.StringIO()):
    pairs = synthesize_pairs(2)

for pair in pairs:
    print(json.dumps(pair, ensure_ascii=False))
PY