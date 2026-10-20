#!/bin/bash
# Usage: sbatch slurm/test_sft.sbatch

#SBATCH --job-name=a2-sft-test
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
from train_sft import build_model_and_tokenizer, train_loop
from data_loader import load_sft_examples

model, tokenizer = build_model_and_tokenizer()
dataset = load_sft_examples().select(range(4))

print(f"Smoke test: {len(dataset)} examples", flush=True)
train_loop(model, tokenizer, dataset)
print("SFT smoke test passed.", flush=True)
PY