#!/bin/bash
# Usage: sbatch slurm/test_dpo.sbatch

#SBATCH --job-name=a2-dpo-test
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
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
import os

from data_loader import load_dpo_examples
from train_dpo import build_policy_and_tokenizer, train_loop

sft_checkpoint = "checkpoints/sft"
policy, tokenizer = build_policy_and_tokenizer(sft_checkpoint)

dataset = load_dpo_examples().select(range(4))
print(f"DPO smoke test: {len(dataset)} examples, beta=0.1", flush=True)

train_loop(policy, tokenizer, dataset, beta=0.1)

output_dir = f"checkpoints/dpo/smoke_test_{os.environ['SLURM_JOB_ID']}"
os.makedirs(output_dir, exist_ok=True)
policy.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"DPO smoke test passed; adapter saved to {output_dir}", flush=True)
PY