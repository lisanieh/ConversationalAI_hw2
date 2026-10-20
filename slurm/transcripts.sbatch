#!/bin/bash
# Usage: sbatch slurm/transcripts.sbatch

#SBATCH --job-name=a2-transcript
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
export CUDA_LAUNCH_BLOCKING=1

cd "$SLURM_SUBMIT_DIR"

python -u - <<'PY'
from configs.paths_config import dpo_checkpoint_dir
from evaluate import generate_responses, load_adapter_policy

beta = 0.05
model = load_adapter_policy(dpo_checkpoint_dir(beta))
model.eval()

print(f"Loaded DPO beta={beta}", flush=True)
responses = generate_responses(model, tokenizer, prompts)

for item, response in zip(items, responses):
    print(f"\n[{item['id']}] {item['prompt']}", flush=True)
    print(f"DPO beta={beta}: {response}", flush=True)
PY