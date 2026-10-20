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
import torch

from configs.paths_config import SFT_CHECKPOINT_DIR
from data_loader import load_ambiguous_set, load_clear_set
from evaluate import generate_responses, load_adapter_policy
from utils.model_utils import load_tokenizer

items = load_ambiguous_set()[:2] + load_clear_set()[:1]
prompts = [item["prompt"] for item in items]

print("Loading tokenizer...", flush=True)
tokenizer = load_tokenizer(SFT_CHECKPOINT_DIR)

print("Loading SFT adapter...", flush=True)
model = load_adapter_policy(SFT_CHECKPOINT_DIR)
model.eval()

print("Generating SFT responses...", flush=True)
responses = generate_responses(model, tokenizer, prompts)
torch.cuda.synchronize()

for item, response in zip(items, responses):
    print(f"\n[{item['id']}] {item['prompt']}", flush=True)
    print(f"SFT: {response}", flush=True)

print("\nSFT transcript test passed.", flush=True)
PY