#!/bin/bash
#
# Ready to run.
# Submits a Part A LoRA SFT training run on PACE-ICE.
#
# Usage: sbatch slurm/train_sft.sbatch

#SBATCH --job-name=a2-sft
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate ~/scratch/conda/envs/CAI_A2

export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

# Requires data/preference_pairs.jsonl (run generate_preference_data.py first).
python train_sft.py \
    --checkpoint-dir "checkpoints/sft" \
    --seed 42
