#!/bin/bash

#SBATCH --job-name=babyai-turn-am
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --output=../scratch/outputs/balrog-babyai-turn-am/job_%j.log
#SBATCH --error=../scratch/outputs/balrog-babyai-turn-am/job_%j.log

export PROJECT_DIR=/home/mila/e/emiliano.penaloza/rsa-compaction

cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

uv run rl @ configs/compaction/qwen3_4b_turn_compaction_babyai_am.toml
