#!/bin/bash

#SBATCH --job-name=compaction-markovian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --output=../scratch/outputs/compaction_markovian_mila/job_%j.log
#SBATCH --error=../scratch/outputs/compaction_markovian_mila/job_%j.log

export PROJECT_DIR=/home/mila/e/emiliano.penaloza/rsa-compaction

cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

uv run rl @ configs/compaction/qwen3_4b_markovian_mila.toml
