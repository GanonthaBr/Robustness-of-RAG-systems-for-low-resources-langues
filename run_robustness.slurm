#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J rag-robustness
#SBATCH -o /ocean/projects/cis260093p/youruser/robustness-of-rag/slurm-%j.out
#SBATCH -e /ocean/projects/cis260093p/youruser/robustness-of-rag/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260093p/youruser/robustness-of-rag"
cd "$PROJECT_ROOT"

# Activate conda
source ~/.bashrc
conda activate rag-robust

# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Run the robustness script (all LLMs, batch size 8)
python scripts/run_robustness_multiseed.py --all-llms --batch-size 8
