#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --account=cis260093p
#SBATCH -J rag-abstention
#SBATCH -o /ocean/projects/cis260093p/gpayang/robustness-of-rag/slurm-%j.out
#SBATCH -e /ocean/projects/cis260093p/gpayang/robustness-of-rag/slurm-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="/ocean/projects/cis260093p/gpayang/robustness-of-rag"
OCEAN_ROOT="/ocean/projects/cis260093p/gpayang"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# HuggingFace cache → ocean (avoids filling 25GB /jet home quota)
# ---------------------------------------------------------------------------
export HF_HOME="${OCEAN_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${OCEAN_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${OCEAN_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${OCEAN_ROOT}/hf_cache"
mkdir -p "${OCEAN_ROOT}/hf_cache"

# ---------------------------------------------------------------------------
# Logs dir
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_ROOT}/logs"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
source ~/.bashrc
conda activate rag-robust

echo "============================================================"
echo "JOB ID       : $SLURM_JOB_ID"
echo "NODE         : $SLURMD_NODENAME"
echo "START        : $(date)"
echo "PROJECT ROOT : $PROJECT_ROOT"
echo "HF CACHE     : $HF_HOME"
echo "PYTHON       : $(which python)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verify CUDA is accessible
# ---------------------------------------------------------------------------
python -c "
import torch
print('CUDA available :', torch.cuda.is_available())
print('GPU            :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print('VRAM total     : {:.1f} GB'.format(total))
"

# ---------------------------------------------------------------------------
# Run full abstention analysis
# ---------------------------------------------------------------------------
python scripts/evaluate_abstention_full.py \
    --all-llms \
    --num-examples 50 \
    --seeds 42 43 44 \
    --languages swa yor kin \
    --embedding-model e5-base \
    --output-dir "${PROJECT_ROOT}/results/abstention_analysis" \
    2>&1 | tee "${PROJECT_ROOT}/logs/abstention_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "END : $(date)"
echo "============================================================"
