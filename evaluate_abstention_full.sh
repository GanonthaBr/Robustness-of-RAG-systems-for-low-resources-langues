#!/bin/bash
#
# Launch full abstention analysis on HPC/VM
#
# Usage:
#   bash evaluate_abstention_full.sh
#   bash evaluate_abstention_full.sh <PROJECT_ROOT>
#

set -euo pipefail

PROJECT_ROOT="${1:-.}"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: PROJECT_ROOT '$PROJECT_ROOT' does not exist"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=========================================="
echo "FULL ABSTENTION ANALYSIS"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo

# Activate conda
source ~/.bashrc
conda activate rag-robust

# Check GPU
echo "Checking GPU..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo

# Run full abstention analysis
echo "Running full abstention analysis..."
python3 scripts/evaluate_abstention_full.py \
    --all-llms \
    --num-examples 50 \
    --seeds 42 43 44 \
    --languages swa yor kin \
    --embedding-model e5-base \
    --output-dir "$PROJECT_ROOT/results/abstention_analysis"

echo
echo "=========================================="
echo "Full Abstention Analysis Complete!"
echo "=========================================="
echo "Outputs in: $PROJECT_ROOT/results/abstention_analysis/"
echo
