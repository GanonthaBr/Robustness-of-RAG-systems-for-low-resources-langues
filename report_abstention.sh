#!/bin/bash
#
# Launch abstention analysis report on HPC/VM
#
# Usage:
#   bash report_abstention.sh
#   bash report_abstention.sh <PROJECT_ROOT>
#

set -euo pipefail

# Allow override of PROJECT_ROOT via argument
PROJECT_ROOT="${1:-.}"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: PROJECT_ROOT '$PROJECT_ROOT' does not exist"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "=========================================="
echo "ABSTENTION ANALYSIS REPORT"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate rag-robust

# Check Python
echo "Python: $(python --version)"
echo

# Run the abstention analysis script
echo "Generating abstention analysis report..."
python3 scripts/report_abstention.py \
    --models afriqueqwen-8b qwen2.5-7b-instruct \
    --languages swa yor kin \
    --seeds 42 43 44 \
    --checkpoint-dir "$PROJECT_ROOT/results/robustness_checkpoints" \
    --output-json "$PROJECT_ROOT/results/abstention_report.json" \
    --output-md "$PROJECT_ROOT/results/abstention_report.md" \
    --output-latex "$PROJECT_ROOT/results/abstention_report.tex"

echo
echo "=========================================="
echo "Report Complete!"
echo "=========================================="
echo "Outputs:"
echo "  JSON:   $PROJECT_ROOT/results/abstention_report.json"
echo "  Markdown: $PROJECT_ROOT/results/abstention_report.md"
echo "  LaTeX:  $PROJECT_ROOT/results/abstention_report.tex"
echo
