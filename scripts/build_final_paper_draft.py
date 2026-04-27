#!/usr/bin/env python3
"""Build a final paper-ready markdown draft from clean and robustness summaries."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from utils.helpers import load_json


def _fmt_pct(x):
    return "{:.1f}%".format(100.0 * float(x))


def main(clean_file, robustness_file, output_file):
    clean = load_json(clean_file)
    robustness = load_json(robustness_file)

    models = clean.get("model_keys", clean.get("models", []))
    languages = clean.get("languages", [])

    lines = []
    lines.append("# Final Results Draft")
    lines.append("")
    lines.append("## Clean Comparison (Best RAG@k)")
    lines.append("")

    for lang in languages:
        lines.append("### {}".format(lang.upper()))
        lines.append("")
        lines.append("| Model | LLM-only | Best RAG | Best k | Gain |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in models:
            node = clean["by_language"][lang][model]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    model,
                    _fmt_pct(node.get("llm_only_mean", 0.0)),
                    _fmt_pct(node.get("best_rag_mean", 0.0)),
                    node.get("best_k", "-"),
                    _fmt_pct(node.get("best_delta_mean", 0.0)),
                )
            )
        lines.append("")

    lines.append("## Robustness Summary")
    lines.append("")

    robust_langs = robustness.get("languages", [])
    robust_models = robustness.get("models", [])

    for lang in robust_langs:
        lines.append("### {}".format(lang.upper()))
        lines.append("")
        lines.append("| Model | LLM-only | Clean RAG | Delta (Clean-LLM) |")
        lines.append("|---|---:|---:|---:|")
        for model in robust_models:
            node = robustness["by_language"][lang][model]
            lines.append(
                "| {} | {} | {} | {} |".format(
                    model,
                    _fmt_pct(node.get("llm_only_mean", 0.0)),
                    _fmt_pct(node.get("clean_mean", 0.0)),
                    _fmt_pct(node.get("clean_minus_llm_only", 0.0)),
                )
            )
        lines.append("")

        lines.append("| Model | Worst Drop vs Clean |")
        lines.append("|---|---:|")
        for model in robust_models:
            rows = robustness["by_language"][lang][model].get("noise_rows", [])
            worst = min([r.get("drop_vs_clean", 0.0) for r in rows], default=0.0)
            lines.append("| {} | {} |".format(model, _fmt_pct(worst)))
        lines.append("")

    out = Path(output_file)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved final paper draft markdown: {}".format(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build final paper-ready markdown from result summaries")
    parser.add_argument(
        "--clean-file",
        type=str,
        default="results/llm_comparison_from_k_sweep.json",
        help="Cross-LLM clean comparison JSON",
    )
    parser.add_argument(
        "--robustness-file",
        type=str,
        default="results/robustness_postrun_summary.json",
        help="Post-processed robustness summary JSON",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/final_paper_results_draft.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    main(args.clean_file, args.robustness_file, args.output_file)
