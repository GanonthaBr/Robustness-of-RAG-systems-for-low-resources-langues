#!/usr/bin/env python3
"""Generate abstention analysis report from robustness checkpoint results."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if sys.version_info[0] < 3:
    raise RuntimeError("report_abstention.py requires Python 3.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from utils.helpers import load_json, save_json


def _model_tag(model_key):
    return str(model_key).replace("/", "-").replace(":", "-")


def collect_abstention_metrics(checkpoint_root, models, languages, seeds):
    """Collect abstention metrics from checkpoint files."""
    results = {
        "metadata": {
            "models": models,
            "languages": languages,
            "seeds": seeds,
            "checkpoint_root": str(checkpoint_root),
        },
        "by_model_language": {},
    }

    for model_key in models:
        llm_tag = _model_tag(model_key)
        results["by_model_language"][model_key] = {}

        for language in languages:
            results["by_model_language"][model_key][language] = {
                "by_condition": {},
                "aggregate": {},
            }

            # Collect metrics for each condition
            condition_data = defaultdict(list)

            for seed in seeds:
                # Try to find the condition files for this seed
                # We'll iterate through all checkpoint files and filter by model/lang/seed
                for checkpoint_file in checkpoint_root.glob(
                    f"robustness_{llm_tag}_{language}_seed{seed}_*.json"
                ):
                    try:
                        data = load_json(str(checkpoint_file))
                        cond_name = data["metadata"]["condition"]
                        abstention = data["metrics"].get("abstention_rate", 0.0)
                        correct = data["metrics"].get("correct_rate", 0.0)
                        num_abstained = data["metrics"].get("num_abstained", 0)
                        num_samples = data["metrics"].get("num_samples", 50)

                        condition_data[cond_name].append(
                            {
                                "abstention_rate": abstention,
                                "correct_rate": correct,
                                "num_abstained": num_abstained,
                                "num_samples": num_samples,
                                "seed": seed,
                            }
                        )
                    except Exception as e:
                        print(f"Warning: Could not load {checkpoint_file}: {e}")

            # Compute statistics per condition
            for cond_name in sorted(condition_data.keys()):
                values = condition_data[cond_name]
                abstention_rates = [v["abstention_rate"] for v in values]
                correct_rates = [v["correct_rate"] for v in values]
                num_abstained_list = [v["num_abstained"] for v in values]

                mean_abstention = sum(abstention_rates) / len(abstention_rates)
                mean_correct = sum(correct_rates) / len(correct_rates)
                mean_abstained = sum(num_abstained_list) / len(num_abstained_list)

                results["by_model_language"][model_key][language]["by_condition"][
                    cond_name
                ] = {
                    "abstention_rate_mean": mean_abstention,
                    "abstention_rate_values": abstention_rates,
                    "correct_rate_mean": mean_correct,
                    "correct_rate_values": correct_rates,
                    "num_abstained_mean": mean_abstained,
                    "num_samples": values[0]["num_samples"] if values else 50,
                }

            # Aggregate across all conditions
            all_abstention_rates = [
                v["abstention_rate_mean"]
                for v in results["by_model_language"][model_key][language][
                    "by_condition"
                ].values()
            ]
            all_correct_rates = [
                v["correct_rate_mean"]
                for v in results["by_model_language"][model_key][language][
                    "by_condition"
                ].values()
            ]

            if all_abstention_rates:
                results["by_model_language"][model_key][language]["aggregate"] = {
                    "mean_abstention_rate": sum(all_abstention_rates)
                    / len(all_abstention_rates),
                    "mean_correct_rate": sum(all_correct_rates)
                    / len(all_correct_rates),
                    "num_conditions": len(all_abstention_rates),
                }

    return results


def generate_latex_table(abstention_results, languages, models):
    """Generate LaTeX table for abstention rates."""
    latex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Abstention Rates (\\%) by Model, Language, and Condition}\n"
    )

    # Build column headers
    col_spec = "l" + "c" * (len(models) * len(languages))
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"

    # Header: languages
    header = "Condition"
    for lang in languages:
        header += f" & \\multicolumn{{{len(models)}}}{{c}}{{{lang.upper()}}}"
    header += " \\\\\n"
    latex += header

    # Subheader: models
    subheader = ""
    for lang in languages:
        for model in models:
            subheader += " & " + model.replace("-", "\\texttt{-}")
    subheader += " \\\\\n"
    latex += subheader
    latex += "\\midrule\n"

    # Rows: conditions
    conditions = [
        "llm_only",
        "clean",
        "irrelevant_10",
        "irrelevant_30",
        "irrelevant_50",
        "contradictory_10",
        "contradictory_30",
        "contradictory_50",
        "partially_correct_10",
        "partially_correct_30",
        "partially_correct_50",
        "translation_artifact_10",
        "translation_artifact_30",
        "translation_artifact_50",
    ]

    for cond in conditions:
        row = cond.replace("_", "\\_")
        for lang in languages:
            for model in models:
                try:
                    abs_rate = (
                        abstention_results["by_model_language"][model][lang][
                            "by_condition"
                        ][cond]["abstention_rate_mean"]
                        * 100
                    )
                    row += f" & {abs_rate:.1f}"
                except KeyError:
                    row += " & --"
        row += " \\\\\n"
        latex += row

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def generate_markdown_report(abstention_results, languages, models):
    """Generate Markdown report for abstention analysis."""
    md = "# Abstention Analysis Report\n\n"

    md += "## Overview\n\n"
    md += "This report summarizes abstention rates across all models, languages, and noise conditions.\n\n"

    md += "## Aggregate Statistics\n\n"
    md += "| Model | Language | Mean Abstention Rate | Mean Correct Rate |\n"
    md += "|-------|----------|----------------------|-------------------|\n"

    for model in models:
        for lang in languages:
            agg = abstention_results["by_model_language"][model][lang].get(
                "aggregate", {}
            )
            mean_abs = agg.get("mean_abstention_rate", 0.0)
            mean_corr = agg.get("mean_correct_rate", 0.0)
            md += f"| {model} | {lang} | {mean_abs:.3f} | {mean_corr:.3f} |\n"

    md += "\n## Detailed Results by Model and Language\n\n"

    for model in models:
        md += f"### Model: {model}\n\n"
        for lang in languages:
            md += f"#### Language: {lang.upper()}\n\n"
            md += "| Condition | Abstention Rate | Correct Rate |\n"
            md += "|-----------|-----------------|---------------|\n"

            by_cond = abstention_results["by_model_language"][model][lang][
                "by_condition"
            ]
            for cond in sorted(by_cond.keys()):
                abs_rate = by_cond[cond]["abstention_rate_mean"]
                corr_rate = by_cond[cond]["correct_rate_mean"]
                md += f"| {cond} | {abs_rate:.3f} | {corr_rate:.3f} |\n"

            md += "\n"

    md += "## Observations\n\n"
    md += (
        "- **Abstention Rate**: Percentage of examples where the model abstained (did not answer).\n"
    )
    md += (
        "- **Correct Rate**: Accuracy among examples where the model provided an answer.\n"
    )
    md += (
        "- Models with higher abstention under high-noise conditions may indicate cautious behavior.\n"
    )
    md += "- Relationship between abstention and correctness reveals confidence calibration quality.\n"

    return md


def main(models, languages, seeds, checkpoint_dir, output_json, output_md, output_latex):
    """Run abstention analysis and generate reports."""
    checkpoint_root = Path(checkpoint_dir)

    if not checkpoint_root.exists():
        print(f"Error: Checkpoint directory {checkpoint_root} does not exist.")
        return

    print("=" * 80)
    print("ABSTENTION ANALYSIS REPORT")
    print("=" * 80)
    print(f"Models     : {models}")
    print(f"Languages  : {languages}")
    print(f"Seeds      : {seeds}")
    print(f"Checkpoint : {checkpoint_root}")

    # Collect metrics
    print("\nCollecting abstention metrics...")
    results = collect_abstention_metrics(checkpoint_root, models, languages, seeds)

    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex_table = generate_latex_table(results, languages, models)

    # Generate Markdown report
    print("Generating Markdown report...")
    md_report = generate_markdown_report(results, languages, models)

    # Save outputs
    if output_json:
        save_json(results, output_json)
        print(f"Saved JSON results: {output_json}")

    if output_md:
        with open(output_md, "w") as f:
            f.write(md_report)
        print(f"Saved Markdown report: {output_md}")

    if output_latex:
        with open(output_latex, "w") as f:
            f.write(latex_table)
        print(f"Saved LaTeX table: {output_latex}")

    print("\nAbstention analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate abstention analysis report from robustness results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["afriqueqwen-8b", "qwen2.5-7b-instruct"],
        help="Model keys to analyze",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["swa", "yor", "kin"],
        help="Languages to analyze",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds to analyze",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: results/robustness_checkpoints)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Output Markdown file path",
    )
    parser.add_argument(
        "--output-latex",
        type=str,
        default=None,
        help="Output LaTeX file path",
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or str(
        PROJECT_ROOT / "results/robustness_checkpoints"
    )
    output_json = args.output_json or str(PROJECT_ROOT / "results/abstention_report.json")
    output_md = args.output_md or str(PROJECT_ROOT / "results/abstention_report.md")
    output_latex = args.output_latex or str(PROJECT_ROOT / "results/abstention_report.tex")

    main(
        models=args.models,
        languages=args.languages,
        seeds=args.seeds,
        checkpoint_dir=checkpoint_dir,
        output_json=output_json,
        output_md=output_md,
        output_latex=output_latex,
    )
