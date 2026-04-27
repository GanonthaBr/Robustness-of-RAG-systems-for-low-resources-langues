#!/usr/bin/env python3
"""Post-process robustness run outputs into comparison tables and CSV."""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from utils.helpers import load_json, save_json


NOISE_TYPES = ["irrelevant", "contradictory", "partially_correct", "translation_artifact"]
SEVERITIES = [10, 30, 50]


def _fmt_pct(x):
    return "{:.1f}%".format(100.0 * float(x))


def _parse_condition_name(name):
    if name in ("clean", "llm_only"):
        return name, 0
    for noise in NOISE_TYPES:
        prefix = noise + "_"
        if name.startswith(prefix):
            try:
                sev = int(name[len(prefix):])
            except ValueError:
                sev = 0
            return noise, sev
    return name, 0


def _build_markdown(summary):
    lines = []
    lines.append("# Robustness Post-Run Summary")
    lines.append("")

    for language in summary["languages"]:
        lines.append("## {}".format(language.upper()))
        lines.append("")
        lines.append("| Model | LLM-only | Clean RAG | Delta (Clean-LLM) |")
        lines.append("|---|---:|---:|---:|")
        for model in summary["models"]:
            base = summary["by_language"][language][model]
            lines.append(
                "| {} | {} | {} | {} |".format(
                    model,
                    _fmt_pct(base["llm_only_mean"]),
                    _fmt_pct(base["clean_mean"]),
                    _fmt_pct(base["clean_minus_llm_only"]),
                )
            )
        lines.append("")

        lines.append("| Model | Noise Type | Sev | Correct | Drop vs Clean |")
        lines.append("|---|---|---:|---:|---:|")
        for model in summary["models"]:
            noise_rows = summary["by_language"][language][model]["noise_rows"]
            for row in noise_rows:
                lines.append(
                    "| {} | {} | {} | {} | {} |".format(
                        model,
                        row["noise_type"],
                        row["severity"],
                        _fmt_pct(row["correct_mean"]),
                        _fmt_pct(row["drop_vs_clean"]),
                    )
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(index_file, output_json=None, output_csv=None, output_md=None):
    index = load_json(index_file)
    outputs = index.get("outputs", {})
    if not outputs:
        raise RuntimeError("No model outputs found in robustness index.")

    models = sorted(outputs.keys())
    model_summaries = {}
    languages = set()

    for model in models:
        path = Path(outputs[model])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        summary = load_json(str(path))
        model_summaries[model] = summary
        languages.update(summary.get("by_language", {}).keys())

    languages = sorted(languages)

    combined = {
        "metadata": {
            "source_index": str(index_file),
            "models": models,
        },
        "models": models,
        "languages": languages,
        "by_language": {},
    }

    csv_rows = []

    for language in languages:
        combined["by_language"][language] = {}
        for model in models:
            lang_node = model_summaries[model].get("by_language", {}).get(language, {})
            llm_only_mean = lang_node.get("llm_only", {}).get("correct_rate", {}).get("mean", 0.0)
            clean_mean = lang_node.get("clean", {}).get("correct_rate", {}).get("mean", 0.0)
            clean_minus_llm = float(clean_mean) - float(llm_only_mean)

            noise_rows = []
            for cond_name, cond_data in lang_node.items():
                noise_type, sev = _parse_condition_name(cond_name)
                if noise_type in ("clean", "llm_only"):
                    continue
                correct_mean = float(cond_data.get("correct_rate", {}).get("mean", 0.0))
                drop_vs_clean = float(correct_mean - clean_mean)
                row = {
                    "model": model,
                    "language": language,
                    "noise_type": noise_type,
                    "severity": sev,
                    "correct_mean": correct_mean,
                    "drop_vs_clean": drop_vs_clean,
                }
                noise_rows.append(row)
                csv_rows.append(row)

            noise_rows.sort(key=lambda x: (x["noise_type"], x["severity"]))

            combined["by_language"][language][model] = {
                "llm_only_mean": float(llm_only_mean),
                "clean_mean": float(clean_mean),
                "clean_minus_llm_only": float(clean_minus_llm),
                "noise_rows": noise_rows,
            }

    if output_json is None:
        output_json = PROJECT_ROOT / "results/robustness_postrun_summary.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = PROJECT_ROOT / output_json

    if output_csv is None:
        output_csv = PROJECT_ROOT / "results/robustness_postrun_long.csv"
    else:
        output_csv = Path(output_csv)
        if not output_csv.is_absolute():
            output_csv = PROJECT_ROOT / output_csv

    if output_md is None:
        output_md = PROJECT_ROOT / "results/robustness_postrun_tables.md"
    else:
        output_md = Path(output_md)
        if not output_md.is_absolute():
            output_md = PROJECT_ROOT / output_md

    save_json(combined, str(output_json))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "language", "noise_type", "severity", "correct_mean", "drop_vs_clean"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(combined))

    print("Saved JSON: {}".format(output_json))
    print("Saved CSV: {}".format(output_csv))
    print("Saved Markdown: {}".format(output_md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process robustness outputs")
    parser.add_argument(
        "--index-file",
        type=str,
        default="results/robustness_all_llms_index.json",
        help="Path to robustness all-LLM index JSON",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    main(args.index_file, output_json=args.output_json, output_csv=args.output_csv, output_md=args.output_md)
