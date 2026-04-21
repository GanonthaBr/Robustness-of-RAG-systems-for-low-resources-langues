#!/usr/bin/env python3
"""Build cross-LLM comparison tables from k-sweep multiseed summaries."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from utils.helpers import load_json, save_json


def _fmt_pct(value):
    return "{:.1f}%".format(100.0 * float(value))


def _safe_delta(a, b):
    return float(a) - float(b)


def _extract_model_summary(summary):
    by_lang = {}
    for language, lang_data in summary.get("by_language", {}).items():
        llm_mean = lang_data.get("llm_only_correct_rate", {}).get("mean", 0.0)
        rag_by_k = lang_data.get("rag_by_k", {})

        best_k = None
        best_rag_mean = -1.0
        best_delta_mean = 0.0
        for k, k_data in rag_by_k.items():
            rag_mean = k_data.get("rag_correct_rate", {}).get("mean", 0.0)
            delta_mean = k_data.get("delta_rag_minus_llm_only_correct_rate", {}).get("mean", 0.0)
            if rag_mean > best_rag_mean:
                best_k = str(k)
                best_rag_mean = float(rag_mean)
                best_delta_mean = float(delta_mean)

        by_lang[language] = {
            "llm_only_mean": float(llm_mean),
            "best_k": best_k,
            "best_rag_mean": float(best_rag_mean if best_rag_mean >= 0.0 else 0.0),
            "best_delta_mean": float(best_delta_mean),
        }

    return by_lang


def _render_markdown(comparison):
    lines = []
    lines.append("# Cross-LLM Comparison (Best RAG@k)")
    lines.append("")

    model_keys = comparison.get("model_keys", [])
    if len(model_keys) < 2:
        lines.append("Need at least two models to compare.")
        return "\n".join(lines)

    for language in comparison.get("languages", []):
        lines.append("## {}".format(language.upper()))
        lines.append("")
        lines.append("| Model | LLM-only | Best RAG | Best k | Gain (RAG-LLM) |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in model_keys:
            data = comparison["by_language"][language][model]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    model,
                    _fmt_pct(data["llm_only_mean"]),
                    _fmt_pct(data["best_rag_mean"]),
                    data["best_k"],
                    _fmt_pct(data["best_delta_mean"]),
                )
            )
        lines.append("")

        baseline = model_keys[0]
        for other in model_keys[1:]:
            base_data = comparison["by_language"][language][baseline]
            other_data = comparison["by_language"][language][other]
            llm_gap = _safe_delta(other_data["llm_only_mean"], base_data["llm_only_mean"])
            rag_gap = _safe_delta(other_data["best_rag_mean"], base_data["best_rag_mean"])
            lines.append(
                "- {} vs {}: LLM-only {} | Best-RAG {}".format(
                    other,
                    baseline,
                    _fmt_pct(llm_gap),
                    _fmt_pct(rag_gap),
                )
            )
        lines.append("")

    return "\n".join(lines)


def main(index_file, output_json=None, output_md=None):
    index = load_json(index_file)
    outputs = index.get("outputs", {})
    if len(outputs) < 2:
        raise RuntimeError("Index must contain at least two model outputs.")

    model_keys = sorted(outputs.keys())
    loaded = {}
    languages = set()

    for model_key, path in outputs.items():
        summary_path = Path(path)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        summary = load_json(str(summary_path))
        loaded[model_key] = _extract_model_summary(summary)
        languages.update(loaded[model_key].keys())

    languages = sorted(languages)
    comparison = {
        "metadata": {
            "source_index": str(index_file),
            "models": model_keys,
        },
        "model_keys": model_keys,
        "languages": languages,
        "by_language": {},
    }

    for lang in languages:
        comparison["by_language"][lang] = {}
        for model in model_keys:
            comparison["by_language"][lang][model] = loaded[model].get(
                lang,
                {
                    "llm_only_mean": 0.0,
                    "best_k": None,
                    "best_rag_mean": 0.0,
                    "best_delta_mean": 0.0,
                },
            )

    if output_json is None:
        output_json = PROJECT_ROOT / "results/llm_comparison_from_k_sweep.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = PROJECT_ROOT / output_json

    markdown = _render_markdown(comparison)

    if output_md is None:
        output_md = PROJECT_ROOT / "results/llm_comparison_from_k_sweep.md"
    else:
        output_md = Path(output_md)
        if not output_md.is_absolute():
            output_md = PROJECT_ROOT / output_md

    save_json(comparison, str(output_json))
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown + "\n")

    print("Saved JSON comparison: {}".format(output_json))
    print("Saved Markdown table: {}".format(output_md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple LLM k-sweep summaries")
    parser.add_argument(
        "--index-file",
        type=str,
        default="results/rag_k_sweep_all_llms_multiseed_index.json",
        help="Path to all-LLM index JSON produced by run_rag_k_sweep_multiseed.py --all-llms",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for structured comparison JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional output path for markdown comparison table",
    )
    args = parser.parse_args()

    main(args.index_file, output_json=args.output_json, output_md=args.output_md)
