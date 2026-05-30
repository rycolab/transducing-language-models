#!/usr/bin/env python3
"""
Generate LaTeX tables from CSV results.

Reads CSVs from csv/ and produces .tex files in tables/.

Adapted from benchmarking/utils/scoring_utils.py:create_latex_table().

Tables generated:
  - tables/table_realpha_jsd.tex   (Table 4: realpha JSD)
  - tables/table_ptb_jsd.tex       (Table 5: PTB JSD)
  - tables/table_dna2aa_jsd.tex    (Table 6: DNA JSD)
  - tables/table_realpha_ce.tex    (Table 8: realpha cross-entropy)
  - tables/table_baseline_jsd.tex  (Table 3: Vieira baseline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DISPLAY = {
    "gpt2-large": "GPT2-Large",
    "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
    "gpt2-dna (max_cand=5000)": r"DNA (5k)",
    "gpt2-dna (max_cand=10000)": r"DNA (10k)",
    "gpt2-dna (max_cand=15000)": r"DNA (15k)",
    "gpt2-dna (max_cand=20000)": r"DNA (20k)",
}

NOT_APPLICABLE = r"\textcolor{black!40}{(n/a)}"


def fmt_e(x, digits=1):
    """Format a number in scientific notation."""
    if pd.isna(x):
        return NOT_APPLICABLE
    return f"{x:.{digits}e}"


def fmt_speed(x):
    """Format throughput as integer."""
    if pd.isna(x):
        return NOT_APPLICABLE
    return f"{x:.0f}"


def create_latex_table(
    df: pd.DataFrame,
    metric_name: str = "JSD",
    models_per_table: int = 2,
) -> str:
    """Generate a LaTeX table from a results DataFrame.

    Args:
        df: DataFrame with columns: K, mean_metric, metric_ci_lower,
            metric_ci_upper, chars_per_sec, speed_ci_lower,
            speed_ci_upper, model.
        metric_name: Label for the metric column header.
        models_per_table: Max models per table block (split into multiple
            tables if more models than this).

    Returns:
        LaTeX string for the table.
    """
    df = df.copy()
    cols_per_model = 2  # metric + speed

    # Format columns
    df["Metric"] = df.apply(
        lambda x: (
            NOT_APPLICABLE
            if pd.isna(x["mean_metric"])
            else f"{fmt_e(x['mean_metric'])} "
                 f"({fmt_e(x['metric_ci_lower'])}, {fmt_e(x['metric_ci_upper'])})"
        ),
        axis=1,
    )
    df["Speed"] = df.apply(
        lambda x: (
            NOT_APPLICABLE
            if pd.isna(x.get("chars_per_sec"))
            else f"{x['chars_per_sec']:.0f} "
                 f"({x['speed_ci_lower']:.0f}, {x['speed_ci_upper']:.0f})"
        ),
        axis=1,
    )

    # Sort thresholds descending (largest first)
    tau_values = sorted(df["K"].unique(), reverse=True)

    # Determine model order
    present = list(pd.unique(df["model"]))
    ordered = [m for m in MODEL_DISPLAY if m in present]
    extras = [m for m in present if m not in MODEL_DISPLAY]
    ordered += extras

    def disp(m):
        return MODEL_DISPLAY.get(m, m)

    parts = []
    for start in range(0, len(ordered), max(1, models_per_table)):
        chunk = ordered[start:start + max(1, models_per_table)]
        colspec = (
            "c|"
            + ("c" * cols_per_model + "|") * (len(chunk) - 1)
            + "c" * cols_per_model
        )

        parts.append(f"\\begin{{tabular}}{{{colspec}}}\n")
        parts.append("\\toprule\n")

        # Model header
        header_cells = [
            f"\\multicolumn{{{cols_per_model}}}{{c}}{{\\textbf{{{disp(m)}}}}}"
            for m in chunk
        ]
        parts.append(" & " + " & ".join(header_cells) + " \\\\\n")

        # Sub-header
        subhdr = []
        for _ in chunk:
            subhdr.extend([f"avg {metric_name}/byte", "byte/sec"])
        parts.append("$\\tau$ & " + " & ".join(subhdr) + " \\\\\n")
        parts.append("\\midrule\n")

        # Data rows
        for k in tau_values:
            row = [str(k)]
            for m in chunk:
                model_data = df[(df["model"] == m) & (df["K"] == k)]
                if not model_data.empty:
                    metric_str = model_data["Metric"].iloc[0]
                    speed_str = model_data["Speed"].iloc[0]
                else:
                    metric_str = NOT_APPLICABLE
                    speed_str = NOT_APPLICABLE
                row.extend([metric_str, speed_str])
            parts.append(" & ".join(row) + " \\\\\n")

        parts.append("\\bottomrule\n\\end{tabular}\n\n")

    return "".join(parts)


def create_ce_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table for cross-entropy results.

    Columns: tau | byte/sec | bits/byte for each model.
    """
    df = df.copy()
    cols_per_model = 2  # bits/byte + speed

    df["CE"] = df.apply(
        lambda x: (
            NOT_APPLICABLE
            if pd.isna(x["mean_metric"])
            else f"{x['mean_metric']:.4f} "
                 f"({x['metric_ci_lower']:.4f}, {x['metric_ci_upper']:.4f})"
        ),
        axis=1,
    )
    df["Speed"] = df.apply(
        lambda x: (
            NOT_APPLICABLE
            if pd.isna(x.get("chars_per_sec"))
            else f"{x['chars_per_sec']:.0f} "
                 f"({x['speed_ci_lower']:.0f}, {x['speed_ci_upper']:.0f})"
        ),
        axis=1,
    )

    tau_values = sorted(df["K"].unique(), reverse=True)
    present = list(pd.unique(df["model"]))
    ordered = [m for m in MODEL_DISPLAY if m in present]
    extras = [m for m in present if m not in MODEL_DISPLAY]
    ordered += extras

    def disp(m):
        return MODEL_DISPLAY.get(m, m)

    parts = []
    colspec = "c|" + ("cc|" * (len(ordered) - 1)) + "cc"
    parts.append(f"\\begin{{tabular}}{{{colspec}}}\n")
    parts.append("\\toprule\n")

    header_cells = [
        f"\\multicolumn{{{cols_per_model}}}{{c}}{{\\textbf{{{disp(m)}}}}}"
        for m in ordered
    ]
    parts.append(" & " + " & ".join(header_cells) + " \\\\\n")

    subhdr = []
    for _ in ordered:
        subhdr.extend(["bits/byte", "byte/sec"])
    parts.append("$\\tau$ & " + " & ".join(subhdr) + " \\\\\n")
    parts.append("\\midrule\n")

    for k in tau_values:
        row = [str(k)]
        for m in ordered:
            model_data = df[(df["model"] == m) & (df["K"] == k)]
            if not model_data.empty:
                ce_str = model_data["CE"].iloc[0]
                speed_str = model_data["Speed"].iloc[0]
            else:
                ce_str = NOT_APPLICABLE
                speed_str = NOT_APPLICABLE
            row.extend([ce_str, speed_str])
        parts.append(" & ".join(row) + " \\\\\n")

    parts.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(parts)


def main():
    script_dir = Path(__file__).resolve().parent
    csv_dir = script_dir / "csv"
    tables_dir = script_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    generated = []

    # Table 4: realpha JSD
    path = csv_dir / "realpha_jsd.csv"
    if path.exists():
        df = pd.read_csv(path)
        tex = create_latex_table(df, metric_name="JSD")
        out = tables_dir / "table_realpha_jsd.tex"
        out.write_text(tex)
        generated.append(out)
        print(f"Wrote {out}")

    # Table 5: PTB JSD
    path = csv_dir / "ptb_jsd.csv"
    if path.exists():
        df = pd.read_csv(path)
        tex = create_latex_table(df, metric_name="JSD")
        out = tables_dir / "table_ptb_jsd.tex"
        out.write_text(tex)
        generated.append(out)
        print(f"Wrote {out}")

    # Table 6: DNA JSD
    path = csv_dir / "dna2aa_jsd.csv"
    if path.exists():
        df = pd.read_csv(path)
        tex = create_latex_table(df, metric_name="JSD", models_per_table=4)
        out = tables_dir / "table_dna2aa_jsd.tex"
        out.write_text(tex)
        generated.append(out)
        print(f"Wrote {out}")

    # Table 8: realpha cross-entropy
    path = csv_dir / "realpha_ce.csv"
    if path.exists():
        df = pd.read_csv(path)
        tex = create_ce_table(df)
        out = tables_dir / "table_realpha_ce.tex"
        out.write_text(tex)
        generated.append(out)
        print(f"Wrote {out}")

    # Table 3: Vieira baseline
    path = csv_dir / "baseline_jsd.csv"
    if path.exists():
        df = pd.read_csv(path)
        tex = create_latex_table(df, metric_name="JSD")
        out = tables_dir / "table_baseline_jsd.tex"
        out.write_text(tex)
        generated.append(out)
        print(f"Wrote {out}")

    if not generated:
        print("No CSV files found in csv/. Run analyze.py first.")
    else:
        print(f"\nGenerated {len(generated)} LaTeX tables in tables/")


if __name__ == "__main__":
    main()
