#!/usr/bin/env python3
"""
Generate JSD-vs-throughput plots from CSV results.

Reads CSVs from csv/ and produces PDF plots in figures/.

Adapted from benchmarking/plotting/plotting_legend_short_combined.py.

Figures generated:
  - figures/jsd_vs_speed_realpha_ptb.pdf  (Figure 5: combined realpha + PTB)
  - figures/jsd_vs_speed_dna2aa.pdf       (DNA JSD-vs-throughput)
  - figures/jsd_vs_speed_realpha.pdf      (standalone realpha)
  - figures/jsd_vs_speed_ptb.pdf          (standalone PTB)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style constants (matching paper figures)
# ---------------------------------------------------------------------------

BASE_FONTSIZE = 30
TAU_MARKERS = ["*", "o", "s", "D", "^", "v", "P", "X", "h", ">", "<", "d", "p"]
PALETTE = plt.get_cmap("Set2").colors


def fmt_e(x, digits=1, *, show_plus=False, trim_mantissa=True):
    """Format number in compact scientific notation."""
    if pd.isna(x):
        return str(x)
    x = float(x)
    if x == 0.0:
        return f"{0:.{digits}f}" if digits else "0"
    m, e = f"{x:.{digits}e}".split("e")
    if trim_mantissa:
        m = m.rstrip("0").rstrip(".")
    exp = int(e)
    sign = "+" if (show_plus and exp >= 0) else ""
    return f"{m}e{sign}{exp}"


def fmt_x_string(x, digits=0, *, style="thousands"):
    """Format x-axis (speed) values."""
    if not np.isfinite(x):
        return ""
    s = f"{int(round(x)):,}".replace(",", r"\,")
    return str(s)


def mpl_style():
    """Context manager for paper-quality matplotlib style."""
    return plt.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_FONTSIZE,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": True,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "grid.alpha": 0.25,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.figsize": (18, 3),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble": r"""
        \usepackage{amsmath,amssymb,xcolor}
        \usepackage{inconsolata}
        \renewcommand{\rmdefault}{ptm}
        """,
        "pgf.rcfonts": False,
    })


# ---------------------------------------------------------------------------
# Style maps
# ---------------------------------------------------------------------------


def build_style_maps(df: pd.DataFrame):
    """Build consistent tau -> color and tau -> marker mappings."""
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    taus = np.asarray(sorted(df["K"].dropna().unique()))
    n = len(taus)
    cmap = plt.get_cmap("viridis", n)
    colors = cmap(np.linspace(0, 1, n))

    klabel2color = {fmt_e(k, digits=0): colors[i] for i, k in enumerate(taus)}
    tau2marker = {
        fmt_e(k, digits=0): TAU_MARKERS[i % len(TAU_MARKERS)]
        for i, k in enumerate(taus)
    }
    return klabel2color, tau2marker


def build_union_style_maps(df_list: list[pd.DataFrame]):
    """Unified style maps across multiple DataFrames."""
    df_all = pd.concat(df_list, ignore_index=True)
    return build_style_maps(df_all)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_on_axes(
    df: pd.DataFrame,
    axes_list,
    klabel2color: dict,
    tau2marker: dict,
):
    """Plot JSD-vs-throughput scatter on provided axes (one per model)."""
    models = sorted(df["model"].unique())
    legend_seen = set()

    for ax, model in zip(axes_list, models):
        d = df[df["model"] == model].sort_values("chars_per_sec")

        # Backbone line
        ax.plot(
            d["chars_per_sec"], d["mean_metric"],
            color="0.65", lw=2.5, label="_nolegend_", zorder=1,
        )

        # Error bars
        ax.errorbar(
            d["chars_per_sec"], d["mean_metric"],
            yerr=[
                d["mean_metric"] - d["metric_ci_lower"],
                d["metric_ci_upper"] - d["mean_metric"],
            ],
            fmt="none", ecolor="0.4", elinewidth=3, capsize=4, zorder=0,
        )

        # Points per tau
        for xi, yi, lab in zip(d["chars_per_sec"], d["mean_metric"], d["K_label"]):
            if np.isnan(xi) or np.isnan(yi):
                continue
            m = tau2marker.get(lab, "o")
            ax.scatter(
                [xi], [yi], s=80, marker=m,
                facecolors=klabel2color.get(lab, "gray"),
                edgecolors="black", linewidths=1.8, zorder=3,
                label=(
                    rf"$\tau\mkern-3mu=$" + lab
                    if lab not in legend_seen
                    else "_nolegend_"
                ),
            )
            legend_seen.add(lab)

        ax.set_title(model, fontweight="bold", pad=2, y=1.2, fontsize=BASE_FONTSIZE - 2)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(4))
        ax.yaxis.set_major_locator(mtick.MaxNLocator(4))
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: fmt_e(x, digits=1))
        )
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: fmt_x_string(x, digits=2, style="thousands"))
        )


def add_legend(fig, axes_list):
    """Collect and add a unified legend above the figure."""
    H, L = [], []
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll != "_nolegend_" and ll not in L:
                H.append(hh)
                L.append(ll)
    if H:
        fig.legend(
            H, L, ncols=len(L),
            loc="upper center", bbox_to_anchor=(0.5, 1.22),
            frameon=False, handletextpad=0.01,
            labelspacing=0.01, borderaxespad=0.0,
            scatterpoints=1,
        )


# ---------------------------------------------------------------------------
# High-level plot functions
# ---------------------------------------------------------------------------


def load_and_prep(csv_path: str, max_K: float | None = None) -> pd.DataFrame:
    """Load CSV, add K_label column, optionally filter by max K."""
    df = pd.read_csv(csv_path)
    df["mean_metric"] = pd.to_numeric(df["mean_metric"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["K_label"] = df["K"].map(lambda v: fmt_e(v, digits=0))
    # Filter out reference row (NaN metric) and optionally cap K
    df = df.dropna(subset=["mean_metric"])
    if max_K is not None:
        df = df[df["K"] < max_K]
    return df


def plot_single(df: pd.DataFrame, out_path: Path, title: str = ""):
    """Plot JSD-vs-throughput for a single experiment."""
    models = sorted(df["model"].unique())
    klabel2color, tau2marker = build_style_maps(df)

    with mpl_style():
        fig, axes = plt.subplots(
            1, len(models), sharey=True, constrained_layout=True,
        )
        if len(models) == 1:
            axes = [axes]

        plot_on_axes(df, axes, klabel2color, tau2marker)

        axes[0].set_ylabel("JSD")
        fig.supxlabel("Speed (bytes per second)", fontsize=BASE_FONTSIZE)
        add_legend(fig, axes)

        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


def plot_combined_realpha_ptb(
    df_realpha: pd.DataFrame,
    df_ptb: pd.DataFrame,
    out_path: Path,
):
    """Plot combined realpha + PTB: 6 panels, shared y within each group."""
    klabel2color, tau2marker = build_union_style_maps([df_realpha, df_ptb])

    models_r = sorted(df_realpha["model"].unique())
    models_p = sorted(df_ptb["model"].unique())
    ncols = len(models_r) + len(models_p)

    with mpl_style():
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, ncols, wspace=0)

        # ReAlpha group (shared y)
        axes_r = [fig.add_subplot(gs[0, 0])]
        for i in range(1, len(models_r)):
            axes_r.append(fig.add_subplot(gs[0, i], sharey=axes_r[0]))

        # PTB group (separate y)
        axes_p = [fig.add_subplot(gs[0, len(models_r)])]
        for i in range(1, len(models_p)):
            axes_p.append(fig.add_subplot(gs[0, len(models_r) + i], sharey=axes_p[0]))

        plot_on_axes(df_realpha, axes_r, klabel2color, tau2marker)
        plot_on_axes(df_ptb, axes_p, klabel2color, tau2marker)

        # Hide redundant y labels
        for ax in axes_r[1:] + axes_p[1:]:
            ax.tick_params(labelleft=False)

        fig.supylabel("JSD", fontsize=BASE_FONTSIZE)
        fig.supxlabel("Speed (bytes per second)", fontsize=BASE_FONTSIZE)
        add_legend(fig, axes_r + axes_p)

        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    script_dir = Path(__file__).resolve().parent
    csv_dir = script_dir / "csv"
    fig_dir = script_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    generated = []

    # Standalone: realpha
    path_r = csv_dir / "realpha_jsd.csv"
    if path_r.exists():
        df_r = load_and_prep(str(path_r))
        out = fig_dir / "jsd_vs_speed_realpha.pdf"
        plot_single(df_r, out, title="Tokens to Characters")
        generated.append(out)

    # Standalone: PTB
    path_p = csv_dir / "ptb_jsd.csv"
    if path_p.exists():
        df_p = load_and_prep(str(path_p))
        out = fig_dir / "jsd_vs_speed_ptb.pdf"
        plot_single(df_p, out, title="Tokens to Words")
        generated.append(out)

    # Combined: realpha + PTB (Figure 5)
    if path_r.exists() and path_p.exists():
        df_r = load_and_prep(str(path_r))
        df_p = load_and_prep(str(path_p))
        out = fig_dir / "jsd_vs_speed_realpha_ptb.pdf"
        plot_combined_realpha_ptb(df_r, df_p, out)
        generated.append(out)

    # Standalone: DNA
    path_d = csv_dir / "dna2aa_jsd.csv"
    if path_d.exists():
        df_d = load_and_prep(str(path_d))
        out = fig_dir / "jsd_vs_speed_dna2aa.pdf"
        plot_single(df_d, out, title="DNA to Amino Acids")
        generated.append(out)

    if not generated:
        print("No CSV files found in csv/. Run analyze.py first.")
    else:
        print(f"\nGenerated {len(generated)} figures in figures/")


if __name__ == "__main__":
    main()
