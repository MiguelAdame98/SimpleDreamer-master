# ============================================
# Paper-ready Visualization: Coverage % (bar)
# + Steps to Full Coverage (scatter + medians)
# Compare 3 models across 12/25/36-room environments
# Pure Matplotlib, no extra deps.
# ============================================

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --------------------------------------------
# CONFIG (tweak these)
# --------------------------------------------
SAVE_FIGS = True        # write PNG+PDF+SVG if True
OUTDIR    = "figs"
DPI       = 400
SEED      = 42
USE_TEX   = True        # set True if you want LaTeX text rendering

# Exact labels requested (order is used everywhere)
MODEL_ORDER = [
    "Full model (HMM+Cognitive graph+Dreamer)",
    "Only Dreamer",
    "Full model- no bias",
]

# Colorblind-friendly palette (consistent across figures)
COLORS = {
    "Full model (HMM+Cognitive graph+Dreamer)": "#1f77b4",  # tab:blue
    "Only Dreamer": "#ff7f0e",                               # tab:orange
    "Full model- no bias": "#2ca02c",                        # tab:green
}

# Distinct hatches to survive grayscale printing
HATCHES = {
    "Full model (HMM+Cognitive graph+Dreamer)": "///",
    "Only Dreamer": "\\\\",
    "Full model- no bias": "xx",
}

# Distinct markers per model (used in steps figure)
MARKERS = {
    "Full model (HMM+Cognitive graph+Dreamer)": "o",
    "Only Dreamer": "s",
    "Full model- no bias": "^",
}

# X axis categories (shared)
ENV_LABELS = ["12 rooms", "25 rooms", "36 rooms"]
ENV_SIZES  = np.array([12, 25, 36], dtype=float)

rng = np.random.default_rng(SEED)


# --------------------------------------------
# Paper theme (clean, journal-friendly)
# --------------------------------------------
def set_paper_theme():
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
        "text.usetex": USE_TEX,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#000000",
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =========================================================
# FIGURE 1: Percentage of explored rooms while in budget
# Grouped bars: 3 bars (models) per environment
# =========================================================


coverage_pct = {
    "Only Dreamer": np.array([100.0, 38.0, 22.0]),
    "Full model- no bias": np.array([95.0, 64.0, 27.0]),
    "Full model (HMM+Cognitive graph+Dreamer)": np.array([100.0, 100.0, 96.0]),
}
# ------------------------------------------------------------

def _export(fig, name):
    if not SAVE_FIGS:
        return
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, name)
    fig.savefig(base + ".png")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")

def _annotate_panel(ax, label):
    ax.text(
        0.0, 1.02, label,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=14, weight="bold"
    )

def plot_percentage_bars(coverage_pct_dict, panel_label="A"):
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.set_axisbelow(True)

    x = np.arange(len(ENV_LABELS))
    n_models = len(MODEL_ORDER)
    width = 0.22
    gap = 0.02

    # draw bars
    for i, model in enumerate(MODEL_ORDER):
        offset = (i - (n_models - 1) / 2.0) * (width + gap)
        y = np.asarray(coverage_pct_dict[model], dtype=float)

        rects = ax.bar(
            x + offset, y, width,
            label=model,
            color=COLORS[model],
            hatch=HATCHES[model],
            edgecolor="black",
            linewidth=0.8,
        )

        # Value labels on each bar
        labels = [f"{v:.0f}%" for v in y]
        ax.bar_label(rects, labels=labels, padding=3, fontsize=10)

    # cosmetics
    ax.set_title("Percentage of explored rooms while in budget", pad=10)
    ax.set_ylabel("Percentage of completion")
    ax.set_xlabel("Environment")
    ax.set_xticks(x, ENV_LABELS)
    ax.set_ylim(0, 105)
    ax.margins(x=0.02)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="#888888")
    leg.get_frame().set_alpha(0.9)

    _annotate_panel(ax, panel_label)
    fig.tight_layout()
    _export(fig, "percentage_exploration_by_env")
    plt.show()


# =========================================================
# FIGURE 2: Total steps until full coverage
# Per-run scatter (jitter) + faint paired lines + bold median per model
# =========================================================

# ---------- REPLACE WITH YOUR DATA (steps to full coverage) ----------
# ---------- EXPLICIT DATA ARRAYS (steps to full coverage) ----------
# Columns = [Env1, Env2, Env3]
# ---------- EXPLICIT DATA ARRAYS (steps to full coverage) ----------
# Columns = [Env1, Env2, Env3]
coverage_steps_by_model = {
    "Full model (HMM+Cognitive graph+Dreamer)": np.array([
        [155, 938, 1267],
        [269, 946, 1521],
        [202, 560, 2241],
        [151, 551, 1290],
        [174, 584, 2431],
        [159, 1124, 1789],
        [261, 887, 1657],
        [229, 657, 1820],

    ], dtype=float),

    "Only Dreamer": np.array([
        [252, 2140, 4779],
        [189, 2338, 5694],

    ], dtype=float),

    "Full model- no bias": np.array([
        [312, 1512, 5901],
        [246, 2763, 5870],
        [334, 1831, 5202],


    ], dtype=float),
}
# ------------------------------------------------------------

# Optional: bootstrap CI for medians (falls back to IQR if SciPy missing)
try:
    from scipy.stats import bootstrap
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _median_ci(x, n_resamples=4000, confidence_level=0.95, seed=SEED):
    x = np.asarray(x, dtype=float)
    if not _HAS_SCIPY:
        q1, q3 = np.percentile(x, [25, 75])
        return q1, q3
    rng_local = np.random.default_rng(seed)
    res = bootstrap((x,), np.median, vectorized=False,
                    n_resamples=n_resamples, confidence_level=confidence_level,
                    method="BCa", random_state=rng_local)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)

def plot_steps_scatter_lines(steps_by_model, panel_label="B",
                             show_paired_lines=True, show_ci_bands=True):
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    ax.set_axisbelow(True)
    jitter = 0.18

    # per-model layers
    for model in MODEL_ORDER:
        runs = np.asarray(steps_by_model[model], dtype=float)  # (n_runs, 3)
        n_runs = runs.shape[0]
        color  = COLORS[model]
        marker = MARKERS[model]

        # per-run points (slight jitter) + optional paired lines
        for i in range(n_runs):
            xs = ENV_SIZES + rng.normal(0.0, jitter, size=ENV_SIZES.shape)
            if show_paired_lines:
                ax.plot(xs, runs[i], linewidth=0.7, alpha=0.22, color=color, zorder=1)
            ax.scatter(xs, runs[i], s=22, alpha=0.78, color=color, marker=marker, zorder=2)

        # median line
        med = np.median(runs, axis=0)
        ax.plot(ENV_SIZES, med, marker=marker, markersize=6.5,
                linewidth=2.4, label=model, color=color, zorder=3)

        # CI band (median)
        if show_ci_bands:
            lows, highs = [], []
            for j in range(runs.shape[1]):
                lo, hi = _median_ci(runs[:, j], n_resamples=3500)
                lows.append(lo); highs.append(hi)
            lows, highs = np.asarray(lows), np.asarray(highs)
            ax.fill_between(ENV_SIZES, lows, highs, alpha=0.12, linewidth=0, color=color, zorder=0)

        # annotate median values (small)
        for x, m in zip(ENV_SIZES, med):
            ax.annotate(f"{int(round(m))}", (x, m),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=9)

    # cosmetics
    ax.set_title("Total steps until full coverage", pad=10)
    ax.set_xlabel("Environment Size (rooms)")
    ax.set_ylabel("Steps to Full Coverage (lower is better)")
    ax.set_xticks(ENV_SIZES, ENV_LABELS)
    ax.margins(x=0.02)
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="#888888")
    leg.get_frame().set_alpha(0.9)

    _annotate_panel(ax, panel_label)
    fig.tight_layout()
    _export(fig, "steps_until_full_coverage")
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    set_paper_theme()

    # ---- Figure A: % explored while in budget (grouped bars)
    # Tip: set y-values in `coverage_pct[...]` to your real numbers.
    # Print-friendly: bars have both color and hatch.
    axA = plot_percentage_bars(coverage_pct, panel_label="A")

    # ---- Figure B: steps until full coverage (per-run + medians)
    # Replace `coverage_steps_by_model[...]` with your run matrices.
    axB = plot_steps_scatter_lines(
        coverage_steps_by_model,
        panel_label="B",
        show_paired_lines=True,
        show_ci_bands=True
    )

    # Notes:
    # * Set SAVE_FIGS=True to write PNG, PDF, and SVG into ./figs
    # * If you want LaTeX fonts, set USE_TEX=True (requires a LaTeX install).
    # * Hatches keep models distinct in grayscale printing.
    # * Colors/markers/hatches are consistent across both figures.
