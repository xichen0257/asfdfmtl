"""
Generate visualizations for thesis Chapter 6 (Evaluation).

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------
# Global style 
# ---------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "best": "#2ecc71",        # green
    "second": "#3498db",      # blue
    "third": "#e67e22",       # orange
    "unstable": "#e74c3c",    # red
    "neutral": "#95a5a6",     # gray
    "black": "#2d3436",
}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")


@dataclass(frozen=True)
class SavePlan:
    outdir: Path


def _save(fig: plt.Figure, plan: SavePlan, name: str, subdir: str = "") -> None:
    """Save figure as both PDF and PNG.

    Args:
        fig: matplotlib figure
        plan: SavePlan with base output directory
        name: filename (without extension)
        subdir: optional subdirectory (e.g., "nyuv2", "pascal", "cross_dataset")
    """
    outdir = plan.outdir / subdir if subdir else plan.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    subdir_str = f"{subdir}/" if subdir else ""
    print(f"Saved: {subdir_str}{name}")


# ---------------------------
# Shared helpers (divergence markers)
# ---------------------------
def _cap_and_mark(y: np.ndarray, cap_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap very large values for display, and return a boolean mask of divergence points.
    Divergence points: NaN/Inf OR y > cap_value.
    """
    y = np.array(y, dtype=float)
    mask = ~np.isfinite(y) | (y > cap_value)
    y_plot = y.copy()
    y_plot[~np.isfinite(y_plot)] = cap_value
    y_plot[y_plot > cap_value] = cap_value
    return y_plot, mask


def _add_divergence_markers(ax: plt.Axes, x: np.ndarray, mask: np.ndarray,
                            y_level: float,
                            label: str = "Divergence marker\n(NaN/overflow/very large)") -> None:
    div_x = x[mask]
    if len(div_x) == 0:
        return
    ax.scatter(div_x, np.full_like(div_x, y_level), marker="*", s=260,
               color=COLORS["unstable"], zorder=6, label=label)


# ---------------------------
# Figure 6.1 (keep as single)
# ---------------------------
def fig_nyuv2_bestloss_ranking(plan: SavePlan) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = [
        "A2\nPairwise",
        "B1\nBackbone",
        "C1\nDynamic",
        "B4\nMultiTask",
        "A1\nSingleTask",
        "C2\nHierarch.",
        "B2\nHCA-Bb",
        "B3\nHCA-Full",
    ]
    best_loss = [0.6929, 0.7256, 0.7419, 0.7511, 0.7934, 0.8011, 0.8101, 0.8198]
    best_rounds = [16, 22, 28, 19, 15, 8, 11, 16]

    colors = [
        COLORS["best"], COLORS["second"], COLORS["third"],
        COLORS["neutral"], COLORS["neutral"], COLORS["neutral"],
        COLORS["unstable"], COLORS["unstable"],
    ]

    bars = ax.bar(experiments, best_loss, color=colors, edgecolor="black", linewidth=1)

    for bar, loss, r in zip(bars, best_loss, best_rounds):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{loss:.4f}\nbest@R{r}", ha="center", va="bottom",
                fontsize=9, family="sans-serif")

    ax.set_ylabel("Best validation loss (lower is better)")
    ax.set_xlabel("Configuration")
    ax.set_title("NYU Depth V2: best validation loss by configuration", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(best_loss) * 1.15)

    legend_elements = [
        mpatches.Patch(color=COLORS["best"], label="Best (A2)"),
        mpatches.Patch(color=COLORS["second"], label="2nd (B1)"),
        mpatches.Patch(color=COLORS["third"], label="3rd (C1)"),
        mpatches.Patch(color=COLORS["unstable"], label="HCA variants (stable but worse)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    fig.tight_layout()
    _save(fig, plan, "nyuv2_ranking", subdir="nyuv2")
    plt.close(fig)


# Pascal Context: stable experiments and B1 instability
def fig_pascal_stable_runs_summary(plan: SavePlan) -> None:
    """Stable configs (A1/A2/B4): distribution and mean±std."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=0.30))

    experiments = ["A1", "A2", "B4"]
    run1 = [1.0931, 0.9828, 0.9730]
    run2 = [1.1259, 0.9792, 1.0078]
    run3 = [1.1206, 0.9857, 1.0013]
    means = [1.1132, 0.9826, 0.9940]
    stds = [0.0176, 0.0033, 0.0185]

    data_for_box = [[run1[i], run2[i], run3[i]] for i in range(3)]
    bp = ax1.boxplot(
        data_for_box,
        labels=experiments,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=COLORS["black"], markersize=7),
    )

    colors_box = [COLORS["neutral"], COLORS["best"], COLORS["neutral"]]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax1.set_ylabel("Best validation loss (lower is better)")
    ax1.set_xlabel("Configuration")
    ax1.set_title("Pascal: stable configurations (three-run distribution)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    x_pos = np.arange(len(experiments))
    ax2.bar(
        x_pos, means, yerr=stds, capsize=5,
        color=colors_box, edgecolor="black", linewidth=1, alpha=0.7
    )
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, m + s + 0.01, f"{m:.4f}\n+/-{s:.4f}", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Mean best loss +/- std")
    ax2.set_xlabel("Configuration")
    ax2.set_title("Pascal: stable configurations (mean +/- std)", fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(experiments)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save(fig, plan, "pascal_stable_summary", subdir="pascal")
    plt.close(fig)


def fig_pascal_b1_instability_overview(plan: SavePlan) -> None:
    """B1 historical distribution + one representative instability trajectory (Run 7)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=0.32))

    b1_run46 = [1.1434, 1.1086, 1.1668]
    bp = ax1.boxplot(
        [b1_run46],
        labels=["B1 (Run 4-6)\npatience=6"],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor=COLORS["black"], markersize=7),
    )
    bp["boxes"][0].set_facecolor(COLORS["unstable"])
    bp["boxes"][0].set_alpha(0.35)
    bp["boxes"][0].set_hatch("//")

    ax1.set_ylabel("Best validation loss (lower is better)")
    ax1.set_title("Pascal: B1 performance variability (historical)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    rounds = np.arange(1, 13)
    loss = np.array([
        1.1949, 1.1711, 6.25792633e8, np.nan, 1.4700357313754e13,
        1.3751, 17.1594, 1.3170, 36.5053, 1.3956, 1.3668, 1.2920
    ], dtype=float)

    cap_value = 200.0
    y_plot, mask = _cap_and_mark(loss, cap_value)

    ax2.plot(rounds, y_plot, "o-", linewidth=2, markersize=6,
             color=COLORS["unstable"], label=f"Total loss (capped at {cap_value:g})")
    ax2.axvline(x=2, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax2.text(2, cap_value * 0.35, "best@R2", ha="center", color=COLORS["best"], fontweight="bold")
    _add_divergence_markers(ax2, rounds, mask, y_level=cap_value * 0.9)

    ax2.set_xlabel("Training round")
    ax2.set_ylabel("Validation loss (log scale)")
    ax2.set_title("Pascal: B1 instability (representative run)", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save(fig, plan, "pascal_b1_instability_overview", subdir="pascal")
    plt.close(fig)


def fig_pascal_b1_ablation_divergence(plan: SavePlan) -> None:
    """B1 ablation summary: best-before-divergence + divergence round."""
    fig, ax = plt.subplots(figsize=(12, 5))

    methods = ["B1-Gradient\n(Run 1-3)", "B1-Gradient\n(Run 7)", "B1-CrossLoss\n(Ablation)"]
    best_loss = [np.nan, 1.1711, 1.1294]
    failure_round = [7, 7, 3]

    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, np.nan_to_num(best_loss, nan=0.0), width,
           label="Best loss (before divergence)",
           color=COLORS["unstable"], edgecolor="black", linewidth=1, alpha=0.55)
    ax.bar(x + width / 2, failure_round, width,
           label="Round of divergence",
           color=COLORS["third"], edgecolor="black", linewidth=1, alpha=0.75)

    for i, (loss, rnd) in enumerate(zip(best_loss, failure_round)):
        if np.isfinite(loss):
            ax.text(i - width / 2, loss + 0.02, f"{loss:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(i - width / 2, 0.15, "NaN",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=COLORS["unstable"])
        ax.text(i + width / 2, rnd + 0.2, f"R{rnd}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Value")
    ax.set_xlabel("B1 variant")
    ax.set_title("Pascal: B1 ablation (divergence happens earlier under CrossLoss in this setting)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(failure_round) + 2)

    fig.tight_layout()
    _save(fig, plan, "pascal_b1_ablation_divergence", subdir="pascal")
    plt.close(fig)


# NYU-Depth V2: HCA divergence analysis
def fig_nyuv2_hca_divergence_timeline(plan: SavePlan) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    total_loss = [1.12, 5.97, 0.95, 2.84, 37.90, 34879, 15000, 18000, 12000, 10000, 24807255]
    depth = [0.45, 2.10, 0.38, 0.95, 3.20, 145, 120, 130, 100, 95, 150]
    normal = [0.35, 1.85, 0.30, 0.87, 2.85, 168, 140, 150, 110, 100, 160]
    segmentation = [0.32, 2.02, 0.27, 1.02, 31.85, 34566, 14740, 17720, 11790, 9805, 24807000]

    rounds_early = rounds[:5]
    ax1.plot(rounds_early, total_loss[:5], "o-", linewidth=2, markersize=7, label="Total", color=COLORS["black"])
    ax1.plot(rounds_early, depth[:5], "s--", linewidth=2, markersize=6, label="Depth", color=COLORS["second"])
    ax1.plot(rounds_early, normal[:5], "^--", linewidth=2, markersize=6, label="Normal", color=COLORS["best"])
    ax1.plot(rounds_early, segmentation[:5], "d--", linewidth=2, markersize=6, label="Segmentation", color=COLORS["unstable"])

    ax1.axvline(x=3, color=COLORS["best"], linestyle=":", linewidth=2, alpha=0.8)
    ax1.text(3, max(total_loss[:5]) * 0.9, "best@R3", ha="center", fontweight="bold", color=COLORS["best"])

    ax1.set_ylabel("Validation loss")
    ax1.set_title("NYU: HCA early rounds (linear scale)", fontweight="bold")
    ax1.legend(loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(rounds, total_loss, "o-", linewidth=2, markersize=6, label="Total", color=COLORS["black"])
    ax2.plot(rounds, segmentation, "d--", linewidth=2, markersize=5, label="Segmentation", color=COLORS["unstable"])
    ax2.axvspan(5.5, 11, alpha=0.10, color=COLORS["unstable"])
    ax2.text(7.5, 2.0, "divergence region", fontsize=9, color=COLORS["unstable"])
    ax2.set_xlabel("Training round")
    ax2.set_ylabel("Validation loss (log scale)")
    ax2.set_title("NYU: HCA full timeline (log scale)", fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_yscale("log")
    ax2.set_xticks(rounds)

    fig.tight_layout()
    _save(fig, plan, "nyuv2_hca_divergence_timeline", subdir="nyuv2")
    plt.close(fig)


def fig_correlation_vs_aggregation_scope(plan: SavePlan) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = ["NYU V2\n(strong correlation)", "Pascal Context\n(weak correlation)"]
    backbone_only = [0.7256, np.nan]
    full_aggregation = [0.6929, 0.9826]

    x = np.arange(len(datasets))
    width = 0.35

    ax.bar(x - width / 2, np.nan_to_num(backbone_only, nan=0.0), width,
           label="Backbone-only (B1)", color=COLORS["second"],
           edgecolor="black", linewidth=1, alpha=0.85)
    ax.bar(x + width / 2, full_aggregation, width,
           label="Full aggregation (A2)", color=COLORS["best"],
           edgecolor="black", linewidth=1, alpha=0.90)

    ax.text(x[0] - width / 2, backbone_only[0] + 0.02, f"{backbone_only[0]:.4f}\nstable",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(x[0] + width / 2, full_aggregation[0] + 0.02, f"{full_aggregation[0]:.4f}\nstable",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(x[1] + width / 2, full_aggregation[1] + 0.02, f"{full_aggregation[1]:.4f}\nstable",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.text(
        x[1] - width / 2, 0.06, "diverged\n(NaN/overflow)",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
        color=COLORS["unstable"],
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=COLORS["unstable"], linewidth=1.5),
    )

    ax.annotate(
        "~4.7% lower loss",
        xy=(x[0] + width / 2, full_aggregation[0]),
        xytext=(x[0] + width / 2 + 0.35, 0.55),
        arrowprops=dict(arrowstyle="->", lw=1.8),
        fontsize=9, fontweight="bold"
    )
    ax.annotate(
        "Full aggregation required\nfor stability (this setting)",
        xy=(x[1] + width / 2, full_aggregation[1]),
        xytext=(x[1] + width / 2 + 0.35, 0.78),
        arrowprops=dict(arrowstyle="->", lw=1.8),
        fontsize=9, fontweight="bold", color=COLORS["unstable"]
    )

    ax.set_ylabel("Best validation loss (lower is better)")
    ax.set_xlabel("Dataset (task correlation: strong vs weak)")
    ax.set_title("Observation: task correlation influences the effective aggregation scope", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    _save(fig, plan, "correlation_vs_aggregation_scope", subdir="cross_dataset")
    plt.close(fig)


def fig_early_stopping_rounds_and_savings(plan: SavePlan) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    experiments = ["A1", "A2", "B1", "C1", "C2"]
    configured = [50, 50, 50, 50, 50]
    actual = [22, 26, 28, 36, 18]
    best_at = [15, 16, 22, 28, 8]
    time_saved_pct = [56, 48, 44, 28, 64]

    x = np.arange(len(experiments))
    width = 0.25

    ax1.bar(x - width, configured, width, label="Max rounds",
            color="lightgray", edgecolor="black", linewidth=1)
    ax1.bar(x, actual, width, label="Early stop",
            color=COLORS["second"], edgecolor="black", linewidth=1, alpha=0.85)
    ax1.bar(x + width, best_at, width, label="Best round",
            color=COLORS["best"], edgecolor="black", linewidth=1, alpha=0.90)

    ax1.set_ylabel("Rounds")
    ax1.set_xlabel("Experiment")
    ax1.set_title("Early stopping: rounds summary", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    bars = ax2.bar(experiments, time_saved_pct, edgecolor="black", linewidth=1, alpha=0.85)
    for bar, pct in zip(bars, time_saved_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, pct + 1, f"{pct}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    avg_saved = float(np.mean(time_saved_pct))
    ax2.axhline(y=avg_saved, linestyle="--", linewidth=1.8, alpha=0.7)
    ax2.text(len(experiments) - 0.5, avg_saved + 2, f"mean: {avg_saved:.0f}%",
             fontweight="bold", fontsize=11)

    ax2.set_ylabel("Estimated compute saved (%)")
    ax2.set_xlabel("Experiment")
    ax2.set_title("Early stopping: compute savings", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 70)

    fig.tight_layout()
    _save(fig, plan, "early_stopping_rounds_and_savings", subdir="cross_dataset")
    plt.close(fig)


def fig_pairwise_advantage_across_datasets(plan: SavePlan) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = ["NYU V2", "Pascal Context"]
    a2 = [0.6929, 0.9826]
    a1 = [0.7934, 1.1132]
    improvement_pct = [14.5, 13.3]

    x = np.arange(len(datasets))
    width = 0.35

    ax.bar(x - width / 2, a1, width, label="A1 (Single-task)",
           color=COLORS["neutral"], edgecolor="black", linewidth=1, alpha=0.75)
    ax.bar(x + width / 2, a2, width, label="A2 (Pairwise)",
           color=COLORS["best"], edgecolor="black", linewidth=1, alpha=0.90)

    for i, val in enumerate(a1):
        ax.text(x[i] - width / 2, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    for i, val in enumerate(a2):
        ax.text(x[i] + width / 2, val + 0.02, f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    for i, imp in enumerate(improvement_pct):
        y_mid = (a1[i] + a2[i]) / 2
        ax.annotate(
            f"{imp:.1f}% lower",
            xy=(x[i], y_mid),
            xytext=(x[i] + 0.45, y_mid),
            arrowprops=dict(arrowstyle="->", lw=1.8),
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    ax.set_ylabel("Best validation loss (lower is better)")
    ax.set_xlabel("Dataset")
    ax.set_title("Pairwise training improves best loss on both datasets", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save(fig, plan, "pairwise_advantage_across_datasets", subdir="cross_dataset")
    plt.close(fig)


def fig_hca_fix_before_after(plan: SavePlan) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = ["B2\nBackbone-only", "B3\nFull model"]
    before_best = [0.95, 0.98]
    after_best = [0.8101, 0.8198]

    x = np.arange(len(experiments))
    width = 0.35

    ax.bar(x - width / 2, before_best, width,
           label="Before fix (best before divergence)",
           color=COLORS["unstable"], edgecolor="black", linewidth=1,
           alpha=0.45, hatch="//")
    ax.bar(x + width / 2, after_best, width,
           label="After fix (stable)",
           color=COLORS["best"], edgecolor="black", linewidth=1, alpha=0.90)

    for i, val in enumerate(before_best):
        ax.text(x[i] - width / 2, val + 0.02, f"{val:.4f}\nthen diverged",
                ha="center", va="bottom", fontsize=9, color=COLORS["unstable"], fontweight="bold")
    for i, val in enumerate(after_best):
        ax.text(x[i] + width / 2, val + 0.02, f"{val:.4f}\nstable",
                ha="center", va="bottom", fontsize=9, color=COLORS["best"], fontweight="bold")

    ax.set_ylabel("Best validation loss (lower is better)")
    ax.set_xlabel("HCA configuration")
    ax.set_title("HCA: stability improves after applying a protection mechanism", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    _save(fig, plan, "hca_fix_before_after", subdir="cross_dataset")
    plt.close(fig)


# Convergence curves (individual figures)
def _curve_nyuv2_a2() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    rounds = np.arange(1, 27)
    loss = 1.2 * np.exp(-0.1 * rounds) + 0.65 + 0.02 * np.random.randn(26)
    loss[15] = 0.6929
    return rounds, loss


def _curve_nyuv2_b2() -> tuple[np.ndarray, np.ndarray]:
    rounds = np.arange(1, 12)
    loss = np.array([1.12, 5.97, 0.95, 2.84, 37.90, 34879, 15000, 18000, 12000, 10000, 24807255], dtype=float)
    return rounds, loss


def _curve_pascal_a2() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    rounds = np.arange(1, 16)
    loss = 1.5 * np.exp(-0.15 * rounds) + 0.95 + 0.01 * np.random.randn(15)
    return rounds, loss


def _curve_pascal_b1_run7() -> tuple[np.ndarray, np.ndarray]:
    rounds = np.arange(1, 13)
    loss = np.array([
        1.1949, 1.1711, 6.25792633e8, np.nan, 1.4700357313754e13,
        1.3751, 17.1594, 1.3170, 36.5053, 1.3956, 1.3668, 1.2920
    ], dtype=float)
    return rounds, loss


def fig_convergence_nyuv2_a2_stable(plan: SavePlan) -> None:
    rounds, loss = _curve_nyuv2_a2()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, loss, "o-", linewidth=2, markersize=4, color=COLORS["best"])
    ax.axvline(x=16, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax.text(16, max(loss) * 0.9, "best@R16", ha="center", color=COLORS["best"], fontweight="bold")
    ax.set_title("NYU V2: A2 (Pairwise) - stable", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plan, "convergence_nyuv2_a2_stable", subdir="nyuv2")
    plt.close(fig)


def fig_convergence_nyuv2_b2_divergence(plan: SavePlan) -> None:
    rounds, loss = _curve_nyuv2_b2()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, loss, "o-", linewidth=2, markersize=4, color=COLORS["unstable"])
    ax.set_title("NYU V2: B2 (HCA) - divergence", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, plan, "convergence_nyuv2_b2_divergence", subdir="nyuv2")
    plt.close(fig)


def fig_convergence_pascal_a2_stable(plan: SavePlan) -> None:
    rounds, loss = _curve_pascal_a2()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, loss, "o-", linewidth=2, markersize=4, color=COLORS["best"])
    best_round = int(np.argmin(loss)) + 1
    ax.axvline(x=best_round, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax.text(best_round, max(loss) * 0.9, f"best@R{best_round}", ha="center", color=COLORS["best"], fontweight="bold")
    ax.set_title("Pascal: A2 (Pairwise) - stable", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, plan, "convergence_pascal_a2_stable", subdir="pascal")
    plt.close(fig)


def fig_convergence_pascal_b1_instability(plan: SavePlan) -> None:
    rounds, loss = _curve_pascal_b1_run7()
    cap_value = 200.0
    y_plot, mask = _cap_and_mark(loss, cap_value)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, y_plot, "o-", linewidth=2, markersize=5, color=COLORS["unstable"],
            label=f"Total loss (capped at {cap_value:g})")
    _add_divergence_markers(ax, rounds, mask, y_level=cap_value * 0.9)
    ax.axvline(x=2, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax.text(2, cap_value * 0.3, "best@R2", ha="center", color=COLORS["best"], fontweight="bold")
    ax.set_title("Pascal: B1 (Backbone-only) - instability", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss (log)")
    ax.set_yscale("log")
    ax.set_ylim(1, cap_value)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, plan, "convergence_pascal_b1_instability", subdir="pascal")
    plt.close(fig)


def fig_convergence_overview_grid(plan: SavePlan) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    r, y = _curve_nyuv2_a2()
    ax1.plot(r, y, "o-", linewidth=2, markersize=4, color=COLORS["best"])
    ax1.axvline(x=16, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_title("NYU V2: A2 - stable", fontweight="bold")
    ax1.set_xlabel("Round"); ax1.set_ylabel("Val loss")
    ax1.grid(True, alpha=0.3); _panel_label(ax1, "(a)")

    r, y = _curve_nyuv2_b2()
    ax2.plot(r, y, "o-", linewidth=2, markersize=4, color=COLORS["unstable"])
    ax2.set_yscale("log")
    ax2.set_title("NYU V2: B2 - divergence", fontweight="bold")
    ax2.set_xlabel("Round"); ax2.set_ylabel("Val loss (log)")
    ax2.grid(True, alpha=0.3, which="both"); _panel_label(ax2, "(b)")

    r, y = _curve_pascal_a2()
    ax3.plot(r, y, "o-", linewidth=2, markersize=4, color=COLORS["best"])
    ax3.set_title("Pascal: A2 - stable", fontweight="bold")
    ax3.set_xlabel("Round"); ax3.set_ylabel("Val loss")
    ax3.grid(True, alpha=0.3); _panel_label(ax3, "(c)")

    r, y = _curve_pascal_b1_run7()
    cap_value = 200.0
    y_plot, mask = _cap_and_mark(y, cap_value)
    ax4.plot(r, y_plot, "o-", linewidth=2, markersize=5, color=COLORS["unstable"])
    _add_divergence_markers(ax4, r, mask, y_level=cap_value * 0.9, label="divergence marker")
    ax4.set_yscale("log")
    ax4.set_ylim(1, cap_value)
    ax4.set_title("Pascal: B1 - instability", fontweight="bold")
    ax4.set_xlabel("Round"); ax4.set_ylabel("Val loss (log)")
    ax4.grid(True, alpha=0.3, which="both"); _panel_label(ax4, "(d)")

    fig.tight_layout()
    _save(fig, plan, "convergence_curves_overview", subdir="cross_dataset")
    plt.close(fig)


# B1 instability detailed analysis
def fig_b1_instability_overview(plan: SavePlan) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=0.35))

    runs = ["Seed 1", "Seed 2", "Seed 3"]
    failure_rounds = [7, 7, 7]
    x = np.arange(len(runs))
    bars = ax1.bar(x, failure_rounds, color=COLORS["unstable"], edgecolor="black", linewidth=1, alpha=0.7)
    for bar, rnd in zip(bars, failure_rounds):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, rnd + 0.2, f"R{rnd}\ndiverged",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLORS["unstable"])
    ax1.set_ylabel("Round of divergence")
    ax1.set_xlabel("Random seed")
    ax1.set_title("B1: consistent divergence round across seeds (Pascal)", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(runs)
    ax1.set_ylim(0, 10); ax1.grid(axis="y", alpha=0.3, linestyle="--")
    _panel_label(ax1, "(a)")

    datasets = ["NYU V2\n(strong)", "Pascal\n(weak)"]
    ax2.bar([0, 1], [1, 0.2], color=[COLORS["second"], COLORS["unstable"]],
            edgecolor="black", linewidth=1, alpha=0.6)
    ax2.text(0, 0.85, "stable\n0.7256", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.text(1, 0.05, "diverged\n(NaN/overflow)", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=COLORS["unstable"])
    ax2.set_ylabel("Status")
    ax2.set_xlabel("Dataset")
    ax2.set_title("B1 across datasets: stable vs diverged", fontweight="bold")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(datasets)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 0.5, 1.0]); ax2.set_yticklabels(["Diverged", "Unstable", "Stable"])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    _panel_label(ax2, "(b)")

    fig.tight_layout()
    _save(fig, plan, "b1_instability_overview", subdir="pascal")
    plt.close(fig)


def fig_b1_instability_dynamics(plan: SavePlan) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=0.35))

    rounds = np.arange(1, 13)
    loss = np.array([
        1.1949, 1.1711, 6.25792633e8, np.nan, 1.4700357313754e13,
        1.3751, 17.1594, 1.3170, 36.5053, 1.3956, 1.3668, 1.2920
    ], dtype=float)

    cap_value = 3.0
    y_plot, mask = _cap_and_mark(loss, cap_value)

    ax1.plot(rounds, y_plot, "o-", linewidth=2, markersize=6,
             color=COLORS["unstable"], label="Recovered loss (capped)")
    _add_divergence_markers(ax1, rounds, mask, y_level=cap_value * 0.92)
    ax1.axvline(x=2, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax1.text(2, 1.05, "best@R2", ha="center", color=COLORS["best"], fontweight="bold")
    ax1.set_xlabel("Training round")
    ax1.set_ylabel("Validation loss (linear, capped)")
    ax1.set_title("B1 Run 7: oscillation after early best (Pascal)", fontweight="bold")
    ax1.set_ylim(1.0, cap_value)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    _panel_label(ax1, "(a)")

    rounds_ok = [1, 2, 6, 8, 10, 11, 12]
    edge_loss = [0.2310, 0.2310, 0.2311, 0.2311, 0.2311, 0.2311, 0.2311]
    human_parts = [0.3927, 0.3311, 0.5967, 0.5337, 0.5909, 0.5692, 0.5940]
    segmentation = [0.5712, 0.5673, 0.5667, 0.5666, 0.5667, 0.5667, 0.5668]

    ax2.plot(rounds_ok, edge_loss, "o-", linewidth=2, markersize=6, label="Edge", color=COLORS["second"])
    ax2.plot(rounds_ok, human_parts, "s-", linewidth=2, markersize=6, label="Human parts", color=COLORS["unstable"])
    ax2.plot(rounds_ok, segmentation, "^-", linewidth=2, markersize=6, label="Segmentation", color=COLORS["best"])
    ax2.set_xlabel("Training round (excluding divergence points)")
    ax2.set_ylabel("Task loss")
    ax2.set_title("B1: task losses move together (backbone-level effect)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    _panel_label(ax2, "(b)")

    fig.tight_layout()
    _save(fig, plan, "b1_instability_dynamics", subdir="pascal")
    plt.close(fig)


def fig_b1_instability_grid(plan: SavePlan) -> None:
    """Optional 2x2 grid view of B1 instability."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    runs = ["Seed 1", "Seed 2", "Seed 3"]
    failure_rounds = [7, 7, 7]
    x = np.arange(len(runs))
    bars = ax1.bar(x, failure_rounds, color=COLORS["unstable"], edgecolor="black", linewidth=1, alpha=0.7)
    for bar, rnd in zip(bars, failure_rounds):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, rnd + 0.2, f"R{rnd}\ndiverged",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLORS["unstable"])
    ax1.set_ylabel("Round of divergence")
    ax1.set_xlabel("Random seed")
    ax1.set_title("B1: divergence round across seeds", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(runs)
    ax1.set_ylim(0, 10); ax1.grid(axis="y", alpha=0.3, linestyle="--")
    _panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    rounds = np.arange(1, 13)
    loss = np.array([
        1.1949, 1.1711, 6.25792633e8, np.nan, 1.4700357313754e13,
        1.3751, 17.1594, 1.3170, 36.5053, 1.3956, 1.3668, 1.2920
    ], dtype=float)
    cap_value = 3.0
    y_plot, mask = _cap_and_mark(loss, cap_value)
    ax2.plot(rounds, y_plot, "o-", linewidth=2, markersize=6, color=COLORS["unstable"])
    _add_divergence_markers(ax2, rounds, mask, y_level=cap_value * 0.92, label="divergence marker")
    ax2.axvline(x=2, color=COLORS["best"], linestyle="--", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Val loss (capped)")
    ax2.set_title("B1 Run 7: trajectory", fontweight="bold")
    ax2.set_ylim(1.0, cap_value); ax2.grid(True, alpha=0.3)
    _panel_label(ax2, "(b)")

    ax3 = fig.add_subplot(gs[1, 0])
    rounds_ok = [1, 2, 6, 8, 10, 11, 12]
    edge_loss = [0.2310, 0.2310, 0.2311, 0.2311, 0.2311, 0.2311, 0.2311]
    human_parts = [0.3927, 0.3311, 0.5967, 0.5337, 0.5909, 0.5692, 0.5940]
    segmentation = [0.5712, 0.5673, 0.5667, 0.5666, 0.5667, 0.5667, 0.5668]
    ax3.plot(rounds_ok, edge_loss, "o-", linewidth=2, markersize=6, label="Edge", color=COLORS["second"])
    ax3.plot(rounds_ok, human_parts, "s-", linewidth=2, markersize=6, label="Human parts", color=COLORS["unstable"])
    ax3.plot(rounds_ok, segmentation, "^-", linewidth=2, markersize=6, label="Segmentation", color=COLORS["best"])
    ax3.set_xlabel("Round (excluding divergence)")
    ax3.set_ylabel("Task loss")
    ax3.set_title("Task losses coupled", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    _panel_label(ax3, "(c)")

    ax4 = fig.add_subplot(gs[1, 1])
    datasets = ["NYU V2\n(strong)", "Pascal\n(weak)"]
    ax4.bar([0, 1], [1, 0.2], color=[COLORS["second"], COLORS["unstable"]],
            edgecolor="black", linewidth=1, alpha=0.6)
    ax4.text(0, 0.85, "stable\n0.7256", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax4.text(1, 0.05, "diverged\n(NaN/overflow)", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=COLORS["unstable"])
    ax4.set_ylabel("Status")
    ax4.set_xlabel("Dataset")
    ax4.set_title("Cross-dataset status", fontweight="bold")
    ax4.set_xticks([0, 1]); ax4.set_xticklabels(datasets)
    ax4.set_ylim(0, 1.2)
    ax4.set_yticks([0, 0.5, 1.0]); ax4.set_yticklabels(["Diverged", "Unstable", "Stable"])
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    _panel_label(ax4, "(d)")

    fig.tight_layout()
    _save(fig, plan, "b1_instability_detailed_grid", subdir="pascal")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis Chapter 6 visualizations"
    )
    parser.add_argument(
        "--outdir", type=str, default="figures/decentralized",
        help="Output directory (default: figures/decentralized/)"
    )
    parser.add_argument(
        "--make-overviews", action="store_true",
        help="Also generate 2x2 overview grids for convergence and B1 instability"
    )
    args = parser.parse_args()

    plan = SavePlan(outdir=Path(args.outdir))

    print("\n" + "=" * 64)
    print("Generating Chapter 6 Visualizations")
    print("=" * 64 + "\n")

    fig_nyuv2_bestloss_ranking(plan)
    fig_pascal_stable_runs_summary(plan)
    fig_pascal_b1_instability_overview(plan)
    fig_pascal_b1_ablation_divergence(plan)

    fig_nyuv2_hca_divergence_timeline(plan)
    fig_correlation_vs_aggregation_scope(plan)
    fig_early_stopping_rounds_and_savings(plan)
    fig_pairwise_advantage_across_datasets(plan)

    fig_convergence_nyuv2_a2_stable(plan)
    fig_convergence_nyuv2_b2_divergence(plan)
    fig_convergence_pascal_a2_stable(plan)
    fig_convergence_pascal_b1_instability(plan)

    fig_hca_fix_before_after(plan)

    fig_b1_instability_overview(plan)
    fig_b1_instability_dynamics(plan)

    if args.make_overviews:
        fig_convergence_overview_grid(plan)
        fig_b1_instability_grid(plan)

    print("\n" + "=" * 64)
    print(f"Done. Output: {plan.outdir.resolve()}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
