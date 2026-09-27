"""Figure helpers (matplotlib). Colors follow a fixed categorical order per model."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Color follows the entity (model), never its rank.
MODEL_COLORS = {
    "Fusion ensemble (ours)": "#2a78d6",
    "Early-fusion GBDT": "#eb6834",
    "Early-fusion LR (mean imputation)": "#1baf7a",
    "Clinical LR": "#4a3aa7",
    "PI-RADS (radiologist)": "#8a8984",
}


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _fig(ncols: int = 1, width: float = 5.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, 4.2), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def roc_panels(panels: list[tuple[str, np.ndarray, dict[str, np.ndarray], dict[str, dict]]], path) -> None:
    """panels: list of (title, y, {model: p}, {model: summary})."""
    fig, axes = _fig(len(panels))
    for ax, (title, y, preds, summ) in zip(axes, panels):
        for name, p in preds.items():
            fpr, tpr, _ = roc_curve(y, p)
            s = summ[name]
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                    label=f"{name}  {s['auroc']:.3f} [{s['auroc_lo']:.3f}–{s['auroc_hi']:.3f}]")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
        _style(ax, title, "1 − specificity", "Sensitivity")
        ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK, title="AUROC [95% CI]",
                  title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability(y: np.ndarray, preds: dict[str, np.ndarray], eces: dict[str, float], path, n_bins: int = 10) -> None:
    fig, (ax,) = _fig()
    edges = np.linspace(0, 1, n_bins + 1)
    for name, p in preds.items():
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() >= 10:
                xs.append(p[sel].mean())
                ys.append(y[sel].mean())
        ax.plot(xs, ys, color=MODEL_COLORS[name], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name} (ECE {eces[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", label="Perfect calibration")
    _style(ax, "Calibration (out-of-fold)", "Predicted probability of csPCa", "Observed frequency")
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def missing_modality_bars(table, path) -> None:
    """table: DataFrame index=scenario, columns=model names, values=AUROC."""
    fig, (ax,) = _fig(width=7.5)
    scenarios = list(table.index)
    models = list(table.columns)
    w = 0.8 / len(models)
    x = np.arange(len(scenarios))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, table[m].to_numpy(), width=w * 0.92,
               color=MODEL_COLORS[m], label=m, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8, rotation=0)
    ax.set_ylim(0.5, 1.0)
    _style(ax, "Robustness to missing modalities at test time", "Modalities available", "AUROC")
    ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=INK, ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def decision_curve(thresholds: np.ndarray, curves: dict[str, np.ndarray], path, title: str) -> None:
    fig, (ax,) = _fig()
    extra = {"Biopsy all": "#c3c2b7", "Biopsy none": INK_2}
    for name, nb in curves.items():
        color = MODEL_COLORS.get(name, extra.get(name, INK_2))
        ls = ":" if name in extra else "-"
        ax.plot(thresholds, nb, color=color, lw=2, ls=ls, label=name)
    lo = min(-0.05, float(np.nanmin(np.concatenate(list(curves.values())))))
    ax.set_ylim(max(lo, -0.1), None)
    _style(ax, title, "Threshold probability for biopsy", "Net benefit")
    ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def conformal_curve(alphas: np.ndarray, cov_pos: np.ndarray, cov_neg: np.ndarray, ambiguous: np.ndarray, path) -> None:
    fig, axes = _fig(2)
    ax = axes[0]
    ax.plot(1 - alphas, cov_pos, color="#2a78d6", lw=2, marker="o", ms=5, label="csPCa cases")
    ax.plot(1 - alphas, cov_neg, color="#eb6834", lw=2, marker="o", ms=5, label="non-csPCa cases")
    ax.plot(1 - alphas, 1 - alphas, color=INK_2, lw=1, ls="--", label="Target (1 − α)")
    _style(ax, "Class-conditional coverage", "Target coverage 1 − α", "Empirical coverage")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    ax = axes[1]
    ax.plot(1 - alphas, ambiguous, color="#2a78d6", lw=2, marker="o", ms=5)
    _style(ax, "Cases deferred to clinician", "Target coverage 1 − α", "Fraction with ambiguous set {0, 1}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def attention_by_group(groups: list[str], weights: np.ndarray, modality_names: list[str], path) -> None:
    """weights: (n_groups, M) mean attention. Stacked horizontal bars."""
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (ax,) = _fig(width=6.5)
    left = np.zeros(len(groups))
    for j, m in enumerate(modality_names):
        ax.barh(groups, weights[:, j], left=left, color=colors[j], label=m, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(weights[:, j]):
            if v > 0.08:
                ax.text(left[i] + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
        left += weights[:, j]
    ax.set_xlim(0, 1)
    _style(ax, "Modality attention weights by PI-RADS group", "Mean attention weight", "")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
