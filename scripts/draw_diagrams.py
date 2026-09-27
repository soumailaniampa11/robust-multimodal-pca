"""Draw the architecture and agent-workflow diagrams used in the README."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcafusion import config  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
LINE = "#8a8984"
SURFACE = "#ffffff"
NEUTRAL = "#f1f0ec"
MODALITY = {"clinical": "#2a78d6", "radiology": "#eb6834", "imaging": "#1baf7a"}
TINT = {"clinical": "#e3eefb", "radiology": "#fde9e0", "imaging": "#dff4ec"}


def box(ax, x, y, w, h, title, sub=None, face=NEUTRAL, edge=LINE, title_color=INK, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.3,
                                linestyle="--" if dashed else "-"))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=9.5,
                color=title_color, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.8, color=INK_2,
                linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=9.5, color=title_color,
                fontweight="bold")


def arrow(ax, p, q, color=LINE, style="-|>", rad=0.0, lw=1.3, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2))


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    subs = {
        "clinical": "age, PSA, PSA density,\nreported volume",
        "radiology": "max PI-RADS, number of\nsuspicious lesions",
        "imaging": "gland, PZ and TZ volumes\nfrom AI segmentations",
    }
    ys = {"clinical": 3.3, "radiology": 1.95, "imaging": 0.6}
    for m, y in ys.items():
        c = MODALITY[m]
        box(ax, 0.2, y, 2.3, 0.95, m.capitalize(), subs[m], face=TINT[m], edge=c, title_color=c)
        box(ax, 3.0, y + 0.12, 1.55, 0.7, "values + mask", None, face=SURFACE)
        box(ax, 5.0, y + 0.12, 1.25, 0.7, f"encoder", None, face=SURFACE, edge=c, title_color=c)
        arrow(ax, (2.5, y + 0.47), (3.0, y + 0.47))
        arrow(ax, (4.55, y + 0.47), (5.0, y + 0.47))
        arrow(ax, (6.25, y + 0.47), (6.9, 2.42), color=c)

    box(ax, 6.9, 1.62, 1.75, 1.6, "Attention\nfusion", None, face=NEUTRAL)
    ax.text(7.775, 1.38, "softmax over available\nmodalities only", ha="center", va="top", fontsize=7.6,
            color=INK_2)
    box(ax, 9.05, 1.97, 1.7, 0.9, "Classifier", "logit, calibrated\nby temperature", face=SURFACE)
    arrow(ax, (8.65, 2.42), (9.05, 2.42))

    ax.text(0.2, 4.45, "Input modalities", fontsize=8.5, color=INK_2, va="top")
    ax.text(3.0, 4.45, "Observed-feature mask", fontsize=8.5, color=INK_2, va="top")
    ax.text(5.0, 4.45, "Per-modality MLP", fontsize=8.5, color=INK_2, va="top")
    ax.text(9.05, 3.25, "5-member deep ensemble;\nspread = epistemic uncertainty", fontsize=7.6,
            color=INK_2, va="bottom")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def agent_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.9), facecolor=SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    steps = [
        ("get_case", "available and\nmissing modalities"),
        ("predict_risk", "probability, conformal\nset, disagreement"),
        ("modality_ablation", "effect of withholding\neach modality"),
        ("find_similar_cases", "outcomes of nearest\ntraining cases"),
        ("retrieve_evidence", "guideline snippets\nwith identifiers"),
    ]
    w, h, gap, y = 1.82, 1.05, 0.33, 2.45
    for i, (t, s) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, y, w, h, t, s, face=SURFACE)
        if i:
            arrow(ax, (x - gap, y + h / 2), (x, y + h / 2))

    box(ax, 1.2, 0.35, 2.6, 1.15, "Safety rules (code)", "LOW only if the conformal set is\nnegative, no modality is missing\nand the threshold agrees",
        face=NEUTRAL)
    box(ax, 4.3, 0.35, 2.6, 1.15, "Citation check", "every cited identifier must have\nbeen retrieved in this run",
        face=NEUTRAL)
    box(ax, 7.4, 0.35, 3.1, 1.15, "Report and audit trace",
        "triage flag LOW / ELEVATED / UNCERTAIN,\nevidence, limitations; JSON log of\nevery tool call", face=TINT["clinical"],
        edge=MODALITY["clinical"], title_color=MODALITY["clinical"])
    arrow(ax, (0.2 + 1 * (w + gap) + w / 2, y), (2.5, 1.5), rad=0.0)
    arrow(ax, (0.2 + 4 * (w + gap) + w / 2, y), (5.6, 1.5), rad=0.0)
    arrow(ax, (3.8, 0.92), (4.3, 0.92))
    arrow(ax, (6.9, 0.92), (7.4, 0.92))
    ax.text(0.2, 3.8, "Tool calls (identical in the deterministic and the LLM-orchestrated mode)",
            fontsize=8.5, color=INK_2, va="top")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    architecture(config.FIGURES_DIR / "architecture.png")
    agent_workflow(config.FIGURES_DIR / "agent_workflow.png")
    print("saved diagrams to", config.FIGURES_DIR)
