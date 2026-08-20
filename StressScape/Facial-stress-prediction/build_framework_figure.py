"""Redraws Figure 4 (the proposed framework flowchart).

The original 'unnamed.png' depicted the pre-revision pipeline and contained
three inconsistencies a reviewer would flag:
  1. "Error rate Calculated = total_words/ bksp" - the backspace/error-rate
     feature does not exist in the public CMU dataset (and the ratio was
     inverted), so it no longer matches the methodology.
  2. "Random Forest Accuracy = 83.5%" - matched neither the old reported
     84.56% nor the real trained 73.65%.
  3. "Tabs open count" - a modality never described anywhere in the paper.

This regenerates it from the corrected methodology and the real metrics.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = (Path(__file__).resolve().parent.parent.parent
       / "overleaf_submission" / "all models compariosn" / "unnamed.png")

INK = "#111111"
EDGE = "#333333"
FACIAL = "#EAF2FB"
KEY = "#FDF0E4"
MODEL = "#F2F2F2"
PICK = "#E8F5E9"
DEPLOY = "#F3E8FB"

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=9.5, bold_first=True, radius=0.12):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=fc, edgecolor=EDGE, linewidth=1.3))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x, y + h / 2 - 0.28, lines[0], ha="center", va="center",
                fontsize=fs + 0.7, fontweight="bold", color=INK)
        ax.text(x, y - 0.16, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=fs - 0.6, color=INK, linespacing=1.45)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold_first else "normal",
                color=INK, linespacing=1.45)


def arrow(x1, y1, x2, y2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.4,
        color=EDGE, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2))


# ------------------------------------------------ lane labels -------------
ax.text(0.15, 6.55, "FACIAL\nMODALITY", ha="left", va="center", fontsize=9,
        fontweight="bold", color="#1F4E79", linespacing=1.3)
ax.text(0.15, 3.35, "KEYSTROKE\nMODALITY", ha="left", va="center", fontsize=9,
        fontweight="bold", color="#B25F00", linespacing=1.3)

# ------------------------------------------------ facial lane -------------
box(2.35, 6.55, 2.5, 1.5,
    "Dataset 1: Facial Images\n"
    "FER2013 benchmark +\n1,400 author-consented\n"
    "53,571 images (7 classes)", FACIAL)

box(5.5, 6.55, 2.3, 1.5,
    "Preprocessing\n"
    "1. Face detection\n2. Augmentation\n"
    "3. Binary stress mapping", FACIAL)

box(8.35, 6.55, 1.85, 1.5,
    "Transfer Learning\n"
    "MobileNetV2\nEfficientNetB0\nResNet50V2", MODEL)

# ------------------------------------------------ keystroke lane ----------
box(2.35, 3.35, 2.5, 1.5,
    "Dataset 2: Keystroke\n"
    "CMU Benchmark\n51 subjects x 400 reps\n"
    "20,400 typing samples", KEY)

box(5.5, 3.35, 2.3, 1.7,
    "Feature Engineering\n"
    "hold_mean, hold_std,\nflight_mean, flight_std,\ntotal_duration\n"
    "+ within-subject label", KEY)

box(8.35, 3.35, 1.85, 1.5,
    "Classifiers\n"
    "Random Forest\n(100 trees)\n"
    "Decision Tree\n(baseline)", MODEL)

# ------------------------------------------------ model selection ---------
box(11.35, 4.95, 2.5, 2.0,
    "Model Selection\n"
    "ResNet50V2\nAccuracy = 88.64%\n\n"
    "Random Forest\nAccuracy = 73.65%\n(78.78% $\\pm$ 2.52% CV)", PICK)

# ------------------------------------------------ fusion + deploy ---------
box(11.35, 2.05, 2.5, 1.35,
    "Weighted Late Fusion\n"
    "$P$ = 0.6$\\cdot P_{face}$ + 0.4$\\cdot P_{key}$\n"
    "10-frame temporal smoothing", DEPLOY)

box(11.35, 0.55, 2.5, 1.0,
    "Live Stress Monitor\n"
    "Local inference, 25-30 FPS\n(webcam + keystroke timing)", DEPLOY)

# ------------------------------------------------ arrows ------------------
arrow(3.62, 6.55, 4.33, 6.55)
arrow(6.68, 6.55, 7.41, 6.55)
arrow(3.62, 3.35, 4.33, 3.35)
arrow(6.68, 3.35, 7.41, 3.35)

arrow(9.30, 6.55, 10.10, 5.60)
arrow(9.30, 3.35, 10.10, 4.30)

arrow(11.35, 3.93, 11.35, 2.75)
arrow(11.35, 1.36, 11.35, 1.07)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
