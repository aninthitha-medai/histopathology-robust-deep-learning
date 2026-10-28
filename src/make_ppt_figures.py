import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Admin\histopathology_project\data\camelyon17_v1.0"
OUTPUT_DIR = r"C:\Users\Admin\histopathology_project\results\report_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = pd.read_csv(
    os.path.join(BASE_DIR, "metadata.csv")
)

# ============================================================
# FIGURE 1 — CENTRE-BASED DATASET SPLIT
# ============================================================

train = metadata[
    (metadata["split"] == 0) &
    (metadata["center"].isin([0, 3, 4]))
]

val = metadata[
    (metadata["center"] == 1) |
    (
        (metadata["split"] == 1) &
        (metadata["center"].isin([0, 3, 4]))
    )
]

test = metadata[
    metadata["center"] == 2
]

print("Dataset split:")
print("Training:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")

ax.text(
    0.5, 0.92,
    "CAMELYON17-WILDS Cross-Centre Experimental Design",
    ha="center",
    fontsize=18,
    fontweight="bold"
)

ax.text(
    0.5, 0.76,
    "CAMELYON17-WILDS\n455,954 histopathology patches",
    ha="center",
    va="center",
    fontsize=15,
    bbox=dict(boxstyle="round,pad=0.6")
)

ax.text(
    0.20, 0.48,
    f"TRAINING\n\nCentres 0, 3 and 4\n\n{len(train):,} patches",
    ha="center",
    va="center",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.7")
)

ax.text(
    0.50, 0.48,
    f"VALIDATION\n\nCentre 1 + validation split\n\n{len(val):,} patches",
    ha="center",
    va="center",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.7")
)

ax.text(
    0.80, 0.48,
    f"UNSEEN TEST\n\nCentre 2\n\n{len(test):,} patches",
    ha="center",
    va="center",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.7")
)

ax.text(
    0.80, 0.22,
    "Centre 2 was not used during training",
    ha="center",
    fontsize=12,
    fontweight="bold"
)

output = os.path.join(
    OUTPUT_DIR,
    "fig_3_3_centre_based_dataset_split.png"
)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Created:", output)


# ============================================================
# FIGURE 2 — OVERALL EXPERIMENTAL WORKFLOW
# ============================================================

fig, ax = plt.subplots(figsize=(12, 9))
ax.axis("off")

steps = [
    "CAMELYON17-WILDS\n455,954 patches",
    "96 × 96 RGB\nHistopathology patches",
    "Centre-based\ntrain / validation / test split",
    "ImageNet-pretrained\nResNet-50",
    "Training + validation",
    "Completely unseen\nCentre 2 testing",
    "Accuracy • Precision • Recall\nF1-score • ROC-AUC"
]

y_positions = np.linspace(0.90, 0.10, len(steps))

for i, (step, y) in enumerate(zip(steps, y_positions)):

    ax.text(
        0.5,
        y,
        step,
        ha="center",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.6")
    )

    if i < len(steps) - 1:
        ax.annotate(
            "",
            xy=(0.5, y_positions[i+1] + 0.045),
            xytext=(0.5, y - 0.045),
            arrowprops=dict(arrowstyle="->", lw=2)
        )

ax.set_title(
    "Overall Experimental Workflow",
    fontsize=18,
    fontweight="bold",
    pad=15
)

output = os.path.join(
    OUTPUT_DIR,
    "fig_3_1_overall_experimental_workflow.png"
)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Created:", output)


# ============================================================
# FIGURE 3 — SEEN VS UNSEEN PERFORMANCE
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

seen = [
    99.34,
    99.45,
    99.21,
    99.33,
    99.96
]

unseen = [
    78.54,
    98.04,
    58.24,
    73.07,
    90.44
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar(x - width/2, seen, width, label="Represented Centres")
ax.bar(x + width/2, unseen, width, label="Unseen Centre 2")

ax.set_ylabel("Performance (%)")
ax.set_title(
    "Baseline Performance: Represented vs Unseen Centre",
    fontsize=16,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 110)

ax.legend()

for i, value in enumerate(seen):
    ax.text(
        i - width/2,
        value + 2,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )

for i, value in enumerate(unseen):
    ax.text(
        i + width/2,
        value + 2,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

output = os.path.join(
    OUTPUT_DIR,
    "fig_4_2_seen_vs_unseen_performance.png"
)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Created:", output)


# ============================================================
# FIGURE 4 — BASELINE TRAINING / VALIDATION ROC-AUC
# ============================================================

epochs = [1, 2, 3, 4, 5]

val_auc = [
    0.9880,
    0.9881,
    0.9835,
    0.9882,
    0.9724
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    epochs,
    val_auc,
    marker="o",
    linewidth=2
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation ROC-AUC")
ax.set_title(
    "ResNet-50 Validation ROC-AUC",
    fontsize=16,
    fontweight="bold"
)

ax.set_xticks(epochs)
ax.set_ylim(0.95, 1.00)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output = os.path.join(
    OUTPUT_DIR,
    "fig_4_1_validation_roc_auc.png"
)

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Created:", output)


# ============================================================
# FIGURE 5 — UNSEEN CENTRE-2 CONFUSION MATRIX
# ============================================================
#
# IMPORTANT:
# The exact TP/TN/FP/FN values are not available from the
# summary metrics alone. Therefore we DO NOT invent them.
#
# This figure is intentionally not generated here.
# We will generate it directly from your prediction file
# when available.
# ============================================================


# ============================================================
# FIGURE 6 — UNSEEN CENTRE-2 ROC CURVE
# ============================================================
#
# Exact ROC curve requires the individual Centre-2 prediction
# probabilities. ROC-AUC=0.9044 alone is insufficient to
# reconstruct the curve.
#
# We therefore DO NOT fabricate this figure.
# ============================================================


print()
print("==============================================")
print("ALL AVAILABLE FIGURES CREATED")
print("==============================================")
print()
print("Open this folder:")
print(OUTPUT_DIR)