
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

ROOT = Path(r"C:\Users\Admin\histopathology_project")
RESULTS = ROOT / "results"

MODELS = {
    "Robust Augmentation": (
        "report_figures_robust_aug",
        "centre2_predictions_robust_aug.csv",
        "robust_aug",
    ),
    "DenseNet-121": (
        "report_figures_densenet121",
        "centre2_predictions_densenet121.csv",
        "densenet121",
    ),
}

for model_name, (folder, csv_name, suffix) in MODELS.items():
    output_dir = RESULTS / folder
    csv_path = output_dir / csv_name

    if not csv_path.exists():
        print(f"SKIPPED: prediction file not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Support the column names used by the existing evaluation scripts.
    label_col = next(
        (c for c in ["true_label", "label", "tumor"] if c in df.columns),
        None,
    )
    prob_col = next(
        (c for c in ["tumour_probability", "probability_tumor",
                     "tumor_probability", "probability"] if c in df.columns),
        None,
    )

    if label_col is None or prob_col is None:
        print(f"SKIPPED: unexpected CSV columns in {csv_path}")
        print("Available columns:", list(df.columns))
        continue

    y_true = df[label_col].astype(int)
    y_prob = df[prob_col].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Non-Tumour", "Tumour"])
    ax.set_yticks([0, 1], labels=["Non-Tumour", "Tumour"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{model_name} — Unseen Centre 2")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")

    fig.tight_layout()
    cm_path = output_dir / f"fig_centre2_{suffix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {score:.4f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC — Unseen Centre 2")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    roc_path = output_dir / f"fig_centre2_{suffix}_roc_curve.png"
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{model_name}: figures saved")
    print(cm_path)
    print(roc_path)

print("\nFigure generation finished.")
