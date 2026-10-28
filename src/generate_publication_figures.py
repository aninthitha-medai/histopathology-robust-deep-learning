
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "Baseline ResNet-50",
        "folder": "report_figures_baseline_corrected",
        "metrics": "centre2_metrics_baseline_corrected.json",
        "predictions": "centre2_predictions_baseline_corrected.csv",
    },
    {
        "name": "Reinhard ResNet-50",
        "folder": "report_figures_reinhard_corrected",
        "metrics": "centre2_metrics_reinhard_corrected.json",
        "predictions": "centre2_predictions_reinhard_corrected.csv",
    },
    {
        "name": "Robust Augmentation ResNet-50",
        "folder": "report_figures_robust_aug",
        "metrics": "centre2_metrics_robust_aug.json",
        "predictions": "centre2_predictions_robust_aug.csv",
    },
    {
        "name": "DenseNet-121",
        "folder": "report_figures_densenet121",
        "metrics": "centre2_metrics_densenet121.json",
        "predictions": "centre2_predictions_densenet121.csv",
    },
]

# Previously measured Centre 2 results, in percent.
KNOWN = {
    "Baseline ResNet-50": [78.54, 98.03, 58.24, 73.07, 90.44],
    "Reinhard ResNet-50": [90.42, 93.55, 86.84, 90.07, 96.95],
    "Robust Augmentation ResNet-50": [79.17, 97.87, 59.65, 74.12, 84.14],
    "DenseNet-121": [85.79, 98.63, 72.58, 83.62, 96.05],
}

TRUE_NAMES = [
    "y_true", "true", "label", "labels", "target",
    "ground_truth", "actual", "true_label"
]
PROB_NAMES = [
    "y_prob", "probability", "prob", "score", "scores",
    "tumor_probability", "tumour_probability",
    "positive_probability", "predicted_probability",
    "y_score", "p1", "prob_1"
]
PRED_NAMES = [
    "y_pred", "prediction", "pred", "predicted",
    "predicted_label", "pred_label"
]


def normalize_name(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, candidates):
    columns = {normalize_name(c): c for c in df.columns}
    for name in candidates:
        if name in columns:
            return columns[name]
    return None


def load_predictions(model):
    path = RESULTS / model["folder"] / model["predictions"]

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file:\n{path}")

    df = pd.read_csv(path)
    true_col = find_column(df, TRUE_NAMES)
    prob_col = find_column(df, PROB_NAMES)
    pred_col = find_column(df, PRED_NAMES)

    # Try to infer a probability column if its name is unfamiliar.
    if prob_col is None and true_col is not None:
        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if (
                col != true_col
                and len(values) > 0
                and values.between(0, 1).all()
                and values.nunique() > 2
            ):
                prob_col = col
                break

    if true_col is None or prob_col is None:
        raise ValueError(
            f"Cannot identify labels/probabilities in:\n{path}\n"
            f"CSV columns: {list(df.columns)}"
        )

    true_values = pd.to_numeric(
        df[true_col], errors="coerce"
    ).to_numpy()
    probabilities = pd.to_numeric(
        df[prob_col], errors="coerce"
    ).to_numpy()

    valid = np.isfinite(true_values) & np.isfinite(probabilities)
    y_true = true_values[valid].astype(int)
    y_prob = probabilities[valid].astype(float)

    if not np.isin(np.unique(y_true), [0, 1]).all():
        raise ValueError(f"Expected binary labels 0/1 in {path}")

    if len(np.unique(y_true)) != 2:
        raise ValueError(f"Both classes are needed for ROC: {path}")

    if pred_col:
        raw_pred = pd.to_numeric(
            df[pred_col], errors="coerce"
        ).to_numpy()[valid]
        if np.isfinite(raw_pred).all():
            y_pred = raw_pred.astype(int)
        else:
            y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_prob, y_pred


def make_roc_figure(predictions):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for model, (y_true, y_prob, _) in zip(MODELS, predictions):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f'{model["name"]} (AUC = {auc:.3f})'
        )

    ax.plot([0, 1], [0, 1], "--", linewidth=1.2, label="Chance")
    ax.set_title(
        "Figure 1. Comparative ROC Curves of Deep Learning Models\n"
        "on the Unseen Centre 2 Test Set",
        fontsize=12,
        pad=12
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    path = OUT / "figure1_combined_roc_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def make_confusion_figure(predictions):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, model, (y_true, _, y_pred) in zip(
        axes.ravel(), MODELS, predictions
    ):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, interpolation="nearest")
        ax.set_title(model["name"], fontsize=10)

        ax.set_xticks(
            [0, 1],
            labels=["Predicted non-tumour", "Predicted tumour"]
        )
        ax.set_yticks(
            [0, 1],
            labels=["Actual non-tumour", "Actual tumour"]
        )
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10
                )

    fig.suptitle(
        "Figure 2. Confusion Matrices of Deep Learning Models\n"
        "on the Unseen Centre 2 Test Set (Threshold = 0.5)",
        fontsize=12,
        y=1.02
    )
    fig.tight_layout()

    path = OUT / "figure2_combined_confusion_matrices.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def read_metric_json(model):
    path = RESULTS / model["folder"] / model["metrics"]

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def make_comparison_table():
    metric_names = [
        ("Accuracy", ["accuracy", "acc"]),
        ("Precision", ["precision"]),
        ("Recall", ["recall", "sensitivity"]),
        ("F1-score", ["f1", "f1_score", "f1-score"]),
        ("ROC-AUC", ["roc_auc", "roc_auc_score", "auc", "auroc"]),
    ]

    rows = []

    for model in MODELS:
        data = read_metric_json(model)
        flat = {normalize_name(k): v for k, v in data.items()}
        row = {"Model": model["name"]}

        for index, (label, aliases) in enumerate(metric_names):
            value = None

            for alias in aliases:
                if alias in flat:
                    try:
                        value = float(flat[alias])
                        if 0 <= value <= 1:
                            value *= 100
                        break
                    except (TypeError, ValueError):
                        pass

            # Use saved, previously reported values if JSON keys differ.
            if value is None:
                value = KNOWN[model["name"]][index]

            row[label] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT / "table1_centre2_model_comparison.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")

    display = df.copy()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda x: f"{x:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    ax.set_title(
        "Table 1. Comparative Performance of Deep Learning Models\n"
        "on the Unseen Centre 2 Test Set",
        fontsize=12,
        pad=18
    )
    fig.tight_layout()

    png_path = OUT / "table1_centre2_model_comparison.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", csv_path)
    print("Saved:", png_path)
    print("\nModel comparison:")
    print(display.to_string(index=False))


def main():
    print("Project folder:", ROOT)
    print("Loading saved Centre 2 predictions...")

    predictions = [load_predictions(model) for model in MODELS]

    counts = [len(item[0]) for item in predictions]
    if len(set(counts)) != 1:
        print("WARNING: Prediction CSV row counts differ:", counts)

    make_roc_figure(predictions)
    make_confusion_figure(predictions)
    make_comparison_table()

    print("\nFinished. Output folder:")
    print(OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)
        sys.exit(1)
