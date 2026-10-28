
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Users\Admin\histopathology_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "camelyon17_v1.0")
PATCH_DIR = os.path.join(DATA_DIR, "patches")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

# IMPORTANT: This must be the baseline checkpoint.
MODEL_PATH = os.path.join(
    PROJECT_DIR, "results", "resnet50_baseline_best.pth"
)

# Keep baseline results separate from Reinhard results.
OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "results", "report_figures_baseline_corrected"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
print("Checkpoint:", MODEL_PATH)
print("Output folder:", OUTPUT_DIR)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Baseline checkpoint not found: {MODEL_PATH}")

if not os.path.isfile(METADATA_PATH):
    raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

# ============================================================
# DATASET: ALL CENTRE 2 PATCHES
# ============================================================

metadata = pd.read_csv(METADATA_PATH)
test_metadata = metadata[metadata["center"] == 2].reset_index(drop=True)

print("Centre 2 samples:", len(test_metadata))

if len(test_metadata) == 0:
    raise RuntimeError("No Centre 2 patches found in metadata.")

class Centre2Dataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patient = int(row["patient"])
        node = int(row["node"])
        x = int(row["x_coord"])
        y = int(row["y_coord"])

        folder = f"patient_{patient:03d}_node_{node}"
        filename = (
            f"patch_patient_{patient:03d}"
            f"_node_{node}_x_{x}_y_{y}.png"
        )
        path = os.path.join(PATCH_DIR, folder, filename)

        with Image.open(path) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        label = int(row["tumor"])

        return image, label, path


dataset = Centre2Dataset(test_metadata)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

# ============================================================
# LOAD BASELINE RESNET-50 CHECKPOINT
# ============================================================

print("Loading baseline ResNet-50...")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Baseline checkpoint loaded successfully.")

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

all_labels = []
all_probs = []
all_preds = []
all_paths = []

print("Evaluating all Centre 2 patches...")

with torch.no_grad():
    for batch_idx, (images, labels, paths) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probabilities.cpu().numpy().tolist())
        all_preds.extend(predictions.cpu().numpy().tolist())
        all_paths.extend(paths)

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {(batch_idx + 1) * BATCH_SIZE:,} patches")

# ============================================================
# METRICS
# ============================================================

y_true = np.asarray(all_labels, dtype=int)
y_prob = np.asarray(all_probs, dtype=float)
y_pred = np.asarray(all_preds, dtype=int)

metrics = {
    "model": "Baseline ResNet-50",
    "checkpoint": MODEL_PATH,
    "evaluation_centre": 2,
    "n_samples": int(len(y_true)),
    "threshold": 0.5,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist(),
}

print("\n===== BASELINE RESNET-50: CENTRE 2 =====")
print("Samples:   ", metrics["n_samples"])
print(f"Accuracy:   {metrics['accuracy'] * 100:.2f}%")
print(f"Precision:  {metrics['precision'] * 100:.2f}%")
print(f"Recall:     {metrics['recall'] * 100:.2f}%")
print(f"F1-score:   {metrics['f1_score'] * 100:.2f}%")
print(f"ROC-AUC:    {metrics['roc_auc'] * 100:.2f}%")
print("Confusion matrix:")
print(np.array(metrics["confusion_matrix"]))

# ============================================================
# SAVE RESULTS WITHOUT OVERWRITING PREVIOUS FILES
# ============================================================

predictions_df = pd.DataFrame({
    "label": y_true,
    "probability_tumor": y_prob,
    "prediction": y_pred,
    "path": all_paths,
})

csv_path = os.path.join(
    OUTPUT_DIR, "centre2_predictions_baseline_corrected.csv"
)
predictions_df.to_csv(csv_path, index=False)

json_path = os.path.join(
    OUTPUT_DIR, "centre2_metrics_baseline_corrected.json"
)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

cm = np.array(metrics["confusion_matrix"])

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Non-Tumour", "Tumour"])
ax.set_yticklabels(["Non-Tumour", "Tumour"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Baseline ResNet-50 — Unseen Centre 2")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, f"{cm[i, j]:,}",
            ha="center", va="center", fontsize=14
        )

plt.tight_layout()
cm_path = os.path.join(
    OUTPUT_DIR, "fig_centre2_baseline_corrected_confusion_matrix.png"
)
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# ROC CURVE FIGURE
# ============================================================

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc_value = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, label=f"Baseline ResNet-50 (AUC = {roc_auc_value:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Baseline ResNet-50 ROC — Unseen Centre 2")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR, "fig_centre2_baseline_corrected_roc_curve.png"
)
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n===== FILES SAVED =====")
print("Predictions:", csv_path)
print("Metrics:    ", json_path)
print("Confusion:  ", cm_path)
print("ROC curve:  ", roc_path)
print("\nBaseline evaluation completed.")
