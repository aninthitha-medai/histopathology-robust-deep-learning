
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "camelyon17_v1.0"
PATCHES_DIR = DATA_DIR / "patches"
METADATA_PATH = DATA_DIR / "metadata.csv"

MODEL_PATH = (
    PROJECT_DIR / "results" / "resnet50_robust_aug_best.pth"
)
OUTPUT_DIR = PROJECT_DIR / "results" / "report_figures_robust_aug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
THRESHOLD = 0.5
EXPECTED_SAMPLES = 85054

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("EXPERIMENT 3: ROBUST AUGMENTATION — CENTRE 2 EVALUATION")
print(f"Device: {device}")
print(f"Checkpoint: {MODEL_PATH}")
print("=" * 65)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

# IMPORTANT: No Reinhard normalization is applied here.
# Only standard ImageNet normalization is used.

image_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# 2. CENTRE 2 DATASET
# ============================================================

class Centre2Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        patient = int(row["patient"])
        node = int(row["node"])
        x_coord = int(row["x_coord"])
        y_coord = int(row["y_coord"])

        patch_path = (
            PATCHES_DIR
            / f"patient_{patient:03d}_node_{node}"
            / (
                f"patch_patient_{patient:03d}_node_{node}"
                f"_x_{x_coord}_y_{y_coord}.png"
            )
        )

        with Image.open(patch_path) as image:
            image = image.convert("RGB")
            image_tensor = image_transform(image)

        label = int(row["tumor"])
        return image_tensor, label, str(patch_path)


# ============================================================
# 3. LOAD ALL CENTRE 2 SAMPLES
# ============================================================

print("\nLoading metadata...")
metadata = pd.read_csv(METADATA_PATH)

# Use ALL Centre 2 rows. Do not filter by split.
test_metadata = metadata[metadata["center"] == 2].copy()

if len(test_metadata) != EXPECTED_SAMPLES:
    raise RuntimeError(
        f"Expected {EXPECTED_SAMPLES} Centre 2 samples, "
        f"found {len(test_metadata)}. Check metadata."
    )

print(f"Centre 2 test samples: {len(test_metadata):,}")

dataset = Centre2Dataset(test_metadata)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# 4. LOAD ROBUST-AUGMENTATION RESNET-50
# ============================================================

print("\nLoading robust-augmentation checkpoint...")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ============================================================
# 5. EVALUATE
# ============================================================

print("\nEvaluating all Centre 2 patches...")

all_labels = []
all_probabilities = []
all_predictions = []
all_paths = []

with torch.no_grad():
    for batch_index, (images, labels, paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = (probabilities >= THRESHOLD).long()

        all_labels.extend(labels.numpy().tolist())
        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_paths.extend(paths)

        if (batch_index + 1) % 100 == 0:
            print(f"Processed {len(all_labels):,} patches...")

y_true = np.asarray(all_labels)
y_prob = np.asarray(all_probabilities)
y_pred = np.asarray(all_predictions)

if len(y_true) != EXPECTED_SAMPLES:
    raise RuntimeError(
        f"Evaluated {len(y_true)} samples, expected {EXPECTED_SAMPLES}."
    )

metrics = {
    "model": "Robust-augmentation ResNet-50",
    "test_domain": "Centre 2",
    "n_samples": int(len(y_true)),
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
}

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
metrics["confusion_matrix"] = cm.tolist()

metrics_path = OUTPUT_DIR / "centre2_metrics_robust_aug.json"
predictions_path = OUTPUT_DIR / "centre2_predictions_robust_aug.csv"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame({
    "patch_path": all_paths,
    "true_label": y_true,
    "tumour_probability": y_prob,
    "predicted_label": y_pred,
}).to_csv(predictions_path, index=False)

print("\n========== ROBUST AUGMENTATION: CENTRE 2 ==========")
print(f"Samples:   {metrics['n_samples']:,}")
print(f"Accuracy:  {metrics['accuracy']:.4%}")
print(f"Precision: {metrics['precision']:.4%}")
print(f"Recall:    {metrics['recall']:.4%}")
print(f"F1-score:  {metrics['f1']:.4%}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4%}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(cm)

print("\nSaved metrics:", metrics_path)
print("Saved predictions:", predictions_path)
print("\nCENTRE 2 EVALUATION COMPLETE")
