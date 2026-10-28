import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Admin\histopathology_project"

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "camelyon17_v1.0",
    "metadata.csv"
)

PATCHES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "camelyon17_v1.0",
    "patches"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "resnet50_baseline_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET CLASS
# ============================================================

class SeenCentreDataset(Dataset):

    def __init__(
        self,
        metadata,
        patches_dir,
        transform=None
    ):

        self.metadata = metadata.reset_index(
            drop=True
        )

        self.patches_dir = patches_dir
        self.transform = transform

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        patient = int(row["patient"])
        node = int(row["node"])
        x = int(row["x_coord"])
        y = int(row["y_coord"])

        label = int(row["tumor"])

        folder = (
            f"patient_{patient:03d}_node_{node}"
        )

        filename = (
            f"patch_patient_{patient:03d}"
            f"_node_{node}"
            f"_x_{x}"
            f"_y_{y}.png"
        )

        image_path = os.path.join(
            self.patches_dir,
            folder,
            filename
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "Total metadata rows:",
    len(metadata)
)


# ============================================================
# SELECT HELD-OUT SEEN CENTRES
# ============================================================

# split == 1:
# held-out portion of the dataset
#
# Centres 0, 3, 4:
# these are the centres used during training

seen_val = metadata[
    (metadata["split"] == 1)
    &
    (metadata["center"].isin([0, 3, 4]))
].copy()


print(
    "\nHeld-out seen-centre samples:",
    len(seen_val)
)

print("\nSamples by centre:")

print(
    seen_val["center"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LOAD RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_subset(subset):

    dataset = SeenCentreDataset(
        subset,
        PATCHES_DIR,
        transform
    )

    # IMPORTANT:
    # num_workers=0 avoids the Windows
    # multiprocessing error.

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    all_labels = []
    all_preds = []
    all_probs = []

    print(
        "Evaluating",
        len(dataset),
        "images..."
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            # Probability of Tumor
            tumor_prob = probabilities[:, 1]

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_preds.extend(
                predictions.cpu().numpy()
            )

            all_probs.extend(
                tumor_prob.cpu().numpy()
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# OVERALL SEEN-CENTRE PERFORMANCE
# ============================================================

print(
    "\n======================================"
)

print(
    "EVALUATING HELD-OUT SEEN CENTRES"
)

print(
    "======================================"
)


(
    acc,
    precision,
    recall,
    f1,
    auc
) = evaluate_subset(
    seen_val
)


print(
    "\n======================================"
)

print(
    "HELD-OUT SEEN-CENTRE PERFORMANCE"
)

print(
    "======================================"
)

print(
    f"Accuracy : {acc:.4f} "
    f"({acc * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall   : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC  : {auc:.4f} "
    f"({auc * 100:.2f}%)"
)


# ============================================================
# CENTRE-WISE PERFORMANCE
# ============================================================

print(
    "\n======================================"
)

print(
    "CENTRE-WISE PERFORMANCE"
)

print(
    "======================================"
)


for centre in [0, 3, 4]:

    centre_data = seen_val[
        seen_val["center"] == centre
    ].copy()

    print(
        f"\n---------- Centre {centre} ----------"
    )

    print(
        "Samples:",
        len(centre_data)
    )

    (
        acc,
        precision,
        recall,
        f1,
        auc
    ) = evaluate_subset(
        centre_data
    )

    print(
        f"Accuracy : {acc:.4f} "
        f"({acc * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC  : {auc:.4f} "
        f"({auc * 100:.2f}%)"
    )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n======================================"
)

print(
    "EVALUATION COMPLETE"
)

print(
    "======================================"
)