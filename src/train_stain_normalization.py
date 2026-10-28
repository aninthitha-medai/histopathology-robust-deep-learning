import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import torchstain


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = r"C:\Users\Admin\histopathology_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "camelyon17_v1.0")
META_FILE = os.path.join(DATA_DIR, "metadata.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "resnet50_reinhard_best.pth"
)

BATCH_SIZE = 64
EPOCHS = 5
PATIENCE = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BUILD IMAGE PATH
# ============================================================

def build_image_path(row):

    patient = int(row["patient"])
    node = int(row["node"])
    x = int(row["x_coord"])
    y = int(row["y_coord"])

    folder = os.path.join(
        DATA_DIR,
        "patches",
        f"patient_{patient:03d}_node_{node}"
    )

    filename = (
        f"patch_patient_{patient:03d}"
        f"_node_{node}"
        f"_x_{x}"
        f"_y_{y}.png"
    )

    return os.path.join(folder, filename)


# ============================================================
# REINHARD DATASET
# ============================================================

class ReinhardDataset(Dataset):

    def __init__(
        self,
        dataframe,
        normalizer,
        training=False
    ):

        self.df = dataframe.reset_index(drop=True)
        self.normalizer = normalizer
        self.training = training

        self.tensor_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        # ----------------------------------------------------
        # Reinhard stain normalization
        # ----------------------------------------------------

        image_np = np.asarray(
            image,
            dtype=np.uint8
        )

        image_tensor = torch.from_numpy(
            image_np
        ).permute(2, 0, 1).contiguous()

        normalized = self.normalizer.normalize(
            I=image_tensor
        )

        normalized = normalized.clamp(
            0,
            255
        ).byte()

        normalized_tensor = torch.as_tensor(normalized); normalized_tensor = normalized_tensor.permute(2, 0, 1) if normalized_tensor.ndim == 3 and normalized_tensor.shape[-1] == 3 else normalized_tensor; image = transforms.ToPILImage()(normalized_tensor)

        # ----------------------------------------------------
        # Same augmentation as baseline
        # ----------------------------------------------------

        if self.training:

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            if random.random() < 0.5:

                image = image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

        image = self.tensor_transform(
            image
        )

        label = int(
            row["tumor"]
        )

        return image, label


# ============================================================
# CREATE REINHARD NORMALIZER
# ============================================================

def create_reinhard_normalizer(
    reference_path
):

    print()
    print("Reinhard reference:")
    print(reference_path)

    reference = Image.open(
        reference_path
    ).convert("RGB")

    reference_np = np.asarray(
        reference,
        dtype=np.uint8
    )

    reference_tensor = torch.from_numpy(
        reference_np
    ).permute(2, 0, 1).contiguous()

    normalizer = torchstain.normalizers.ReinhardNormalizer(
        backend="torch"
    )

    normalizer.fit(
        reference_tensor
    )

    print("Reinhard normalizer fitted.")

    return normalizer


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    print("Loading metadata...")

    df = pd.read_csv(
        META_FILE
    )

    print(
        "Metadata rows:",
        len(df)
    )

    # --------------------------------------------------------
    # BUILD PATHS
    # --------------------------------------------------------

    df["image_path"] = df.apply(
        build_image_path,
        axis=1
    )

    df = df[
        df["image_path"].map(
            os.path.exists
        )
    ].reset_index(drop=True)

    print(
        "Images found:",
        len(df)
    )

    # --------------------------------------------------------
    # EXACT BASELINE SPLIT
    # --------------------------------------------------------

    train_df = df[
        (df["split"] == 0)
        &
        (df["center"].isin([0, 3, 4]))
    ].copy()

    val_df = df[
        (df["center"] == 1)
        |
        (
            (df["split"] == 1)
            &
            (df["center"].isin([0, 3, 4]))
        )
    ].copy()

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(val_df)
    )

    # --------------------------------------------------------
    # REINHARD REFERENCE
    #
    # IMPORTANT:
    # Reference comes ONLY from training data.
    # Centre 2 is never used here.
    # --------------------------------------------------------

    reference_path = train_df.iloc[0][
        "image_path"
    ]

    normalizer = create_reinhard_normalizer(
        reference_path
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = ReinhardDataset(
        train_df,
        normalizer,
        training=True
    )

    val_dataset = ReinhardDataset(
        val_df,
        normalizer,
        training=False
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # --------------------------------------------------------
    # RESNET-50
    # --------------------------------------------------------

    print()
    print("Loading ResNet-50...")

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model = model.to(device)

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    from sklearn.metrics import roc_auc_score

    best_auc = -1.0
    patience_counter = 0

    for epoch in range(EPOCHS):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print("=" * 60)

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0
        total_samples = 0

        for batch_idx, (
            images,
            labels
        ) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(device.type == "cuda")
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += (
                labels.size(0)
            )

            if (
                batch_idx + 1
            ) % 200 == 0:

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )

        train_loss = (
            running_loss /
            total_samples
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_labels = []
        val_probabilities = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(device.type == "cuda")
                ):

                    outputs = model(
                        images
                    )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )[:, 1]

                val_labels.extend(
                    labels.numpy().tolist()
                )

                val_probabilities.extend(
                    probabilities.cpu()
                    .numpy()
                    .tolist()
                )

        val_auc = roc_auc_score(
            val_labels,
            val_probabilities
        )

        print()
        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{val_auc:.4f}"
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val_auc > best_auc:

            best_auc = val_auc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_auc":
                        val_auc,

                    "model_name":
                        "ResNet50_Reinhard",

                    "input_size":
                        96,

                    "batch_size":
                        BATCH_SIZE
                },
                MODEL_PATH
            )

            print(
                "BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "REINHARD STAIN NORMALIZATION TRAINING COMPLETE"
    )
    print("=" * 60)

    print(
        f"Best Validation ROC-AUC: "
        f"{best_auc:.4f}"
    )

    print(
        "Saved model:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()

