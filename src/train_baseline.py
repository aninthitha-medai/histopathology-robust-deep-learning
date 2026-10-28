import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = r"C:\Users\Admin\histopathology_project"

METADATA_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "camelyon17_v1.0",
    "metadata.csv"
)

IMAGE_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "camelyon17_v1.0",
    "patches"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "resnet50_baseline_best.pth"
)

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

BATCH_SIZE = 64

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

NUM_CLASSES = 2

IMAGE_SIZE = 96

# Number of batches used only for the initial speed test.
# This does NOT train the full model.
SPEED_TEST_BATCHES = 100


# ------------------------------------------------------------
# Import our dataset
# ------------------------------------------------------------

sys.path.append(
    os.path.join(PROJECT_ROOT, "src")
)

from dataset import CamelyonDataset


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("CAMELYON17 ROBUST HISTOPATHOLOGY BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Training must use the RTX 3070 Ti."
        )

    device = torch.device("cuda")

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    # Enable cuDNN autotuning for fixed image sizes.
    torch.backends.cudnn.benchmark = True

    print()

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_CSV}"
        )

    if not os.path.exists(IMAGE_ROOT):
        raise FileNotFoundError(
            f"Image folder not found:\n{IMAGE_ROOT}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Image preprocessing
    #
    # CAMELYON17-WILDS images are already 96x96.
    # We keep them at their native resolution.
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.5
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("Loading datasets...")

    train_dataset = CamelyonDataset(
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        dataset_type="train",
        transform=train_transform
    )

    val_dataset = CamelyonDataset(
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        dataset_type="val",
        transform=eval_transform
    )

    test_dataset = CamelyonDataset(
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        dataset_type="test",
        transform=eval_transform
    )

    print()
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))
    print()

    # --------------------------------------------------------
    # DataLoaders
    #
    # Windows-safe because everything is inside main().
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print(
        "Training batches per epoch:",
        len(train_loader)
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading pretrained ResNet-50...")

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(
        weights=weights
    )

    # Two classes:
    # 0 = non-tumour
    # 1 = tumour

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    print("Model: ResNet-50")
    print("Input size:", f"{IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Classes:", NUM_CLASSES)

    print()

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Mixed precision
    #
    # This reduces GPU computation/memory cost on CUDA.
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    # ========================================================
    # SPEED TEST
    # ========================================================

    print("=" * 70)
    print("RUNNING 100-BATCH SPEED TEST")
    print("=" * 70)

    model.train()

    speed_start = time.time()

    speed_batches = 0

    for images, labels in train_loader:

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
            dtype=torch.float16
        ):

            outputs = model(images)

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

        speed_batches += 1

        if speed_batches >= SPEED_TEST_BATCHES:
            break

    torch.cuda.synchronize()

    speed_time = (
        time.time() - speed_start
    )

    images_processed = (
        speed_batches * BATCH_SIZE
    )

    images_per_second = (
        images_processed / speed_time
    )

    print()
    print(
        "Speed-test batches:",
        speed_batches
    )

    print(
        "Speed-test time:",
        f"{speed_time:.2f} seconds"
    )

    print(
        "Images processed:",
        images_processed
    )

    print(
        "Images/second:",
        f"{images_per_second:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Estimate only for planning.
    # These are NOT experimental results.
    # --------------------------------------------------------

    estimated_epoch_seconds = (
        len(train_loader) / speed_batches
    ) * speed_time

    print(
        "Estimated full-epoch time:",
        f"{estimated_epoch_seconds / 60:.2f} minutes"
    )

    print()
    print(
        "IMPORTANT: The estimated time above is "
        "only a speed estimate."
    )

    print(
        "It is NOT a research result."
    )

    print()

    # ========================================================
    # RESET MODEL
    #
    # The speed test performed 100 optimization steps.
    # We reset the model so the actual experiment starts
    # from the original pretrained weights.
    # ========================================================

    print("=" * 70)
    print("RESETTING MODEL FOR ACTUAL EXPERIMENT")
    print("=" * 70)

    model = models.resnet50(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    print("Model reset successfully.")
    print()

    # ========================================================
    # EVALUATION FUNCTION
    # ========================================================

    def evaluate(
        model,
        loader,
        dataset_name
    ):

        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(images)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                all_labels.extend(
                    labels.numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_probabilities.extend(
                    probabilities[:, 1]
                    .cpu()
                    .numpy()
                )

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        precision = precision_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )

        print()
        print(
            f"{dataset_name} Accuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"{dataset_name} Precision: "
            f"{precision:.4f}"
        )

        print(
            f"{dataset_name} Recall   : "
            f"{recall:.4f}"
        )

        print(
            f"{dataset_name} F1       : "
            f"{f1:.4f}"
        )

        print(
            f"{dataset_name} ROC-AUC  : "
            f"{auc:.4f}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc
        }

    # ========================================================
    # ACTUAL TRAINING
    # ========================================================

    print("=" * 70)
    print("STARTING ACTUAL BASELINE TRAINING")
    print("=" * 70)

    best_val_auc = -1.0

    patience = 2

    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        model.train()

        epoch_start = time.time()

        running_loss = 0.0

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
                dtype=torch.float16
            ):

                outputs = model(images)

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

            running_loss += loss.item()

            if (
                batch_idx + 1
            ) % 100 == 0:

                average_loss = (
                    running_loss / 100
                )

                print(
                    f"Batch "
                    f"[{batch_idx + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: "
                    f"{average_loss:.4f}"
                )

                running_loss = 0.0

        torch.cuda.synchronize()

        epoch_time = (
            time.time() - epoch_start
        )

        print()
        print(
            f"Epoch time: "
            f"{epoch_time / 60:.2f} minutes"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            "Validation"
        )

        current_auc = (
            val_metrics["roc_auc"]
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if current_auc > best_val_auc:

            best_val_auc = current_auc

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_metrics":
                        val_metrics,

                    "model_name":
                        "ResNet-50",

                    "input_size":
                        IMAGE_SIZE,

                    "batch_size":
                        BATCH_SIZE
                },
                MODEL_PATH
            )

            print()
            print(
                "New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "Validation ROC-AUC did not improve."
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best model loaded from:"
    )

    print(MODEL_PATH)

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL UNSEEN-CENTRE TEST")
    print("=" * 70)

    test_metrics = evaluate(
        model,
        test_loader,
        "Test"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print("FINAL TEST RESULTS")

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print()
    print("Best model:")
    print(MODEL_PATH)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ------------------------------------------------------------
# Windows-safe entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()