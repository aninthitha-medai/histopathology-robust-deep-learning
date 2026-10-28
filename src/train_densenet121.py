
import os
import sys
import time
import json

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
    roc_auc_score,
)

PROJECT_ROOT = r"C:\Users\Admin\histopathology_project"
METADATA_CSV = os.path.join(
    PROJECT_ROOT, "data", "camelyon17_v1.0", "metadata.csv"
)
IMAGE_ROOT = os.path.join(
    PROJECT_ROOT, "data", "camelyon17_v1.0", "patches"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "densenet121_best.pth")
SUMMARY_PATH = os.path.join(
    RESULTS_DIR, "densenet121_training_summary.json"
)

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
PATIENCE = 2
IMAGE_SIZE = 96

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from dataset import CamelyonDataset


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; expected RTX 3070 Ti.")

    device = torch.device("cuda")
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print("\nLoading datasets...")
    train_dataset = CamelyonDataset(
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        dataset_type="train",
        transform=train_transform,
    )
    val_dataset = CamelyonDataset(
        metadata_csv=METADATA_CSV,
        image_root=IMAGE_ROOT,
        dataset_type="val",
        transform=eval_transform,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print("\nLoading pretrained DenseNet-121...")
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")

    def evaluate():
        model.eval()
        labels_all, predictions_all, probabilities_all = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    outputs = model(images)

                probabilities = torch.softmax(outputs.float(), dim=1)
                predictions = outputs.argmax(dim=1)

                labels_all.extend(labels.numpy().tolist())
                predictions_all.extend(predictions.cpu().tolist())
                probabilities_all.extend(
                    probabilities[:, 1].cpu().tolist()
                )

        return {
            "accuracy": float(accuracy_score(labels_all, predictions_all)),
            "precision": float(precision_score(
                labels_all, predictions_all, zero_division=0
            )),
            "recall": float(recall_score(
                labels_all, predictions_all, zero_division=0
            )),
            "f1": float(f1_score(
                labels_all, predictions_all, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(
                labels_all, probabilities_all
            )),
        }

    best_auc = -1.0
    no_improvement = 0
    history = []

    print("\n" + "=" * 60)
    print("STARTING DENSENET-121 TRAINING")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}", flush=True)
        model.train()
        start = time.time()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {running_loss / 100:.4f}",
                    flush=True,
                )
                running_loss = 0.0

        torch.cuda.synchronize()
        epoch_seconds = time.time() - start
        metrics = evaluate()

        record = {
            "epoch": epoch + 1,
            "epoch_seconds": round(epoch_seconds, 2),
            **metrics,
        }
        history.append(record)

        print(f"Epoch time: {epoch_seconds / 60:.2f} minutes")
        print(
            "Validation — "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}",
            flush=True,
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            no_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": metrics,
                    "model_name": "DenseNet-121",
                    "input_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                },
                MODEL_PATH,
            )
            print("Best DenseNet-121 checkpoint saved.", flush=True)
        else:
            no_improvement += 1
            print("Validation ROC-AUC did not improve.", flush=True)

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": "DenseNet-121",
                    "best_validation_roc_auc": best_auc,
                    "epochs_completed": epoch + 1,
                    "history": history,
                    "checkpoint": MODEL_PATH,
                },
                f,
                indent=2,
            )

        if no_improvement >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

    print("\n" + "=" * 60)
    print("DENSENET-121 TRAINING COMPLETE")
    print("Best validation ROC-AUC:", f"{best_auc:.4f}")
    print("Checkpoint:", MODEL_PATH)
    print("Summary:", SUMMARY_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
