import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = r"C:\Users\Admin\histopathology_project"
METADATA_CSV = os.path.join(PROJECT_ROOT, "data", "camelyon17_v1.0", "metadata.csv")
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "data", "camelyon17_v1.0", "patches")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "resnet50_robust_aug_best.pth")
METRICS_PATH = os.path.join(RESULTS_DIR, "robust_aug_training_summary.json")

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
NUM_CLASSES = 2
IMAGE_SIZE = 96
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device, name="Validation"):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs.float().cpu().numpy().tolist())
    metrics = {
        "accuracy": float(accuracy_score(labels_all, preds_all)),
        "precision": float(precision_score(labels_all, preds_all, zero_division=0)),
        "recall": float(recall_score(labels_all, preds_all, zero_division=0)),
        "f1": float(f1_score(labels_all, preds_all, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_all, probs_all)),
    }
    print(f"{name}: Accuracy={metrics['accuracy']:.4f} | Precision={metrics['precision']:.4f} | "
          f"Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics

def main():
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This experiment expects the RTX 3070 Ti.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("=" * 72)
    print("EXPERIMENT 3: RESNET-50 WITH ROBUST DATA AUGMENTATION")
    print("Device:", torch.cuda.get_device_name(0))
    print("Checkpoint:", MODEL_PATH)
    print("=" * 72)

    for path in (METADATA_CSV, IMAGE_ROOT):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from dataset import CamelyonDataset

    # Moderate augmentation; this experiment does not use stain normalization.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CamelyonDataset(metadata_csv=METADATA_CSV, image_root=IMAGE_ROOT,
                                    dataset_type="train", transform=train_transform)
    val_dataset = CamelyonDataset(metadata_csv=METADATA_CSV, image_root=IMAGE_ROOT,
                                  dataset_type="val", transform=eval_transform)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    loader_options = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    best_val_auc = -1.0
    patience = 2
    epochs_without_improvement = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        start = time.time()
        running_loss = 0.0
        batch_count = 0
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batch_count += 1
            if batch_idx % 100 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] mean loss={running_loss / batch_count:.4f}")
                running_loss, batch_count = 0.0, 0

        torch.cuda.synchronize()
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "validation": val_metrics,
                        "epoch_minutes": round((time.time() - start) / 60, 2)})
        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "model_name": "ResNet-50 robust augmentation",
                "input_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "augmentation": ["RandomHorizontalFlip(0.5)", "RandomVerticalFlip(0.5)",
                                 "RandomRotation(15 degrees)",
                                 "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)"]
            }, MODEL_PATH)
            print("  Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print("  Validation ROC-AUC did not improve.")
        print(f"  Epoch duration: {(time.time() - start) / 60:.2f} minutes")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    summary = {
        "experiment": "ResNet-50 with robust data augmentation",
        "checkpoint": MODEL_PATH,
        "best_validation_roc_auc": best_val_auc,
        "history": history,
        "note": "Evaluate all metadata rows with center == 2 separately; validation metrics are not held-out-centre results."
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nTRAINING COMPLETE")
    print("Best validation ROC-AUC:", f"{best_val_auc:.4f}")
    print("Checkpoint:", MODEL_PATH)
    print("Training summary:", METRICS_PATH)
    print("Next: evaluate this checkpoint on every metadata row where center == 2.")

if __name__ == "__main__":
    main()