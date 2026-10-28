import os
import random

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from PIL import Image
import matplotlib.pyplot as plt


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
# MODEL
# ============================================================

print("\nLoading baseline ResNet-50...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

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
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)

# Centre 2 = completely unseen test centre
centre2 = metadata[
    metadata["center"] == 2
].copy()

print(
    "\nCentre 2 images available:",
    len(centre2)
)


# ============================================================
# GET IMAGE PATH
# ============================================================

def get_image_path(row):

    patient = int(row["patient"])
    node = int(row["node"])
    x = int(row["x_coord"])
    y = int(row["y_coord"])

    folder = (
        f"patient_{patient:03d}_node_{node}"
    )

    filename = (
        f"patch_patient_{patient:03d}"
        f"_node_{node}"
        f"_x_{x}"
        f"_y_{y}.png"
    )

    return os.path.join(
        PATCHES_DIR,
        folder,
        filename
    )


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_row(row):

    image_path = get_image_path(row)

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0, predicted_class]
            .item()
        )

    actual_class = int(
        row["tumor"]
    )

    return (
        image,
        image_path,
        actual_class,
        predicted_class,
        confidence
    )


# ============================================================
# SEARCH FOR THREE EXAMPLES
# ============================================================

print("\nSearching for demonstration examples...")

# Fixed seed so the examples don't randomly change
random.seed(42)

sample = centre2.sample(
    n=min(3000, len(centre2)),
    random_state=42
)

correct_tumor = None
correct_non_tumor = None
incorrect = None


for _, row in sample.iterrows():

    try:

        result = predict_row(row)

        image, image_path, actual, predicted, confidence = result

        # ----------------------------------------------------
        # Correct Tumor
        # ----------------------------------------------------

        if (
            actual == 1
            and predicted == 1
            and correct_tumor is None
        ):

            correct_tumor = result

            print(
                "Found correct Tumor example."
            )

        # ----------------------------------------------------
        # Correct Non-Tumor
        # ----------------------------------------------------

        if (
            actual == 0
            and predicted == 0
            and correct_non_tumor is None
        ):

            correct_non_tumor = result

            print(
                "Found correct Non-Tumor example."
            )

        # ----------------------------------------------------
        # Incorrect Prediction
        # ----------------------------------------------------

        if (
            actual != predicted
            and incorrect is None
        ):

            incorrect = result

            print(
                "Found incorrect prediction."
            )

        # Stop once all three are found
        if (
            correct_tumor is not None
            and correct_non_tumor is not None
            and incorrect is not None
        ):

            break

    except Exception:
        continue


# ============================================================
# CHECK
# ============================================================

if correct_tumor is None:
    print(
        "\nCould not find a correct Tumor example."
    )

if correct_non_tumor is None:
    print(
        "\nCould not find a correct Non-Tumor example."
    )

if incorrect is None:
    print(
        "\nCould not find an incorrect example."
    )


# ============================================================
# PRINT RESULTS
# ============================================================

examples = [
    (
        "CORRECT TUMOR",
        correct_tumor
    ),
    (
        "CORRECT NON-TUMOR",
        correct_non_tumor
    ),
    (
        "INCORRECT PREDICTION",
        incorrect
    )
]


print(
    "\n"
    + "=" * 60
)

print(
    "BASELINE MODEL DEMONSTRATION"
)

print(
    "=" * 60
)


for title, result in examples:

    if result is None:
        continue

    image, image_path, actual, predicted, confidence = result

    actual_name = (
        "Tumor"
        if actual == 1
        else "Non-Tumor"
    )

    predicted_name = (
        "Tumor"
        if predicted == 1
        else "Non-Tumor"
    )

    print(
        f"\n{title}"
    )

    print(
        "Image:",
        os.path.basename(image_path)
    )

    print(
        "Actual:",
        actual_name
    )

    print(
        "Predicted:",
        predicted_name
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        "Correct:",
        actual == predicted
    )


# ============================================================
# DISPLAY THREE EXAMPLES
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


for ax, (title, result) in zip(
    axes,
    examples
):

    if result is None:

        ax.axis("off")
        continue

    image, image_path, actual, predicted, confidence = result

    actual_name = (
        "Tumor"
        if actual == 1
        else "Non-Tumor"
    )

    predicted_name = (
        "Tumor"
        if predicted == 1
        else "Non-Tumor"
    )

    ax.imshow(image)

    ax.axis("off")

    ax.set_title(
        f"{title}\n"
        f"Actual: {actual_name}\n"
        f"Predicted: {predicted_name}\n"
        f"Confidence: {confidence * 100:.2f}%",
        fontsize=11
    )


plt.tight_layout()

plt.show()