import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Admin\histopathology_project"
)

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "resnet50_baseline_best.pth"
)

METADATA_PATH = (
    PROJECT_DIR
    / "data"
    / "camelyon17_v1.0"
    / "metadata.csv"
)

PATCH_DIR = (
    PROJECT_DIR
    / "data"
    / "camelyon17_v1.0"
    / "patches"
)


# ============================================================
# 2. CHECK PATHS
# ============================================================

print("=" * 60)
print("       RANDOM CENTRE-2 HISTOPATHOLOGY PREDICTION")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"\nMetadata file not found:\n{METADATA_PATH}"
    )

if not PATCH_DIR.exists():
    raise FileNotFoundError(
        f"\nPatch folder not found:\n{PATCH_DIR}"
    )


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. CREATE THE SAME RESNET-50 MODEL
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(
    weights=None
)

# Two classes:
# 0 = Non-Tumor
# 1 = Tumor

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# ============================================================
# 5. LOAD OUR BEST TRAINED CHECKPOINT
# ============================================================

print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Our checkpoint is a dictionary containing:
# model_state_dict
# optimizer_state_dict
# epoch
# val_metrics
# etc.

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "\nThe checkpoint does not contain "
        "'model_state_dict'."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Trained model loaded successfully!")


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor()
])


# ============================================================
# 7. READ METADATA
# ============================================================

print("\nReading metadata...")

metadata = pd.read_csv(
    METADATA_PATH
)

# Centre 2 is our unseen test centre.

centre2 = metadata[
    metadata["center"] == 2
].copy()

print(
    "Centre-2 images available:",
    len(centre2)
)


# ============================================================
# 8. SELECT ONE RANDOM IMAGE FROM CENTRE 2
# ============================================================

random_row = centre2.sample(
    n=1,
    random_state=random.randint(
        0,
        1_000_000
    )
).iloc[0]


patient = int(
    random_row["patient"]
)

node = int(
    random_row["node"]
)

x_coord = int(
    random_row["x_coord"]
)

y_coord = int(
    random_row["y_coord"]
)

actual_label = int(
    random_row["tumor"]
)


print("\nSelected metadata:")

print(
    "Patient:",
    patient
)

print(
    "Node:",
    node
)

print(
    "X coordinate:",
    x_coord
)

print(
    "Y coordinate:",
    y_coord
)


# ============================================================
# 9. FIND THE ACTUAL IMAGE FILE
# ============================================================

print("\nSearching for the matching image...")

# We DON'T assume a filename format.
# Instead, we search the actual files in the patches folder.

patient_text = str(patient)
node_text = str(node)
x_text = str(x_coord)
y_text = str(y_coord)

image_path = None

valid_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".PNG",
    ".JPG",
    ".JPEG"
}

for root, dirs, files in os.walk(PATCH_DIR):

    for file_name in files:

        extension = Path(
            file_name
        ).suffix

        if extension not in valid_extensions:
            continue

        lower_name = file_name.lower()

        # Match all four pieces of metadata
        # somewhere in the actual filename.

        if (
            patient_text in lower_name
            and node_text in lower_name
            and x_text in lower_name
            and y_text in lower_name
        ):

            image_path = (
                Path(root) / file_name
            )

            break

    if image_path is not None:
        break


# ============================================================
# 10. MAKE SURE IMAGE WAS FOUND
# ============================================================

if image_path is None:

    print(
        "\nERROR: Could not find the image "
        "matching the selected metadata."
    )

    print(
        "\nSelected values:"
    )

    print(
        "Patient =",
        patient
    )

    print(
        "Node =",
        node
    )

    print(
        "X =",
        x_coord
    )

    print(
        "Y =",
        y_coord
    )

    raise FileNotFoundError(
        "\nNo matching image was found "
        "inside the patches folder."
    )


print(
    "\nImage found:"
)

print(
    image_path
)


# ============================================================
# 11. LOAD IMAGE
# ============================================================

image = Image.open(
    image_path
).convert("RGB")


print(
    "\nImage size:",
    image.size
)


# ============================================================
# 12. PREPARE IMAGE FOR MODEL
# ============================================================

input_tensor = transform(
    image
)

input_tensor = input_tensor.unsqueeze(
    0
)

input_tensor = input_tensor.to(
    device
)


# ============================================================
# 13. MAKE PREDICTION
# ============================================================

print(
    "\nRunning prediction..."
)

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()


# ============================================================
# 14. CLASS NAMES
# ============================================================

class_names = {
    0: "Non-Tumor",
    1: "Tumor"
}

actual_name = class_names[
    actual_label
]

predicted_name = class_names[
    predicted_class
]


# ============================================================
# 15. PRINT PREDICTION RESULT
# ============================================================

print("\n")
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print(
    "\nActual Label:"
)

print(
    actual_name
)

print(
    "\nPredicted Label:"
)

print(
    predicted_name
)

print(
    "\nConfidence:"
)

print(
    f"{confidence * 100:.2f}%"
)

if predicted_class == actual_label:

    print(
        "\nResult:"
    )

    print(
        "CORRECT PREDICTION"
    )

else:

    print(
        "\nResult:"
    )

    print(
        "INCORRECT PREDICTION"
    )

print(
    "\nImage:"
)

print(
    image_path
)

print("=" * 60)


# ============================================================
# 16. DISPLAY THE HISTOPATHOLOGY IMAGE
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    image
)

plt.axis(
    "off"
)

plt.title(
    f"Actual: {actual_name}\n"
    f"Predicted: {predicted_name}\n"
    f"Confidence: {confidence * 100:.2f}%",
    fontsize=14
)

plt.tight_layout()

plt.show()