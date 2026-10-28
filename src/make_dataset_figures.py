import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Admin\histopathology_project\data\camelyon17_v1.0"

METADATA_PATH = os.path.join(BASE_DIR, "metadata.csv")
PATCH_DIR = os.path.join(BASE_DIR, "patches")

OUTPUT_DIR = r"C:\Users\Admin\histopathology_project\results\report_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

metadata = pd.read_csv(METADATA_PATH)

print(f"Total metadata rows: {len(metadata)}")

# ============================================================
# FIND ACTUAL IMAGE FILES
# ============================================================

print("Searching for available image patches...")

available_images = []

for root, dirs, files in os.walk(PATCH_DIR):

    for file in files:

        if file.lower().endswith(".png"):

            full_path = os.path.join(root, file)

            available_images.append(full_path)

print(f"Total PNG images found: {len(available_images)}")

if len(available_images) == 0:

    raise FileNotFoundError(
        f"No PNG images found inside:\n{PATCH_DIR}"
    )

# ============================================================
# MATCH IMAGES WITH METADATA
# ============================================================

image_records = []

for image_path in available_images:

    filename = os.path.basename(image_path)

    # Example:
    # patch_patient_004_node_4_x_10016_y_16704.png

    name = filename.replace(".png", "")

    parts = name.split("_")

    try:

        patient = int(parts[2])
        node = int(parts[4])
        x_coord = int(parts[6])
        y_coord = int(parts[8])

        image_records.append({
            "path": image_path,
            "patient": patient,
            "node": node,
            "x_coord": x_coord,
            "y_coord": y_coord
        })

    except Exception:

        continue

images_df = pd.DataFrame(image_records)

print(f"Images successfully parsed: {len(images_df)}")

# ============================================================
# MATCH WITH METADATA
# ============================================================

merged = images_df.merge(
    metadata,
    on=["patient", "node", "x_coord", "y_coord"],
    how="inner"
)

print(f"Images matched with metadata: {len(merged)}")

if len(merged) == 0:

    raise RuntimeError(
        "Images were found, but none matched metadata."
    )

# ============================================================
# SELECT TUMOUR AND NON-TUMOUR IMAGES
# ============================================================

tumour_images = merged[merged["tumor"] == 1]

non_tumour_images = merged[merged["tumor"] == 0]

print(f"Tumour images available: {len(tumour_images)}")
print(f"Non-tumour images available: {len(non_tumour_images)}")

if len(tumour_images) < 3 or len(non_tumour_images) < 3:

    raise RuntimeError(
        "Not enough tumour/non-tumour images available."
    )

# ============================================================
# RANDOM SAMPLE
# ============================================================

tumour_sample = tumour_images.sample(
    n=3,
    random_state=42
)

non_tumour_sample = non_tumour_images.sample(
    n=3,
    random_state=42
)

samples = pd.concat([
    tumour_sample,
    non_tumour_sample
])

# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

# ------------------------------------------------------------
# Tumour images
# ------------------------------------------------------------

for i, (_, row) in enumerate(tumour_sample.iterrows()):

    image = Image.open(row["path"])

    axes[0, i].imshow(image)

    axes[0, i].set_title(
        f"Tumour\nCentre {row['center']}"
    )

    axes[0, i].axis("off")

# ------------------------------------------------------------
# Non-tumour images
# ------------------------------------------------------------

for i, (_, row) in enumerate(non_tumour_sample.iterrows()):

    image = Image.open(row["path"])

    axes[1, i].imshow(image)

    axes[1, i].set_title(
        f"Non-Tumour\nCentre {row['center']}"
    )

    axes[1, i].axis("off")

# ============================================================
# SAVE FIGURE
# ============================================================

plt.suptitle(
    "CAMELYON17-WILDS Histopathology Patch Samples",
    fontsize=16
)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "fig_3_2_camelyon17_histopathology_samples.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("==============================================")
print("FIGURE CREATED SUCCESSFULLY")
print("==============================================")
print(output_path)