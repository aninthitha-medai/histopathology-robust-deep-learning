import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CamelyonDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        image_root,
        dataset_type,
        transform=None
    ):

        self.metadata = pd.read_csv(metadata_csv)

        # -----------------------------
        # TRAINING SET
        # split 0 from centres 0, 3, 4
        # -----------------------------
        if dataset_type == "train":

            self.metadata = self.metadata[
                (self.metadata["split"] == 0) &
                (self.metadata["center"].isin([0, 3, 4]))
            ]

        # -----------------------------
        # VALIDATION SET
        # centre 1 validation data
        # + validation portions of 0,3,4        
        # -----------------------------
        elif dataset_type == "val":

            self.metadata = self.metadata[
                (
                    (self.metadata["center"] == 1)
                )
                |
                (
                    (self.metadata["split"] == 1) &
                    (self.metadata["center"].isin([0, 3, 4]))
                )
            ]

        # -----------------------------
        # TEST SET
        # completely unseen centre 2
        # -----------------------------
        elif dataset_type == "test":

            self.metadata = self.metadata[
                self.metadata["center"] == 2
            ]

        else:
            raise ValueError(
                "dataset_type must be 'train', 'val', or 'test'"
            )

        self.metadata = self.metadata.reset_index(drop=True)

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        patient = int(row["patient"])
        node = int(row["node"])
        x = int(row["x_coord"])
        y = int(row["y_coord"])

        folder = f"patient_{patient:03d}_node_{node}"

        filename = (
            f"patch_patient_{patient:03d}"
            f"_node_{node}"
            f"_x_{x}"
            f"_y_{y}.png"
        )

        image_path = os.path.join(
            self.image_root,
            folder,
            filename
        )

        image = Image.open(image_path).convert("RGB")

        label = int(row["tumor"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# VERIFY DATASET SPLITS
# ============================================================

if __name__ == "__main__":

    metadata_csv = (
        r"C:\Users\Admin\histopathology_project"
        r"\data\camelyon17_v1.0\metadata.csv"
    )

    image_root = (
        r"C:\Users\Admin\histopathology_project"
        r"\data\camelyon17_v1.0\patches"
    )

    train_dataset = CamelyonDataset(
        metadata_csv,
        image_root,
        "train"
    )

    val_dataset = CamelyonDataset(
        metadata_csv,
        image_root,
        "val"
    )

    test_dataset = CamelyonDataset(
        metadata_csv,
        image_root,
        "test"
    )

    print("=" * 60)
    print("CAMELYON17 DATASET SPLIT VERIFICATION")
    print("=" * 60)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    print("=" * 60)

    # Test one image from each dataset
    train_image, train_label = train_dataset[0]
    val_image, val_label = val_dataset[0]
    test_image, test_label = test_dataset[0]

    print("Training image:", train_image.size)
    print("Training label:", train_label)

    print("Validation image:", val_image.size)
    print("Validation label:", val_label)

    print("Test image:", test_image.size)
    print("Test label:", test_label)

    print("=" * 60)
    print("DATASET READY")
    print("=" * 60)