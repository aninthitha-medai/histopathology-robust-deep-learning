 from wilds import get_dataset

# Path where CAMELYON17-WILDS is stored
dataset = get_dataset(
    dataset="camelyon17",
    root_dir=r"C:\Users\Admin\histopathology_project\data"
)

print("CAMELYON17-WILDS loaded successfully!")
print("Number of samples:", len(dataset))
print("Dataset:", dataset)
mport pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
from torchvision import transforms


class CamelyonDataset(Dataset):

    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            BytesIO(row["image"]["bytes"])
        )

        label = int(row["label"])

        image = self.transform(image)

        return image, label


if __name__ == "__main__":

    dataset = CamelyonDataset(
        "./data/camelyon17_wilds/data/train-00000-of-00014.parquet"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    images, labels = next(iter(loader))

    print("Dataset samples:", len(dataset))
    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Batch labels:", labels.tolist())