import os
import torch

import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_dataloaders_cifar(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    x_train_var = np.var(train_dataset.data / 255.0)

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, x_train_var

# Download the dataset @ https://huggingface.co/datasets/deboradum/GeoGuessr-coordinates
class GeoGuessrDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=["path", "lat", "lng"])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row["path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        lat, lng = float(row["lat"]), float(row["lng"])
        target = torch.tensor([lat, lng], dtype=torch.float32)

        return image, target


def compute_dataset_variance(data_loader):
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for images, _ in data_loader:
        images = images.view(-1)
        pixel_sum += images.sum().item()
        pixel_squared_sum += (images**2).sum().item()
        pixel_count += images.numel()

    mean = pixel_sum / pixel_count
    variance = (pixel_squared_sum / pixel_count) - (mean**2)
    return variance


def get_loaders_geoGuessr(
    batch_size,
    size,
    directory="/Users/personal/Desktop/geoGuessV2/createDataset/geoGuessrDataset",
):
    if size == "small":
        img_size = 512
        transform = transforms.Compose(
            [
                transforms.RandomCrop((img_size, img_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif size == "large":
        img_size = 900
        transform = transforms.Compose(
            [
                transforms.RandomCrop((img_size, img_size)),
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        raise NotImplementedError()

    datasets = {
        "train": GeoGuessrDataset(
            os.path.join(directory, "train.csv"), directory, transform
        ),
        "val": GeoGuessrDataset(
            os.path.join(directory, "val.csv"), directory, transform
        ),
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
        )
        for split in ["train", "val"]
    }

    # Hardcode x_train_var because it takes a while to compute.
    # x_train_var = compute_dataset_variance(loaders["train"])
    x_train_var = 0.05026310263180217

    return loaders["train"], loaders["val"], x_train_var


def get_dataloaders(dataset, size, batch_size):
    if dataset == "CIFAR10":
        return get_dataloaders_cifar(batch_size)
    elif dataset == "geoguessr":
        return get_loaders_geoGuessr(batch_size, size)
    else:
        raise NotImplementedError
