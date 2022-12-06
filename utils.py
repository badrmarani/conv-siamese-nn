from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
import random
import json
import math

dtype = torch.float
with open("args.json", "r") as f:
    args = json.load(f)


def weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 1e-2)

        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 2e-1)

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.bias.data.normal_(0.5, 1e-2)


def train_test_split(dataset, train_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(math.floor(train_size*len(dataset)))
    train_sampler, test_sampler = (
        SubsetRandomSampler(indices[:split]),
        SubsetRandomSampler(indices[split:]),
    )

    return (
        DataLoader(
            dataset, batch_size=args["batch_size_per_ep"], sampler=train_sampler, num_workers=8),
        DataLoader(
            dataset, batch_size=args["batch_size_per_ev"], sampler=test_sampler, num_workers=8),
    )

class LogoDataset(Dataset):
    def __init__(
        self,
        folder_dataset: Path,
        transform=None,
    ) -> None:
        self.dataset = ImageFolder(root=folder_dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset.imgs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x1 = random.choice(self.dataset.imgs)

        is_same = random.randint(0, 1)
        if is_same:
            while True:
                x2 = random.choice(self.dataset.imgs)
                if x1[1] == x2[1]:
                    break
        else:
            while True:
                x2 = random.choice(self.dataset.imgs)
                if x1[1] != x2[1]:
                    break

        im1 = Image.open(x1[0]).convert("RGBA").convert("L")
        im2 = Image.open(x2[0]).convert("RGBA").convert("L")

        if self.transform is not None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((200, 200)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0,), (1.0,)),
                ]
            )

            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return im1, im2, torch.tensor([int(x1[1] == x2[1])], dtype=dtype)
