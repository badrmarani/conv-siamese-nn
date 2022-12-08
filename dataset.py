from torch.utils.data import Dataset
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from PIL import Image
import torch
import random

dtype = torch.float

class LogoDataset(Dataset):
    def __init__(
            self,
            folder_dataset: Path,
            augment: bool,
    ) -> None:
        self.dataset = ImageFolder(root=folder_dataset)
        self.num_labels = len(self.dataset.classes)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomApply(nn.ModuleList([
                    transforms.RandomAffine(degrees=10, translate=(
                        0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize(50),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
                ]), p=.5),
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0)),
                transforms.Resize((200, 200)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0)),
                transforms.Resize((200, 200)),
            ])

        arr = torch.tensor(self.dataset.targets)
        self.memo = {index: None for index in range(self.num_labels)}
        for x in self.memo.keys():
            self.memo[x] = torch.where(arr == x)[0]

    def __len__(self) -> int:
        return len(self.dataset.imgs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        im_class = random.randint(0, self.num_labels-1)

        rnd_index_1 = random.randint(0, self.memo[im_class].size(0)-1)
        x1 = self.dataset.imgs[
            self.memo[im_class][rnd_index_1]
        ][0]

        if not index % 2:
            while True:
                rnd_index_2 = random.randint(0, self.memo[im_class].size(0)-1)
                if rnd_index_1 != rnd_index_2:
                    break

            x2 = self.dataset.imgs[
                self.memo[im_class][rnd_index_2]
            ][0]
            y = torch.tensor(1, dtype=dtype)
        else:
            im_class_2 = random.randint(0, self.num_labels-1)
            while im_class == im_class_2:
                im_class_2 = random.randint(0, self.num_labels-1)

            rnd_index = random.randint(0, self.memo[im_class_2].size(0)-1)
            x2 = self.dataset.imgs[
                self.memo[im_class_2][rnd_index]
            ][0]
            y = torch.tensor(0, dtype=dtype)

        x1 = Image.open(x1).convert("RGBA").convert("L")
        x2 = Image.open(x2).convert("RGBA").convert("L")

        x1 = self.transform(x1).clone().float()
        x2 = self.transform(x2).clone().float()
        return x1, x2, y
