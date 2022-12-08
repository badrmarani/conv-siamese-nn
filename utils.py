from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import json
import math

dtype = torch.float
with open("args.json", "r") as f:
    args = json.load(f)

def plot_images(x1, x2, distance, ytrue, epoch=0):
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            np.concatenate([x1[i,...],x2[i,...]], axis=1),
            interpolation="spline16",
            cmap="gray",
        )

        xlabel = f"ground truth {ytrue[i]}; distance {distance[i]}"
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=300)
    plt.close()

def predict(epoch, model, device, nsamples):
    pass

def train(epoch, model, loss_fn, data_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0.0

    for batch, (x1, x2, y) in enumerate(data_loader, 0):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        out1, out2 = model(x1, x2)
        distance = nn.functional.pairwise_distance(out1, out2).to(device)

        optimizer.zero_grad()

        loss = loss_fn(distance, y).unsqueeze(0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()

    train_loss /= len(data_loader)

    print(
        f"train epoch {epoch}/{args['num_epochs']} ",
        f"loss {train_loss:.5f} ",
    )

    return train_loss

def test(model, loss_fn, data_loader, device):
    model.eval()
    test_loss = 0.0 
    for batch, (x1, x2, y) in enumerate(data_loader, 0):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            

        with torch.no_grad():
            out1, out2 = model(x1, x2)
            distance = nn.functional.pairwise_distance(out1, out2).to(device)
            loss = loss_fn(distance, y).unsqueeze(0)
            test_loss += loss.item()

    test_loss /= len(data_loader)

    print(
        f"eval ",
        f"loss {test_loss:.5f} ",
    )
    
    return test_loss

def train_test_split(dataset, train_size, shuffle=False):
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(math.floor(train_size*len(dataset)))

    # if shuffle:
    #     np.random.shuffle(indices)

    # train_sampler, test_sampler = (
    #     SubsetRandomSampler(indices[:split]),
    #     SubsetRandomSampler(indices[split:]),
    # )

    return (
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0),
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0),
    )

def augment(dataset):
    return transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0)),
        transforms.Resize((200,200)),
    ])(dataset)

class LogoDataset(Dataset):
    def __init__(
            self,
            folder_dataset: Path,
    ) -> None:
        self.dataset = ImageFolder(root=folder_dataset)
        self.num_labels = len(self.dataset.classes)

        self.transform = transforms.Compose([
            transforms.RandomApply(nn.ModuleList([
                # transforms.RandomHorizontalFlip(.5),
                # transforms.RandomVerticalFlip(.5),
                transforms.RandomInvert(.5),
                transforms.RandomRotation(degrees=90)
                # transforms.GaussianBlur(kernel_size=(3,3), sigma=.1)
            ]), p=.5),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0)),
            transforms.Resize((200,200)),
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

        rnd_index_1  = random.randint(0, self.memo[im_class].size(0)-1)
        x1 = self.dataset.imgs[
            self.memo[im_class][rnd_index_1]
        ][0]

        if not index%2:
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