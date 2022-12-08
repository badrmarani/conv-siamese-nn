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

def plot_images(x1, x2, yhat, ytrue, epoch=0):
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            np.concatenate([x1[i,...],x2[i,...]], axis=1),
            interpolation="spline16",
        )

        pred = np.where(yhat>.5, 1, 0)
        xlabel = f"ground truth {ytrue[i]}; prediction {pred[i]}\nprobability {yhat[i]}"

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f"results/sample_epoch_{epoch}.jpg", dpi=300)
    plt.close()

def predict(epoch, model, device, nsamples):
    pass

def train(epoch, model, data_loader, optimizer, device):
    model.train()
    loss_fn = nn.BCELoss()
    train_loss = 0.0
    correct = 0.0

    for batch, (x1, x2, y) in enumerate(data_loader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        yhat = model(x1, x2).squeeze(1)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.item()
            pred = torch.where(yhat > 0.5, 1, 0)
            correct += pred.eq(y.view_as(pred)).sum().item()

    train_loss = train_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    print(
        f"train epoch {epoch}/{args['num_epochs']} ",
        # f"batch {batch+1}/{len(data_loader.dataset)} ",
        f"loss {train_loss:.5f} ",
        f"acc {accuracy:.5f} ",
    )

    return loss.item(), accuracy

def test(model, data_loader, device):
    model.eval()
    loss_fn = nn.BCELoss()
    test_loss = 0.0 
    correct = 0.0

    with torch.no_grad():
        for batch, (x1, x2, y) in enumerate(data_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            yhat = model(x1, x2).squeeze(1)
            test_loss += loss_fn(yhat, y).item()
            pred = torch.where(yhat > 0.5, 1, 0)
            correct += pred.eq(y.view_as(pred)).sum().item()
    

    test_loss = test_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print(
        f"eval ",
        f"loss {test_loss:.5f} ",
        f"acc {accuracy:.5f}",
    )
    
    return test_loss, accuracy

def train_test_split(dataset, train_size, shuffle=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(math.floor(train_size*len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    train_sampler, test_sampler = (
        SubsetRandomSampler(indices[:split]),
        SubsetRandomSampler(indices[split:]),
    )

    return (
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0, sampler=train_sampler),
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0, sampler=test_sampler),
    )


class LogoDataset(Dataset):
    def __init__(
            self,
            folder_dataset: Path,
            transform=None,
    ) -> None:
        self.dataset = ImageFolder(root=folder_dataset)
        self.num_labels = len(self.dataset.classes)

        self.transform = transform
        self.tr = transforms.Compose([
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

        if self.transform:
            x1 = self.tr(x1).clone().float()
            x2 = self.tr(x2).clone().float()

        return x1, x2, y