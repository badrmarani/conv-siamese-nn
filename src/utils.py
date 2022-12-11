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
import yaml
import math
import os

num_workers = os.cpu_count()

dtype = torch.float
with open("args.yml", "r") as f:
    args = yaml.safe_load(f)


def plot_images(x1, x2, distance, ytrue, epoch=0):
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(
            np.concatenate([x1[i, ...], x2[i, ...]], axis=1),
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


def train_test_split(dataset, train_size, shuffle=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(math.floor(train_size * len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    train_sampler, test_sampler = (
        SubsetRandomSampler(indices[:split]),
        SubsetRandomSampler(indices[split:]),
    )

    return (
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0, sampler=train_sampler
        ),
        DataLoader(
            dataset, batch_size=args["batch_size"], num_workers=0, sampler=test_sampler
        ),
    )


def get_mean_std_dataset(dataset):
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        # num_workers=num_workers,
    )

    images, _ = next(iter(loader))
    print(images)
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
    return mean, std


class TripletLossTrainer:
    def train(epoch, model, loss_fn, optim, dataloader, device):
        model.train()
        train_loss = 0.0
        for batch, (x, labels) in enumerate(dataloader, start=1):
            x, labels = x.to(device), labels.to(device)
            embeddings = model(x)
            loss = loss_fn(embeddings, labels)
            train_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(
                "<TRAIN> EPOCH: {:d} BATCH: {:d}/{} LOSS: {:>7f}".format(
                    epoch, batch * len(x), len(dataloader.dataset), loss.item()
                )
            )
        train_loss /= len(dataloader)
        print("<TRAIN ERROR> EPOCH: {:d} AVG LOSS: {:>7f}".format(epoch, train_loss))
        return train_loss

    def test(epoch, model, loss_fn, dataloader, device):
        model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch, (x, labels) in enumerate(dataloader, start=1):
                x, labels = x.to(device), labels.to(device)
                embeddings = model(x)
                loss += loss_fn(embeddings, labels).item()

        loss /= len(dataloader)
        print("<TEST ERROR> EPOCH: {:d} AVG LOSS: {:>7f}".format(epoch, loss))
        return loss
