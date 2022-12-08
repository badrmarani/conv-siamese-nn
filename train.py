from torch.utils.data import DataLoader
from model import ConvSiameseNet
from utils import plot_images, train, test, train_test_split, LogoDataset
from torch import nn
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import torch
import json
import os

torch.cuda.empty_cache()

font = {
    "weight": "bold",
    "size": 5
}

matplotlib.rc("font", **font)

with open("args.json", "r") as f:
    args = json.load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = "./checkpoints"
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)
if not os.path.exists("results"):
    os.makedirs("results/")

# DATASET
dataset = LogoDataset(
    folder_dataset=args["folder_dataset"],
    transform=True)

fit, val = train_test_split(
    dataset,
    train_size=args["split_sizes"][0],
    shuffle=True)

model = ConvSiameseNet()
# model = nn.DataParallel(model)
model = model.to(device)

if args["warmup_start"]:
    model = torch.load(checkpoints+".pkl", map_location=device)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=.1)

# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
#     optimizer, lr_lambda=lambda epoch: 0.99)

loss_history = {"fit": [], "val": []}
acc_history = {"fit": [], "val": []}


# print("global")
# print(len(fit.dataset), len(fit), len(iter(fit)))
# print(len(val.dataset), len(val), len(iter(fit)))

# print(len(fit.dataset))

show_sample = True

for epoch in range(1, args["num_epochs"]+1):
    train_loss, train_acc = train(epoch, model, fit, optimizer, device)
    test_loss, test_acc = test(model, val, device)

    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=True,
            num_workers=0)

        data_iter = iter(sample_loader)
        x1, x2, y = next(data_iter)
        yhat = model(x1,x2)
        yhat = yhat.detach().numpy()
        x1 = x1.numpy().transpose([0, 2, 3, 1])
        x2 = x2.numpy().transpose([0, 2, 3, 1])
        plot_images(x1, x2, yhat, y, epoch)

    loss_history["fit"].append(train_loss)
    loss_history["val"].append(test_loss)

    loss_history["fit"].append(train_acc)
    loss_history["val"].append(test_acc)

    torch.save(
        {
            "loss_history": loss_history,
            "acc_history": acc_history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"./checkpoints/checkpoint_{epoch}.pkl",
    )
