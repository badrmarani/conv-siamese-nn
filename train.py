from torch.utils.data import DataLoader
from model import ConvSiameseNet
from utils import weight_init, train_test_split, LogoDataset
from torch import nn
from sklearn.metrics import accuracy_score
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
    train_size=args["split_sizes"][0])

model = ConvSiameseNet()
# model = nn.DataParallel(model)
model = model.to(device)
model.apply(weight_init)


if args["warmup_start"]:
    model = torch.load(checkpoints+".pkl", map_location=device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=.1)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=lambda epoch: 0.99)

loss_history = {"fit": [], "val": []}
acc_history = {"fit": [], "val": []}
for epoch in range(args["num_epochs"]):

    model.train()
    for batch, (x1, x2, y) in enumerate(fit):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()

        yhat = model(x1, x2)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            acc = accuracy_score(yhat.cpu().numpy() > .5, y.cpu().numpy())
            if not batch%10:
                print(
                    "TRAIN ",
                    f"epoch {epoch+1}/{args['num_epochs']} ",
                    f"batch {batch+1}/{len(fit)} ",
                    f"loss {loss.item():.4f} ",
                    f"accuracy {acc:.4f}"
                )

            loss_history["fit"].append(loss.item())
            acc_history["fit"].append(acc.item())

    # model.eval()
    # with torch.no_grad():
    #     for batch, (x1, x2, y) in enumerate(val):
    #         x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    #         yhat = model(x1, x2)
    #         loss = loss_fn(yhat, y)
    #         acc = accuracy_score(yhat.cpu().numpy() > 0.5, y.cpu().numpy())
            
    #         if not batch%10:
    #             print(
    #                 "EVAL ",
    #                 f"epoch {epoch+1}/{args['num_epochs']} ",
    #                 f"batch {batch+1}/{len(val)} ",
    #                 f"loss {loss.item():.4f} ",
    #                 f"accuracy {acc:.4f}"
    #             )

    #         loss_history["val"].append(loss.item())
    #         acc_history["val"].append(acc.item())

    with torch.no_grad():
        nsamples = 5
        test = DataLoader(dataset, batch_size=nsamples, num_workers=8)
        data = iter(test)
        x1, x2, y = next(data)
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        yhat = model(x1, x2)
        predictions = yhat.cpu().numpy() > 0.5

        fig, axs = plt.subplots(2, nsamples)
        for i in range(2):
            for j in range(nsamples):
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_aspect("equal")

                xx1 = x1[j].cpu().numpy().transpose(1, 2, 0)
                a = y[j].item()
                b = predictions[j].item()
                p = yhat[j].float().item()

                if i:
                    axs[i, j].imshow(xx1, cmap="gray")
                if not i:
                    axs[i, j].set_title(
                        f"G{a:.0f} P{b:.0f} - {'correct' if a==b else 'false'} - {p:.3f}")
                    xx2 = x2[j].cpu().numpy().transpose(1, 2, 0)
                    axs[i, j].imshow(xx2, cmap="gray")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"results/{epoch}_model_predictions.jpg", dpi=600)
        plt.close()

    torch.save(
        {
            "loss_history": loss_history,
            "acc_history": acc_history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"./checkpoints/checkpoint_{epoch}.pkl",
    )
