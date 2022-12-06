from model import ConvSiameseNet
from utils import weight_init, train_test_split, LogoDataset
from torch import nn
from sklearn.metrics import accuracy_score
import torch
import json
import os

with open("args.json", "r") as f:
    args = json.load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = "./checkpoints"
if os.path.exists(checkpoints):
    os.makedirs(checkpoints)

# DATASET
dataset = LogoDataset(
    folder_dataset=args["folder_dataset"],
    transform=True)


fit, val = train_test_split(
    dataset,
    train_size=args["split_sizes"][0])


model = ConvSiameseNet()
model = nn.DataParallel(model)
model = model.to(device)
model.apply(weight_init)


if args["warmup_start"]:
    model = torch.load(checkpoints+".pkl", map_location=device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=args["lr"], weight_decay=.1)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
    optimizer, lr_lambda=0.99)

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
            acc = accuracy_score(yhat.item() > 0.5, y.item())
            print(
                f"epoch {epoch}/{args['num_epochs']} ",
                f"batch {epoch}/{len(fit)} ",
                f"loss {loss.item():.4f} ",
                f"accuracy {acc:.4f}"
            )

            loss_history["fit"].append(loss.item())
            acc_history["fit"].append(acc.item())

    model.eval()
    for batch, (x1, x2, y) in enumerate(val):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        yhat = model(x1, x2)
        loss = loss_fn(yhat, y)
        acc = accuracy_score(yhat.item() > 0.5, y.item())
        print(
            f"epoch {epoch}/{args['num_epochs']} ",
            f"batch {batch}/{len(val)} ",
            f"loss {loss.item():.4f} ",
            f"accuracy {acc:.4f}"
        )

        loss_history["val"].append(loss.item())
        acc_history["val"].append(acc.item())

    torch.save(
        {
            "loss_history": loss_history,
            "acc_history": acc_history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"../checkpoints/checkpoint_{epoch}.pkl",
    )

# # --- DEBUG ---
# x = torch.randn(size=(2, 1, 250, 250), dtype=dtype).to(device)
# # print(model(x,x).size())


# # print(model(x,x).size())
# # print(model.features[0].bias)
