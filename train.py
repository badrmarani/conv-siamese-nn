from torch.utils.data import DataLoader
from torch import nn
from src.model import ConvSiameseNet
from src.loss import ContrastiveLoss, TripletLoss
from src.dataset import CLDataset, TLDataset
from src.utils import plot_images, train, test, train_test_split

# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import torch
import yaml
import os

torch.cuda.empty_cache()

font = {"weight": "bold", "size": 5}

matplotlib.rc("font", **font)

with open("args.yml", "r") as f:
    args = yaml.safe_load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = "./checkpoints"
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)
if not os.path.exists("results"):
    os.makedirs("results/")

# DATASET
dataset = CLDataset(folder_dataset=args["folder_dataset"])

fit, val = train_test_split(dataset, train_size=args["split_sizes"][0], shuffle=True)


model = ConvSiameseNet(pretrained=True, add_layer=False)
# model = nn.DataParallel(model)
model = model.to(device)

if args["warmup_start"]:
    model = torch.load(checkpoints + ".pkl", map_location=device)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=0.1)

# scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
#     optimizer, lr_lambda=lambda epoch: 0.99)

loss_history = {"fit": [], "val": []}
acc_history = {"fit": [], "val": []}


show_sample = True

loss_fn = ContrastiveLoss(margin=10.0, reduce="mean")

for epoch in range(1, args["num_epochs"] + 1):
    train_loss = train(epoch, model, loss_fn, fit, optimizer, device)
    test_loss = test(model, loss_fn, val, device)

    loss_history["fit"].append(train_loss)
    loss_history["val"].append(test_loss)

    if show_sample and not epoch % 5:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=True, num_workers=0
        )

        data_iter = iter(sample_loader)
        x1, x2, y = next(data_iter)
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        out1, out2 = model(x1, x2)
        distance = nn.functional.pairwise_distance(out1, out2).detach().cpu().numpy()
        x1 = x1.cpu().numpy().transpose([0, 2, 3, 1])
        x2 = x2.cpu().numpy().transpose([0, 2, 3, 1])
        plot_images(x1, x2, distance, y, epoch)

        torch.save(
            {
                "loss_history": loss_history,
                "acc_history": acc_history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"./checkpoints/checkpoint_{epoch}.pkl",
        )
