from utils import LogoDataset
from torch.utils.data import DataLoader
from model import ConvSiameseNet
import matplotlib.pyplot as plt
import matplotlib
import torch
import json
import os

font = {
    "weight": "bold",
    "size": 5
}

matplotlib.rc("font", **font)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("args.json", "r") as f:
    args = json.load(f)

if os.path.exists("checkpoints"):
    model_path = args["model_path"]
if not os.path.exists("results"):
    os.makedirs("results/")

checkpoint = torch.load(model_path, map_location=device)

model = ConvSiameseNet().to(device)
model.load_state_dict(checkpoint["model"], strict=False)

loss_history = checkpoint["loss_history"]
loss_fit, loss_val = loss_history["fit"], loss_history["val"]

acc_history = checkpoint["acc_history"]
acc_fit, acc_val = acc_history["fit"], acc_history["val"]

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(list(range(len(loss_fit))), loss_fit, label="fit")
axs[0].plot(list(range(len(loss_fit))), loss_val, label="val")
axs[0].grid()
axs[0].legend()
axs[0].set_title("Loss")
axs[1].plot(list(range(len(acc_fit))), acc_fit, label="fit")
axs[1].plot(list(range(len(acc_fit))), acc_fit, label="val")
axs[1].grid()
axs[1].legend()
axs[1].set_title("Accuracy")
plt.savefig(f"results/model_history_{len(loss_fit)}.jpg", dpi=600)
plt.close()

nsamples = 5
with torch.no_grad():
    test = DataLoader(
        dataset=LogoDataset(args["folder_dataset"], transform=True),
        batch_size=nsamples,
    )

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
            axs[i, j].set_aspect('equal')

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
    plt.savefig(f"results/model_pred_{nsamples}.jpg", dpi=600)