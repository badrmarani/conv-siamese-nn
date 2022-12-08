from utils import LogoDataset
from torch.utils.data import DataLoader
from model import ConvSiameseNet
from pathlib import Path
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
    # model_path = args["model_path"]

    files = Path("checkpoints/").iterdir()
    model_path = max(files, key=os.path.getctime)

if not os.path.exists("results"):
    os.makedirs("results/")

checkpoint = torch.load(model_path, map_location=device)

model = ConvSiameseNet().to(device)
model.load_state_dict(checkpoint["model"], strict=False)

loss_history = checkpoint["loss_history"]
loss_fit, loss_val = loss_history["fit"], loss_history["val"]

# acc_history = checkpoint["acc_history"]
# acc_fit, acc_val = acc_history["fit"], acc_history["val"]

fig, axs = plt.subplots(nrows=1, ncols=1)
axs.plot(list(range(len(loss_fit))), loss_fit, label="fit")
axs.plot(list(range(len(loss_fit))), loss_val, label="val")
axs.grid()
axs.legend()
axs.set_title("Loss")
# axs[1].plot(list(range(len(acc_fit))), acc_fit, label="fit")
# axs[1].plot(list(range(len(acc_fit))), acc_fit, label="val")
# axs[1].grid()
# axs[1].legend()
# axs[1].set_title("Accuracy")
plt.savefig(f"results/model_history_{len(loss_fit)}.jpg", dpi=600)
plt.close()


# ROC Curve
