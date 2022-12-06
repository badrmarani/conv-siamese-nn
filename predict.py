from model import ConvSiameseNet
import matplotlib.pyplot as plt
import torch
import json
import os

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

axs[0].plot(list(range(len(loss_fit))), loss_fit, label="training loss")
axs[0].plot(list(range(len(loss_fit))), loss_val, label="validation loss")
axs[1].plot(list(range(len(acc_fit))), acc_fit, label="training acc")
axs[1].plot(list(range(len(acc_fit))), acc_fit, label="validation acc")
plt.legend()
plt.grid()
plt.savefig(f"results/model_history_{len(loss_fit)}.jpg", dpi=600)