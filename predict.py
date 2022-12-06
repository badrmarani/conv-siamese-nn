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

checkpoint = torch.load(model_path, map_location=device)

model = ConvSiameseNet().to(device)
model.load_state_dict(checkpoint["model"], strict=False)

loss_history = checkpoint["loss_history"]
l_fit, l_val = loss_history["fit"], loss_history["val"]

acc_history = checkpoint["acc_history"]
l_fit, l_val = acc_history["fit"], acc_history["val"]

fig, axs = plt.subplots()