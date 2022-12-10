from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import ConvSiameseNet
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import torch
import yaml

import sys

font = {"weight": "bold", "size": 5}
matplotlib.rc("font", **font)

with open("./args.yml", "r", encoding="utf-8") as f:
    args = yaml.safe_load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_filename = args["checkpoint_filename"]
checkpoint = torch.load(checkpoint_filename, map_location=device)

model = ConvSiameseNet().to(device)
model.load_state_dict(checkpoint["model"], strict=False)

new_size = (200,200)
tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(new_size),
])

x1_filename = "dataset/1/cute cat.jpg"
x1 = Image.open(x1_filename)
temp_x1 = x1.convert("L")
temp_x1 = tr(temp_x1).unsqueeze(0)

dataset = ImageFolder(args["dataset_dirname"])
num_folders = len(dataset.classes)
total_num_imgs = len(dataset.targets)

arr = torch.tensor(dataset.targets)
memo = {index: None for index in range(num_folders)}

for x in memo.keys():
    indice = random.choice(torch.where(arr==x)[0].cpu().numpy())
    x2 = dataset.imgs[indice][0]
    x2 = Image.open(x2)
    temp_x2 = x2.convert("L")
    temp_x2 = tr(temp_x2).unsqueeze(0)
    out1, out2 = model(temp_x1, temp_x2)
    distance = torch.norm(out1-out2)

    memo[x] = (indice, distance)

print()

best = min(memo.values(), key=lambda item: item[1])
best_label_indice = dataset.imgs[best[0]][1]
best_label = args["classes"][best_label_indice]


h, w = (num_folders+1)//2, 2
fig, axes = plt.subplots(h, w)
for i, (ax, m) in enumerate(zip(axes.flat, memo.values())):
    x2 = plt.imread(dataset.imgs[m[0]][0])
    ax.imshow(
        np.concatenate([x1, x2], axis=1),
    )
    xlabel = f"distance {m[1]:.5f} - {args['classes'][dataset.imgs[m[0]][1]]}"
    if i == best_label_indice:
        ax.set_title(xlabel, color="red")
    else:
        ax.set_title(xlabel, color="black")
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axes.flat:
    if not bool(ax.has_data()):
        fig.delaxes(ax)
fig.savefig("out_.jpg", dpi=300)