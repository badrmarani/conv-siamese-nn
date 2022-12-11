from src.losses import OnlineTripletLossMining
from src.models import DummyNet
from src.utils import get_mean_std_dataset, TripletLossTrainer, train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import yaml
import os

torch.cuda.empty_cache()
num_workers = os.cpu_count()

with open("args.yml", "r") as f:
    args = yaml.safe_load(f)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = "./checkpoints"
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)
if not os.path.exists("results"):
    os.makedirs("results/")

# all_images = ImageFolder(args["dataset_dirname"])
# mean, std = get_mean_std_dataset(all_images)
# print(mean, std)
tr = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean, std),
    ]
)


all_images = ImageFolder(args["dataset_dirname"], tr)
# dataloader = DataLoader(all_images, batch_size=10, shuffle=True)

trainloader, testloader = train_test_split(all_images, 0.8)
# trainloader, testloader = random_split(all_images, [0.8, 0.2])

num_epochs = 10

model = DummyNet().to(device)
loss_fn = OnlineTripletLossMining(bias=0.2)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)


history = {"train": [], "test": []}
for epoch in range(1, num_epochs + 1):
    train_loss = TripletLossTrainer.train(
        epoch, model, loss_fn, optim, trainloader, device
    )
    test_loss = TripletLossTrainer.test(epoch, model, loss_fn, testloader, device)

    history["train"].append(train_loss)
    history["test"].append(test_loss)

    torch.save(
        {
            "loss_history": history,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        },
        f="checkpoints/checkpoint_{}.pkl".format(epoch),
    )


# checkpoint = torch.load("checkpoints/checkpoint_2.pkl", map_location=device)
# history = checkpoint["loss_history"]
plt.figure()
plt.plot(list(range(1, num_epochs + 1)), history["train"], label="Training loss")
plt.plot(list(range(1, num_epochs + 1)), history["test"], label="Testing loss")
plt.title("Online Triplet Loss (Batch All Strategy)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.locator_params(axis="x", integer=True, tight=True)
plt.savefig("history_dummynet_tl.jpg", dpi=300)
plt.close()
