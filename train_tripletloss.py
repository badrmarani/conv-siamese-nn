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
# if not os.path.exists("results"):
#     os.makedirs("results/")

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

num_epochs = 500

model = DummyNet()
model = torch.nn.DataParallel(model)
model = model.to(device)
loss_fn = OnlineTripletLossMining(bias=0.2)
optim = torch.optim.Adam(model.parameters(), lr=0.005)


history = {"train": [], "test": []}
for epoch in range(1, num_epochs + 1):
    train_loss = TripletLossTrainer.train(
        epoch, model, loss_fn, optim, trainloader, device
    )
    test_loss = TripletLossTrainer.test(epoch, model, loss_fn, testloader, device)

    history["train"].append(train_loss)
    history["test"].append(test_loss)

    if not epoch%10:
        torch.save(
            {
                "loss_history": history,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            },
            f="checkpoints/checkpoint_{}.pkl".format(epoch),
        )

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,5))
# axs[0].set_title("All epochs")
axs[0].plot(list(range(1, num_epochs + 1)), history["train"], label="Training loss")
axs[0].plot(list(range(1, num_epochs + 1)), history["test"], label="Testing loss")
axs[0].grid(True)
axs[0].legend()

axs[1].set_title("Last 80 epochs")
axs[1].plot(list(range(1, num_epochs + 1))[80:], history["train"][80:], label="Training loss")
axs[1].plot(list(range(1, num_epochs + 1))[80:], history["test"][80:], label="Testing loss")
axs[1].grid(True)
axs[1].legend()

fig.suptitle("Online Triplet Loss (Batch All Strategy)")
fig.supxlabel("Loss")
fig.supylabel("Loss")
plt.tight_layout()
plt.locator_params(axis="x", integer=True, tight=True)
plt.savefig("history_{}_{}.jpg".format(model.__class__.__name__, num_epochs), dpi=300)
# plt.close()
