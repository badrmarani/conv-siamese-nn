import torch
from torch import nn
from torchvision.models import vgg16_bn

class ConvSiameseNet(nn.Module):
    def __init__(self) -> None:
        super(ConvSiameseNet, self).__init__()

        self.encoder = vgg16_bn(progress=False)
        self.encoder.features[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        num_batchs = x1.size(0)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        feature_vector = torch.abs(x1-x2)
        return self.fc(feature_vector)
