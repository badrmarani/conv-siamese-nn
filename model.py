import torch
from torch import nn
from torchvision.models import alexnet
torch.cuda.empty_cache()

class ConvSiameseNet(nn.Module):
    def __init__(self) -> None:
        super(ConvSiameseNet, self).__init__()

        self.encoder = alexnet(progress=False)
        self.encoder.features[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(.3),
            nn.Linear(1, 202),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(202, 1),
            nn.Sigmoid(),
        )

        # self.encoder.apply(self.weight_init)
        # self.fc.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0.0, 1e-2)

        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 2e-1)

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(m.bias, 0.5, 1e-2)

    def forward(self, x1, x2):
        num_batchs = x1.size(0)
        x1 = self.encoder(x1).view((num_batchs,-1))
        x2 = self.encoder(x2).view((num_batchs,-1))

        feature_vector = torch.nn.functional.pairwise_distance(x1, x2, keepdim=True)
        # print(feature_vector.size())
        return self.fc(feature_vector)
