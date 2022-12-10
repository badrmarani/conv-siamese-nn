import torch
import yaml
from torch import nn
from torchvision import models

torch.cuda.empty_cache()

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2Pool(nn.Module):
    """Taken from: https://discuss.pytorch.org/t/how-do-i-create-an-l2-pooling-2d-layer/105562/5"""

    def __init__(self, *args, **kwargs) -> None:
        super(L2Pool, self).__init__()
        kwargs["divisor_override"] = 1
        self.pool = nn.AvgPool2d(*args, **kwargs)

    def forward(self, x):
        return torch.sqrt(self.pool(x**2))


class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            b33_rdc,
            b33_out,
            b55_rdc,
            b55_out,
            out_pool=None,
            mode=None,
            stride3=1,
            stride5=1,
    ) -> None:
        super(InceptionBlock, self).__init__()

        self.branchs = []

        if out_channels is not None:
            self.branchs.append(
                self._conv_block(
                    in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0
                )
            )

        self.branchs.append(
            nn.Sequential(
                self._conv_block(
                    in_channels, b33_rdc, kernel_size=(1, 1), stride=1, padding=0
                ),
                self._conv_block(
                    b33_rdc, b33_out, kernel_size=3, stride=stride3, padding=1
                ),
            )
        )
        self.branchs.append(
            nn.Sequential(
                self._conv_block(
                    in_channels, b55_rdc, kernel_size=1, stride=1, padding=0
                ),
                self._conv_block(
                    b55_rdc, b55_out, kernel_size=5, stride=stride5, padding=2
                ),
            )
        )

        if mode.lower() == "l2":
            seq = nn.Sequential(
                L2Pool(3, 1, 1),
                self._conv_block(
                    in_channels, out_pool, kernel_size=1, stride=1, padding=0
                ),
            )
        elif mode.lower() == "max":
            seq = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                self._conv_block(
                    in_channels, out_pool, kernel_size=1, stride=1, padding=0
                ),
            )
        else:
            seq = nn.Sequential(
                nn.MaxPool2d(3, 2, 1),
            )
        self.branchs.append(seq)

    def _conv_block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = []
        for i, branch in enumerate(self.branchs):
            tmp.append(branch(x))
            # print(i, branch(x).size()) # test

        out = torch.cat(tuple(tmp), dim=1)
        return out


class InceptionNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
    ) -> None:
        super(InceptionNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, (7, 7), stride=2, padding=3),
            nn.MaxPool2d((3, 3), stride=2, padding=1),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 192, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d((3, 3), stride=2, padding=1),
            InceptionBlock(192, 64, 96, 128, 16, 32, 32, mode="max"),
            InceptionBlock(256, 64, 96, 128, 32, 64, 64, mode="l2"),
            InceptionBlock(
                320, None, 128, 256, 32, 64, mode="none", stride3=2, stride5=2
            ),
            InceptionBlock(640, 256, 96, 192, 32, 64, 128, mode="l2"),
            InceptionBlock(640, 224, 112, 224, 32, 64, 128, mode="l2"),
            InceptionBlock(640, 192, 128, 256, 32, 64, 128, mode="l2"),
            InceptionBlock(640, 160, 144, 288, 32, 64, 128, mode="l2"),
            InceptionBlock(
                640, None, 160, 256, 64, 128, mode="none", stride3=2, stride5=2
            ),
            InceptionBlock(1024, 384, 192, 384, 48, 128, 128, mode="l2"),
            InceptionBlock(1024, 384, 192, 384, 48, 128, 128, mode="max"),
            nn.AvgPool2d(7, 1),
            nn.Flatten(1),
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = torch.nn.functional.normalize(out, p=2.0, dim=1)
        return out  # size(num_batchs, 128)


class ConvSiameseNet(nn.Module):
    def __init__(
            self,
            pretrained=True,
            add_layer=False,
    ) -> None:
        super(ConvSiameseNet, self).__init__()
        self.add_layer = add_layer
        if pretrained:
            self.encoder = models.alexnet(
                progress=False, weights=models.AlexNet_Weights.IMAGENET1K_V1
            )
        else:
            self.encoder = models.alexnet(progress=False)

        self.encoder.features[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        if pretrained:
            # freeze
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # unfreeze
            for p in self.encoder.parameters():
                p.requires_grad = True

        for p in self.encoder.classifier[4:].parameters():
            p.requires_grad = True

        if add_layer:
            # if we use bceloss function
            num_last_layer = self.encoder.classifier[-1].out_features
            self.fc = nn.Linear(num_last_layer, 1)
            for p in self.fc.parameters():
                p.requires_grad = True

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
        out1 = self.encoder(x1).view((num_batchs, -1))
        out2 = self.encoder(x2).view((num_batchs, -1))
        if self.add_layer:
            # x1 <- (num_batchs, 1)
            diff = torch.abs(out1 - out2)
            out = self.fc(diff)
            return out
        else:
            # out1 <- (num_batchs, 1000)
            return out1, out2
