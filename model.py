import torch
from torch import nn
from torchvision import models 
torch.cuda.empty_cache()

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContrastiveLoss(nn.Module):
    def __init__(self,
        margin:int=1,
        reduce:str="mean",
    ) -> None:
        super().__init__()
        if isinstance(margin, torch.Tensor):
            self.margin = margin
        else:
            self.margin = torch.tensor([margin], dtype=dtype).to(device)

        self.reduce = reduce

    def forward(self, distance, y):
        out = (1 - y) * distance.pow(2) + y * torch.clamp(self.margin - distance, min=0.0).pow(2)
        
        if self.reduce.lower() == "mean":
            out = out.mean()
        else:
            raise NotImplemented
        return .5 * out


class ConvSiameseNet(nn.Module):
    def __init__(self,
        pretrained=True,
        add_layer=False,
    ) -> None:
        super(ConvSiameseNet, self).__init__()
        self.add_layer = add_layer
        if pretrained:
            self.encoder = models.alexnet(progress=False, weights=models.AlexNet_Weights.IMAGENET1K_V1)
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

        for p in self.encoder.classifier[-1].parameters():
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
        out1 = self.encoder(x1).view((num_batchs,-1))
        out2 = self.encoder(x2).view((num_batchs,-1))
        if self.add_layer:
            # x1 <- (num_batchs, 1)
            diff = torch.abs(out1-out2)
            out = self.fc(diff)
            return out
        else:
            # out1 <- (num_batchs, 1000)
            return out1, out2