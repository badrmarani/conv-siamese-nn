import torch
from torch import nn

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TripletLoss(nn.Module):
    def __init__(self, bias: float = 0.0) -> None:
        super(TripletLoss, self).__init__()
        self.bias = bias

    def forward(self, xa, xp, xn):
        raise NotImplemented


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: int = 1, reduce: str = "mean",) -> None:
        super().__init__()
        if isinstance(margin, torch.Tensor):
            self.margin = margin
        else:
            self.margin = torch.tensor([margin], dtype=dtype).to(device)

        self.reduce = reduce

    def forward(self, distance, y):
        out = (1 - y) * distance.pow(2) + y * torch.clamp(
            self.margin - distance, min=0.0
        ).pow(2)

        if self.reduce.lower() == "mean":
            out = out.mean()
        else:
            raise NotImplemented
        return 0.5 * out
