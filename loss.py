import torch
from torch import nn

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OnlineTripletLoss(nn.Module):
    def __init__(self, bias: float = 0.0, reduce: str = "sum") -> None:
        super(OnlineTripletLoss, self).__init__()
        if isinstance(bias, torch.Tensor):
            self.bias = bias
        else:
            self.bias = torch.tensor([bias], dtype=dtype).to(device)

        self.reduce = reduce
        self.eps = 1e-8

    def _euclidean_distance(self, x: torch.Tensor, eps: float = 1e-6):
        # x <- size(batch_size, )
        x2 = torch.mm(x, x.T)       # size(num_batchs, num_batchs)
        x2_norm = torch.diag(x2)    # size(num_batchs,)

        # size(num_batchs, num_batchs)
        distance = x2_norm.unsqueeze(0) + x2_norm.unsqueeze(1) - 2*x2
        distance = torch.nn.functional.relu(distance)

        mask = (distance == 0.0).float()
        distance += mask*eps
        distance = torch.sqrt(distance)
        distance *= (1.0-mask)
        return distance

    def _get_valid_triplets(self, labels) -> torch.Tensor:
        # generating the triplets using the online strategy.
        # labels: batch of labels -> size(num_batchs,)
        # a triplet is valid if:
        # 1) labels[i]==labels[j] and labels[i]!=labels[k]
        # 2) i!=j and j!=k

        same_indices = torch.eye(labels.size(0), dtype=dtype)
        not_same_indices = torch.logical_not(same_indices)

        # indices where labels're distinct
        ij = not_same_indices.unsqueeze(2)
        ik = not_same_indices.unsqueeze(1)
        jk = not_same_indices.unsqueeze(0)
        distinct_indices = torch.logical_and(jk, torch.logical_and(
            ij, ik))  # size(num_batchs, num_batchs, num_batchs)

        same_labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        ij = same_labels.unsqueeze(2)
        ik = same_labels.unsqueeze(1)
        # size(num_batchs, num_batchs, num_batchs)
        valid_indices = torch.logical_and(ij, torch.logical_not(ik))

        return torch.logical_and(distinct_indices, valid_indices)

    def forward(self, embeddings, labels):
        # embeddings <- size(num_batchs, embed_size)

        distance = self._euclidean_distance(embeddings)
        ap_dist = distance.unsqueeze(2)
        an_dist = distance.unsqueeze(1)
        loss = ap_dist - an_dist + self.bias

        mask = self._get_valid_triplets(labels)
        loss *= mask
        loss = torch.nn.functional.relu(loss)

        # average only the remaining positive values
        num_positives = (loss > self.eps).float().sum()
        loss = loss.sum() / num_positives
        return loss


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin: int = 1, reduce: str = "mean") -> None:
        super(OnlineContrastiveLoss, self).__init__()
        if isinstance(margin, torch.Tensor):
            self.margin = margin
        else:
            self.margin = torch.tensor([margin], dtype=dtype).to(device)

        self.reduce = reduce

    def _distance(self, x):
        pass

    def _get_valid_batch(self, labels):
        # labels: batch of labels -> size(num_batchs,)

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        return mask

    def forward(self, embeddings, labels):
        raise NotImplementedError


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: int = 1, reduce: str = "mean") -> None:
        super(ContrastiveLoss, self).__init__()
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
