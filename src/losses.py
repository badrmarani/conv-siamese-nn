import torch
from torch import nn

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OnlineTripletLossMining(nn.Module):
    """References:
    1) https://arxiv.org/abs/1503.03832
    2) https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905
    """

    def __init__(self, bias: float = 0.0, reduce: str = "sum") -> None:
        super(OnlineTripletLossMining, self).__init__()
        if isinstance(bias, torch.Tensor):
            self.bias = bias
        else:
            self.bias = torch.tensor([bias], dtype=dtype).to(device)

        self.reduce = reduce
        self.eps = 1

    def _euclidean_distance(self, x: torch.Tensor, eps: float = 1e-16):
        """Implementation of the 'Euclidean Distance Matrix Trick'
        Reference: https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
        """

        # x <- size(batch_size, )
        x2 = torch.matmul(x, x.T)  # size(num_batchs, num_batchs)
        x2_norm = torch.diag(x2)  # size(num_batchs,)

        # size(num_batchs, num_batchs)
        distance = x2_norm.unsqueeze(0) + x2_norm.unsqueeze(1) - 2.0 * x2
        distance = torch.nn.functional.relu(distance)

        mask = distance.eq(0.0).float()
        distance = distance + mask * eps
        distance = torch.sqrt(distance + eps) * (1.0 - mask) + eps
        return distance

    def _get_valid_triplet_mask(self, labels) -> torch.Tensor:
        # generating the triplets using the online strategy.
        # labels: batch of labels -> size(num_batchs,)
        # a triplet is valid if:
        # 1) labels[i]==labels[j] and labels[i]!=labels[k]
        # 2) i!=j and j!=k

        same_indices = torch.eye(labels.size(0), dtype=dtype).bool()
        not_same_indices = torch.logical_not(same_indices)

        # indices where labels're distinct
        ij = not_same_indices.unsqueeze(2)
        ik = not_same_indices.unsqueeze(1)
        jk = not_same_indices.unsqueeze(0)
        distinct_indices = torch.logical_and(
            jk, torch.logical_and(ij, ik)
        ).to(device)  # size(num_batchs, num_batchs, num_batchs)

        same_labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        ij = same_labels.unsqueeze(2)
        ik = same_labels.unsqueeze(1)
        # size(num_batchs, num_batchs, num_batchs)
        valid_indices = torch.logical_and(ij, torch.logical_not(ik)).to(device)

        return torch.logical_and(distinct_indices, valid_indices)

    def _get_ap_mask(self, labels):
        same_indices = torch.eye(labels.size(0), dtype=dtype).bool()
        not_same_indices = torch.logical_not(same_indices)
        same_labels = torch.logical_not(labels.unsqueeze(0) == labels.unsqueeze(1))
        return torch.logical_and(same_labels, not_same_indices)

    def _get_an_mask(self, labels):
        same_labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        return torch.logical_not(same_labels)

    def _batch_all_triplet_loss(self, embeddings, labels):
        # here we chose all kinds of triplets
        distance = self._euclidean_distance(embeddings)
        ap_dist = distance.unsqueeze(2)
        an_dist = distance.unsqueeze(1)
        loss = ap_dist - an_dist + self.bias

        mask = self._get_valid_triplet_mask(labels).float()
        loss = loss * mask
        loss = torch.nn.functional.relu(loss)

        # average only the remaining positive values
        num_positives = loss[loss > 1e-16].size(0)
        loss = loss.sum() / (num_positives + 1e-16)
        return loss

    def _batch_hard_triplet_loss(self, embeddings, labels):
        # here we only chose hard triplets
        # embeddings <- size(num_batchs, embed_size)

        distance = self._euclidean_distance(embeddings)

    def forward(self, embeddings, labels, mode="all"):
        if mode.lower() == "all":
            loss = self._batch_all_triplet_loss(embeddings, labels)
        elif mode.lower() == "hard":
            loss = self._batch_hard_triplet_loss(embeddings, labels)
        else:
            raise NotImplementedError

        return loss


class OfflineContrastiveLoss(nn.Module):
    def __init__(self, margin: int = 1, reduce: str = "mean") -> None:
        super(OfflineContrastiveLoss, self).__init__()
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
