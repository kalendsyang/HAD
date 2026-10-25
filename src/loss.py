import torch.nn.functional as F
from torch import nn

BENIGN, MALICIOUS, SURROGATE = 0, 1, 2


class HierarchicalLoss(nn.Module):
    """Hierarchical contrastive loss of HAD.

    L = lambda_b * L_b + lambda_s * L_s + lambda_m * L_m, where
        L_b = mean_{x_b} DIS(x_b)
        L_s = mean_{x_b, x_s} max(DIS(x_b) - DIS(x_s) + delta, 0)
        L_m = mean_{x_s, x_m} max(DIS(x_s) - DIS(x_m) + delta, 0)
    """

    def __init__(self, delta=1.0, lambda_b=1.0, lambda_s=1.0, lambda_m=1.0):
        super().__init__()
        self.delta = delta
        self.lambda_b = lambda_b
        self.lambda_s = lambda_s
        self.lambda_m = lambda_m

    def benign_loss(self, d_b):
        return d_b.mean()

    def surrogate_loss(self, d_b, d_s):
        return self.pairwise_hinge(d_b, d_s)

    def malicious_loss(self, d_s, d_m):
        return self.pairwise_hinge(d_s, d_m)

    def pairwise_hinge(self, d_low, d_high):
        """Mean of max(d_low_i - d_high_j + delta, 0) over all pairs (i, j)."""
        return F.relu(d_low.unsqueeze(1) - d_high.unsqueeze(0) + self.delta).mean()

    def forward(self, dis, y):
        d_b = dis[y == BENIGN]
        d_m = dis[y == MALICIOUS]
        d_s = dis[y == SURROGATE]

        loss_b = self.benign_loss(d_b)
        loss_s = self.surrogate_loss(d_b, d_s)
        loss_m = self.malicious_loss(d_s, d_m)
        loss = self.lambda_b * loss_b + self.lambda_s * loss_s + self.lambda_m * loss_m

        return loss, {"L_b": loss_b.item(), "L_s": loss_s.item(), "L_m": loss_m.item()}
