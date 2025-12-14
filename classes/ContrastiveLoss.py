import torch.nn.functional as F
from torch.nn import Module


class ContrastiveLoss(Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        # z1, z2: [B, D], y: [B, 1] (1 = similar, 0 = diferit)
        distances = F.pairwise_distance(z1, z2)  # [B]
        y = y.view(-1)  # [B]

        loss_pos = y * distances.pow(2)
        loss_neg = (1 - y) * F.relu(self.margin - distances).pow(2)

        loss = (loss_pos + loss_neg).mean()
        return loss
