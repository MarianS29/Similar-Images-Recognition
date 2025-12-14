import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Linear, Module


class EmbeddingNet(Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # compatibilitate pentru versiuni diferite de torchvision
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            self.backbone = models.resnet18(pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        # L2-normalizare -> vectorii stau pe sferă; util pt cosine / L2
        x = F.normalize(x, p=2, dim=1)
        return x
