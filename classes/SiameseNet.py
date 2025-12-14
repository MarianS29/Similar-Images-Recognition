from torch.nn import Module

from classes import EmbeddingNet


class SiameseNet(Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim)

    def forward(self, x1, x2):
        z1 = self.embedding_net(x1)
        z2 = self.embedding_net(x2)
        return z1, z2
