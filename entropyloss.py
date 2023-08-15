import torch
import torch.nn as nn

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        ent_loss_val = self.entropy_loss(input)
        return ent_loss_val