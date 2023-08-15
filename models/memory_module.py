import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # N x C
        self.bias = None
        self.shrink_thres= shrink_thres
        self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxN) = TxN
        att_weight = F.softmax(att_weight, dim=1)  # TxN , 논문에선 1xN

        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)    # Re-normalize, TxN

        mem_trans = self.weight.permute(1, 0)  # Mem^T, CxN
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxN) x (NxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )

# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres, device):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        batch, channel, row, col = input.shape
        x = input.permute(0, 2, 3, 1)   # (B, row, col, channel)
        x = x.contiguous()
        x = x.view(-1, channel)         # (B x row x col, channel)
        #
        y_and = self.memory(x)          
        #
        y   = y_and['output']
        att = y_and['att']
        #
        y   = y.view(batch, row, col, channel)
        y   = y.permute(0, 3, 1, 2)     # (B, channel, row, col)
        att = att.view(batch, row, col, self.mem_dim)
        att = att.permute(0, 3, 1, 2)

        return {'output': y, 'att': att}

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output