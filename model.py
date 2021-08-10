import torch.nn as nn
import torch


class BPR(nn.Module):
    def __init__(self, num_user, num_item, dim):
        super(BPR, self).__init__()
        self.U = nn.Parameter(torch.rand(num_user, dim))
        self.V = nn.Parameter(torch.rand(num_item, dim))

    def forward(self, u, i, j):
        x_ui = torch.mul(self.U[u, :], self.V[i, :]).sum(dim=1)
        x_uj = torch.mul(self.U[u, :], self.V[j, :]).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = torch.log(torch.sigmoid(x_uij)).mean()
        return - log_prob
