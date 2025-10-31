import torch
import torch.nn as nn
import torch.nn.functional as F


def half_logdet(X):
    return torch.linalg.cholesky_ex(X)[0].diagonal().log().sum()


class MCR(torch.nn.Module):
    def __init__(self, eps=0.05):
        super(MCR, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        m, p = X.shape
        X = F.normalize(X, dim=-1, p=2)
        cov = X.T @ X  # [p, p]
        scalar = p / (m * self.eps)
        I = torch.eye(p, device=X.device)
        loss = -half_logdet(I + scalar * cov)
        loss *= (p + m) / (p * m)  # balancing factor
        return loss