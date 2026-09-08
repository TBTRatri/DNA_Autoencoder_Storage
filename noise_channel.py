import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableChannel(nn.Module):
    def __init__(self, sub_rate=0.01, tau=1.0):
        super().__init__()
        self.sub_rate = sub_rate
        self.tau = tau

    def forward(self, logits):
        # Continuous Gumbel-Softmax categorical relaxation
        soft_bases = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        
        # Simulated substitution noise layer
        if self.training:
            noise = torch.rand_like(soft_bases) * self.sub_rate
            corrupted = soft_bases * (1.0 - self.sub_rate) + noise
            return F.normalize(corrupted, p=1, dim=-1)
        return soft_bases
