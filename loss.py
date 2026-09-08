import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedLoss(nn.Module):
    def __init__(self, lambda_gc=1.0, lambda_homo=2.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_gc = lambda_gc
        self.lambda_homo = lambda_homo

    def forward(self, reconstructed_bits, target_bits, base_probabilities):
        # 1. Payload reconstruction fidelity
        l_recon = self.bce(reconstructed_bits, target_bits)

        # 2. GC-content quadratic hinge penalty (40% - 60% window)
        c_prob = base_probabilities[:, :, 1]
        g_prob = base_probabilities[:, :, 2]
        gc_ratio = torch.mean(c_prob + g_prob, dim=1)
        l_gc = torch.mean(F.relu(torch.abs(gc_ratio - 0.50) - 0.10) ** 2)

        # 3. Adjacent homopolymer overlap penalty
        p_t = base_probabilities[:, :-1, :]
        p_next = base_probabilities[:, 1:, :]
        adjacent_overlap = torch.sum(p_t * p_next, dim=-1)
        l_homo = torch.mean(adjacent_overlap ** 2)

        # Composite multi-objective loss
        l_total = l_recon + (self.lambda_gc * l_gc) + (self.lambda_homo * l_homo)
        return l_total, l_recon, l_gc, l_homo
