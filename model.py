import torch
import torch.nn as nn
from noise_channel import DifferentiableChannel

class DNAAutoencoder(nn.Module):
    def __init__(self, k_bits=128, seq_len=80, sub_rate=0.01):
        super().__init__()
        self.k_bits = k_bits
        self.seq_len = seq_len

        # Encoder: Projects binary bits to continuous nucleotide logits
        self.encoder = nn.Sequential(
            nn.Linear(k_bits, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * 4)
        )

        # Biological Differentiable Channel
        self.channel = DifferentiableChannel(sub_rate=sub_rate)

        # Decoder: Maps corrupted sequences back to binary payload
        self.decoder = nn.Sequential(
            nn.Linear(seq_len * 4, 256),
            nn.ReLU(),
            nn.Linear(256, k_bits),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        logits = self.encoder(x).view(batch_size, self.seq_len, 4)
        noisy_bases = self.channel(logits)
        flat_bases = noisy_bases.view(batch_size, -1)
        reconstructed_bits = self.decoder(flat_bases)
        return reconstructed_bits, noisy_bases
