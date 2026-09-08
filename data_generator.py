import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def generate_uniform_payloads(num_samples=100000, k=128):
    """Generates high-entropy pseudo-random uniform binary bitstreams."""
    bits = np.random.randint(0, 2, size=(num_samples, k)).astype(np.float32)
    return torch.tensor(bits)

def generate_ascii_payloads(num_samples=100000, k=128):
    """Generates structured binary bitstreams from ASCII-encoded text."""
    text = "Bioinformatics DNA storage constrained autoencoder project dataset payload " * 2000
    byte_array = np.frombuffer(text.encode('ascii'), dtype=np.uint8)
    bit_array = np.unpackbits(byte_array).astype(np.float32)
    total_bits = num_samples * k
    repeated = np.resize(bit_array, total_bits)
    return torch.tensor(repeated.reshape((num_samples, k)))

class DNAPayloadDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(data_type="uniform", k=128, batch_size=64):
    if data_type == "uniform":
        data = generate_uniform_payloads(100000, k)
    else:
        data = generate_ascii_payloads(100000, k)
    return DataLoader(DNAPayloadDataset(data), batch_size=batch_size, shuffle=True)
