import torch
import numpy as np
from data_generator import get_dataloader
from model import DNAAutoencoder

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNAAutoencoder(k_bits=128, seq_len=80, sub_rate=0.01).to(device)
    
    try:
        model.load_state_dict(torch.load("dna_autoencoder.pth", map_location=device))
    except FileNotFoundError:
        print("Note: dna_autoencoder.pth not found. Running evaluation on baseline initialization.")
        
    model.eval()

    loader = get_dataloader(data_type="uniform", k=128, batch_size=100)
    total_ber = []
    violations = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon_bits, bases = model(batch)

            pred = (recon_bits > 0.5).float()
            ber = torch.mean(torch.abs(pred - batch), dim=1).cpu().numpy()
            total_ber.extend(ber)

            discrete_bases = torch.argmax(bases, dim=-1).cpu().numpy()
            for seq in discrete_bases:
                total_sequences += 1
                # GC check (40% - 60%)
                gc_count = np.sum((seq == 1) | (seq == 2))
                gc_ratio = gc_count / len(seq)
                has_gc_violation = (gc_ratio < 0.40) or (gc_ratio > 0.60)

                # Homopolymer check (>= 3)
                diffs = np.diff(seq)
                has_homo_violation = any(diffs[i] == 0 and diffs[i+1] == 0 for i in range(len(diffs)-1))

                if has_gc_violation or has_homo_violation:
                    violations += 1

    print("\n================== Evaluation Results ==================")
    print(f"Mean Bit Error Rate (BER): {np.mean(total_ber):.5f}")
    print(f"Biological Violation Rate: {(violations / total_sequences) * 100:.2f}%")
    print(f"Effective Code Rate:       {128 / 80:.2f} bits/nucleotide")
    print("========================================================")

if __name__ == "__main__":
    evaluate()
