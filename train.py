import torch
from data_generator import get_dataloader
from model import DNAAutoencoder
from loss import ConstrainedLoss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = DNAAutoencoder(k_bits=128, seq_len=80, sub_rate=0.01).to(device)
    criterion = ConstrainedLoss(lambda_gc=1.0, lambda_homo=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader = get_dataloader(data_type="uniform", k=128, batch_size=128)

    model.train()
    print("Starting Constrained Autoencoder Training...")
    for epoch in range(1, 51):
        total_loss, total_ber = 0.0, 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_bits, bases = model(batch)
            loss, l_recon, l_gc, l_homo = criterion(recon_bits, batch, bases)
            loss.backward()
            optimizer.step()

            predicted_bits = (recon_bits > 0.5).float()
            ber = torch.mean(torch.abs(predicted_bits - batch))
            total_loss += loss.item()
            total_ber += ber.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/50 | Loss: {total_loss/len(loader):.4f} | BER: {total_ber/len(loader):.4f}")

    torch.save(model.state_dict(), "dna_autoencoder.pth")
    print("Model checkpoint saved to dna_autoencoder.pth")

if __name__ == "__main__":
    train()
