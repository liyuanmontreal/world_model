
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from .vae import VAE

class VaeDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        x = d["observations"].astype("float32")
        if x.max() > 1.0:
            x /= 255.0
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i])

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = VaeDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    vae = VAE(latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(1, args.epochs+1):
        vae.train()
        tot = rec_tot = kld_tot = 0.0
        for x in dl:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, rec, kld = VAE.loss_fn(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = x.size(0)
            tot += loss.item()*b
            rec_tot += rec.item()*b
            kld_tot += kld.item()*b
        n = len(ds)
        print(f"[VAE] ep{ep}/{args.epochs} loss={tot/n:.6f} rec={rec_tot/n:.6f} kld={kld_tot/n:.6f}")
        torch.save(vae.state_dict(), f"checkpoints/vae_ep{ep}.pt")
    torch.save(vae.state_dict(), "checkpoints/vae_final.pt")
    print("saved checkpoints/vae_final.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/car_racing_samples.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    main(p.parse_args())
