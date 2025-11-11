
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from .vae import VAE
from .mdn_rnn import MDNRNN

class SeqDataset(Dataset):
    def __init__(self, npz_path, vae, seq_len=32, device="cpu"):
        d = np.load(npz_path)
        x = d["observations"].astype("float32")
        a = d["actions"].astype("float32")
        if x.max() > 1.0:
            x /= 255.0
        self.a = torch.from_numpy(a)
        self.seq_len = seq_len
        self.device = device
        vae.eval().to(device)
        zs = []
        with torch.no_grad():
            bs = 256
            for i in trange(0, x.shape[0], bs, desc="encode z"):
                xb = torch.from_numpy(x[i:i+bs]).to(device)
                mu, logvar = vae.encode(xb)
                z = vae.reparameterize(mu, logvar)
                zs.append(z.cpu())
        self.z = torch.cat(zs, 0)
        assert self.z.shape[0] == self.a.shape[0]

    def __len__(self):
        return max(0, self.z.shape[0] - self.seq_len - 1)

    def __getitem__(self, i):
        z = self.z[i:i+self.seq_len]
        a = self.a[i:i+self.seq_len]
        z_next = self.z[i+1:i+self.seq_len+1]
        return z, a, z_next

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(latent_dim=args.latent)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    ds = SeqDataset(args.data, vae, args.seq_len, device)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    mdn = MDNRNN(latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(mdn.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(1, args.epochs+1):
        mdn.train()
        tot = 0.0
        n = 0
        for z, a, z_next in dl:
            z = z.to(device)
            a = a.to(device)
            z_next = z_next.to(device)
            log_pi, mu, logsigma, _ = mdn(z, a)
            loss = mdn.mdn_loss(log_pi, mu, logsigma, z_next)
            opt.zero_grad()
            loss.backward()
            opt.step()
            b = z.size(0)
            tot += loss.item()*b
            n += b
        print(f"[MDN-RNN] ep{ep}/{args.epochs} nll={tot/max(1,n):.6f}")
        torch.save(mdn.state_dict(), f"checkpoints/mdnrnn_ep{ep}.pt")
    torch.save(mdn.state_dict(), "checkpoints/mdnrnn_final.pt")
    print("saved checkpoints/mdnrnn_final.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/car_racing_samples.npz")
    p.add_argument("--vae_ckpt", type=str, default="checkpoints/vae_final.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    main(p.parse_args())
