
import torch
from .vae import VAE
from .mdn_rnn import MDNRNN
from .controller import LinearController

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def imagine(steps=20):
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae_final.pt", map_location=DEVICE))
    mdn = MDNRNN().to(DEVICE)
    mdn.load_state_dict(torch.load("checkpoints/mdnrnn_final.pt", map_location=DEVICE))
    ctrl = LinearController().to(DEVICE)
    ctrl.load_state_dict(torch.load("checkpoints/controller_cmaes.pt", map_location=DEVICE))
    vae.eval(); mdn.eval(); ctrl.eval()
    with torch.no_grad():
        z = torch.zeros(vae.latent_dim, device=DEVICE)
        h = torch.zeros(1,1,mdn.hidden_dim, device=DEVICE)
        c = torch.zeros(1,1,mdn.hidden_dim, device=DEVICE)
        frames = []
        for t in range(steps):
            a = ctrl(z.unsqueeze(0), h).squeeze(0).clamp(-1,1)
            log_pi, mu, logsigma, (h, c) = mdn(
                z.view(1,1,-1), a.view(1,1,-1), (h, c)
            )
            log_pi = log_pi[0,0]
            mu = mu[0,0]
            logsigma = logsigma[0,0]
            pi = torch.exp(log_pi)
            k = torch.multinomial(pi, 1).item()
            m = mu[k]
            s = torch.exp(logsigma[k])
            eps = torch.randn_like(m)
            z = m + s * eps
            x = vae.decode(z.unsqueeze(0)).cpu()
            frames.append(x)
        print(f"Imagined {steps} dream frames in latent world.")
        return frames

if __name__ == "__main__":
    imagine(steps=10)
