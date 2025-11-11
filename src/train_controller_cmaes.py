
import argparse
import torch
import numpy as np
from cma import CMAEvolutionStrategy
from .vae import VAE
from .mdn_rnn import MDNRNN
from .controller import LinearController

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sample_next_z(mdn, z, a, h, c):
    with torch.no_grad():
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
        z_next = m + s * eps
    return z_next, h, c

def rollout_dream(params, vae, mdn, ctrl, horizon=300):
    ctrl.set_flat_params(torch.tensor(params, dtype=torch.float32, device=DEVICE))
    ctrl.eval(); vae.eval(); mdn.eval()
    with torch.no_grad():
        z = torch.zeros(vae.latent_dim, device=DEVICE)
        h = torch.zeros(1,1,mdn.hidden_dim, device=DEVICE)
        c = torch.zeros(1,1,mdn.hidden_dim, device=DEVICE)
        total = 0.0
        for t in range(horizon):
            a = ctrl(z.unsqueeze(0), h).squeeze(0).clamp(-1,1)
            z, h, c = sample_next_z(mdn, z, a, h, c)
            # surrogate reward: keep latent small (placeholder)
            total += float(-0.01 * (z**2).mean().item())
    return -total  # CMA-ES minimizes

def main(args):
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae_final.pt", map_location=DEVICE))
    mdn = MDNRNN().to(DEVICE)
    mdn.load_state_dict(torch.load("checkpoints/mdnrnn_final.pt", map_location=DEVICE))
    ctrl = LinearController().to(DEVICE)
    n_params = ctrl.num_params
    es = CMAEvolutionStrategy(n_params * [0.0], 0.5, {"popsize": args.pop})
    for i in range(args.iters):
        solutions = es.ask()
        fitness = [rollout_dream(s, vae, mdn, ctrl, args.horizon) for s in solutions]
        es.tell(solutions, fitness)
        es.disp()
    best = es.result.xbest
    ctrl.set_flat_params(torch.tensor(best, dtype=torch.float32, device=DEVICE))
    torch.save(ctrl.state_dict(), "checkpoints/controller_cmaes.pt")
    print("saved checkpoints/controller_cmaes.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--horizon", type=int, default=300)
    main(p.parse_args())
