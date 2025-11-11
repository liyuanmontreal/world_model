
import argparse
import numpy as np
import torch
from .vae import VAE
from .controller import LinearController
from .env_wrapper import CarRacingVAEWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    vae = VAE().to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae_final.pt", map_location=DEVICE))
    ctrl = LinearController().to(DEVICE)
    ctrl.load_state_dict(torch.load("checkpoints/controller_cmaes.pt", map_location=DEVICE))
    vae.eval(); ctrl.eval()
    env = CarRacingVAEWrapper()
    scores = []
    with torch.no_grad():
        for ep in range(args.episodes):
            x = env.reset()
            done = False
            total = 0.0
            h = torch.zeros(1,1,ctrl.h_dim, device=DEVICE)
            while not done:
                x_in = x.unsqueeze(0).to(DEVICE)
                mu, logvar = vae.encode(x_in)
                z = vae.reparameterize(mu, logvar)
                a = ctrl(z, h).cpu().numpy()[0]
                a = np.array(a, dtype=np.float64)
                obs, r, terminated, truncated, _ = env.env.step(a)
                total += r
                x = env._proc_obs(obs)
                done = terminated or truncated
            scores.append(total)
            print(f"Episode {ep+1}: reward={total:.2f}")
    env.close()
    print(f"Avg reward over {args.episodes} eps: {np.mean(scores):.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    main(p.parse_args())
