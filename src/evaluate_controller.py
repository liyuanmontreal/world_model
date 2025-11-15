# src/evaluate_controller.py

import argparse
import numpy as np
import torch

from src.vae import VAE
from src.controller import LinearController
from src.simple_carracing_env import SimpleCarRacingEnv


def evaluate(controller, vae, episodes: int = 5, device: str = "cpu"):
    env = SimpleCarRacingEnv(max_steps=500)

    controller.eval()
    vae.eval()

    all_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        # 用一个固定的 h 向量（全 0），不再使用 MDNRNN
        h = torch.zeros(1, controller.h_dim, device=device)

        with torch.no_grad():
            while True:
                # obs: (64,64,3) → (1,3,64,64)
                obs_tensor = torch.from_numpy(obs).to(device).permute(2, 0, 1).unsqueeze(0)

                # VAE encode
                mu, logvar = vae.encode(obs_tensor)
                # 这里直接用 mu 作为 z（和 Ha 论文里 evaluation 类似）
                z = mu  # shape: (1, latent_dim)

                # Controller 输出动作 a: (1,3)
                a = controller(z, h).detach().cpu().numpy().squeeze()
                # 保证是 1D 向量 shape (3,)
                if a.ndim > 1:
                    a = a.reshape(-1)
                if a.shape[0] != 3:
                    raise ValueError(f"Controller action shape expected (3,), got {a.shape}")

                # 环境一步
                obs, r, terminated, truncated, info = env.step(a)
                total_reward += float(r)

                if terminated or truncated:
                    break

        print(f"Episode {ep+1}: reward = {total_reward:.2f}")
        all_rewards.append(total_reward)

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    print(f"Avg reward over {episodes} episodes: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller_ckpt", type=str, default="checkpoints/controller_cmaes.pt")
    parser.add_argument("--vae_ckpt", type=str, default="checkpoints/vae_final.pt")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    device = "cpu"

    print(f"[INFO] Loading VAE from {args.vae_ckpt}")
    vae = VAE(in_channels=3, latent_dim=32).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))

    print(f"[INFO] Loading Controller from {args.controller_ckpt}")
    controller = LinearController(z_dim=32, h_dim=256, action_dim=3).to(device)
    controller.load_state_dict(torch.load(args.controller_ckpt, map_location=device))

    evaluate(controller, vae, episodes=args.episodes, device=device)


if __name__ == "__main__":
    main()
