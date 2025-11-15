import argparse
import numpy as np
import torch
from tqdm import trange
from src.controller import LinearController
from src.vae import VAE
from src.mdn_rnn import MDNRNN
from src.env_wrapper import CarRacingVAEWrapper 
import matplotlib.pyplot as plt

def evaluate_once(env, controller, vae, rnn, horizon=1000):
    obs = env.reset()
    total_reward = 0

    rnn.hidden = rnn.init_hidden(batch_size=1)

    for _ in range(horizon):
        with torch.no_grad():
            # --- VAE encode ---
            mu, logvar = vae.encode(obs.unsqueeze(0))
            z = mu  # (1, latent_dim)

            # --- Controller uses (z, h_t) ---
            h_for_ctrl = rnn.hidden
            a = controller(z, h_for_ctrl).cpu().numpy().squeeze()
            a = np.clip(a, -1, 1).astype(np.float32)

        # --- Env step ---
        next_obs, reward, terminated, truncated, _ = env.env.step(a)
        total_reward += reward
        obs = env._proc_obs(next_obs)

        # --- Prepare RNN input (z_next, a) ---
        a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        mu_next, logvar_next = vae.encode(obs.unsqueeze(0))
        z_next = mu_next
        x = torch.cat([z_next, a_tensor], dim=-1).unsqueeze(0)  # (1,1,latent+action)

        # --- RNN update ---
        log_pi, mu, logsigma, next_hidden = rnn(x, rnn.hidden)
        rnn.hidden = next_hidden

        if terminated or truncated:
            break

    return total_reward


def finetune_real(controller, vae, rnn, iterations=50, popsize=16, sigma=0.05):
    env = CarRacingVAEWrapper()   # ← NEW ENV HERE

    mean_params = controller.get_flat_params().cpu().numpy()
    reward_trace_mean = []
    reward_trace_max = []

    for it in range(iterations):
        noise_list = []
        rewards = []

        for _ in range(popsize):
            noise = np.random.randn(*mean_params.shape) * sigma
            noise_list.append(noise)

            candidate = mean_params + noise
            controller.set_flat_params(torch.tensor(candidate, dtype=torch.float32))

            reward = evaluate_once(env, controller, vae, rnn)
            rewards.append(reward)

        rewards = np.array(rewards)
        noise_list = np.array(noise_list)

        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        mean_params += sigma / popsize * np.dot(normalized_rewards, noise_list)

        r_mean = rewards.mean()
        r_max = rewards.max()
        reward_trace_mean.append(r_mean)
        reward_trace_max.append(r_max)

        print(f"[Finetune] Iter {it+1}/{iterations}  mean={r_mean:.2f}  max={r_max:.2f}")

    controller.set_flat_params(torch.tensor(mean_params, dtype=torch.float32))
    torch.save(controller.state_dict(), "checkpoints/controller_finetuned.pt")
    print("✔ Saved: checkpoints/controller_finetuned.pt")

    plt.figure(figsize=(10,5))
    plt.plot(reward_trace_mean, label="Mean Reward", linewidth=2)
    plt.plot(reward_trace_max, label="Max Reward", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Real Finetuning Reward Curve (CustomCarRacing)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("checkpoints/reward_finetune_curve.png")
    print("📈 Saved → checkpoints/reward_finetune_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller_ckpt", type=str, default="checkpoints/controller_cmaes.pt")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.05)
    args = parser.parse_args()

    vae = VAE()
    vae.load_state_dict(torch.load("checkpoints/vae_final.pt", map_location="cpu"))
    vae.eval()

    rnn = MDNRNN()
    rnn.load_state_dict(torch.load("checkpoints/mdnrnn_final.pt", map_location="cpu"))
    rnn.eval()

    controller = LinearController()
    controller.load_state_dict(torch.load(args.controller_ckpt, map_location="cpu"))
    controller.eval()

    print("Controller parameters:", controller.num_params)

    finetune_real(controller, vae, rnn,
                  iterations=args.iters,
                  popsize=args.pop,
                  sigma=args.sigma)
