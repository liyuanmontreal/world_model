
import argparse
import numpy as np
from tqdm import trange
from .env_wrapper import CarRacingVAEWrapper

def collect(episodes: int, out_path: str):
    env = CarRacingVAEWrapper()
    obs_list, act_list = [], []
    for _ in trange(episodes, desc="Collecting"):
        x = env.reset()
        done = False
        while not done:
            a = env.env.action_space.sample()
            x2, r, terminated, truncated, _ = env.step(a)
            obs_list.append(x.numpy())
            act_list.append(a)
            x = x2
            done = terminated or truncated
    env.close()
    obs = np.stack(obs_list, 0)
    act = np.stack(act_list, 0)
    np.savez_compressed(out_path, observations=obs, actions=act)
    print(f"Saved {obs.shape[0]} steps to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", type=str, default="data/car_racing_samples.npz")
    args = p.parse_args()
    collect(args.episodes, args.out)
