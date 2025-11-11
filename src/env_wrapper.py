import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms


class CarRacingVAEWrapper:
    """Robust wrapper for CarRacing (auto-detect version, fix Box2D dtype issue)."""

    def __init__(self, seed: int = 0):
        # 尝试不同版本
        for ver in ["v3", "v2", "v1"]:
            env_id = f"CarRacing-{ver}"
            try:
                self.env = gym.make(env_id, continuous=True, render_mode="rgb_array")
                print(f"[INFO] Using environment: {env_id}")
                break
            except gym.error.Error:
                continue
        else:
            raise RuntimeError("No compatible CarRacing environment found.")
        self.env.reset(seed=seed)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ])

    def reset(self):
        obs, _ = self.env.reset()
        return self._proc_obs(obs)

    def step(self, action):
        # 转换为 float64，防止 Box2D motorSpeed 类型错误
        action = np.array(action, dtype=np.float64)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._proc_obs(obs), reward, terminated, truncated, info

    def _proc_obs(self, obs):
        """将环境输出图像转换为 tensor"""
        img = np.array(obs)
        return self.transform(img)

    def close(self):
        self.env.close()
