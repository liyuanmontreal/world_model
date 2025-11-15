import numpy as np
import torch
import cv2

from src.custom_carracing import RealisticCarRacing

class CarRacingVAEWrapper:
    """
    Wrap RealisticCarRacing to produce VAE-ready tensors.

    - env.env: underlying RealisticCarRacing
    - reset() / step() 返回处理后的 obs: torch.Tensor [3, H, W] in [-0.5, 0.5]
    """

    def __init__(self, obs_size=64):
        self.obs_size = obs_size      # 供 VAE 使用的分辨率
        self.env = RealisticCarRacing(render_mode="rgb_array", obs_size=96)

    def _proc_obs(self, img):
        img = cv2.resize(img, (self.obs_size, self.obs_size))
        img = img.astype(np.float32) / 255.0 - 0.5
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.tensor(img, dtype=torch.float32)

    def reset(self):
        obs, _ = self.env.reset()
        return self._proc_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._proc_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
