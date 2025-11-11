
import torch
import torch.nn as nn

class LinearController(nn.Module):
    def __init__(self, z_dim=32, h_dim=256, action_dim=3):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.linear = nn.Linear(z_dim + h_dim, action_dim)

    def forward(self, z, h):
        # 兼容多种 h 维度情况
        if isinstance(h, tuple):  # 如果是 LSTM (h, c)
            h = h[0]
        if h.dim() == 3:  # (num_layers, B, H)
            h = h[-1]
        if h.dim() == 2 and h.size(0) == 1:  # (1, H)
            h = h[0]
        if h.dim() == 1:  # (H,)
            h = h.unsqueeze(0)  # 变成 (1, H)

        # 确保 z 也是二维 (B, z_dim)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # 拼接并输出动作
        x = torch.cat([z, h], dim=-1)
        return torch.tanh(self.linear(x))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view_as(p))
            idx += n
