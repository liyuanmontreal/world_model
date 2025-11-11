import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, z_dim=32, h_dim=256, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)
