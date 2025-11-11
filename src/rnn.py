import torch
import torch.nn as nn

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)

    def forward(self, z, a, hidden=None):
        x = torch.cat([z, a], dim=-1).unsqueeze(1)
        h, hidden = self.rnn(x, hidden)
        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h))
        return pi, mu, sigma, hidden
