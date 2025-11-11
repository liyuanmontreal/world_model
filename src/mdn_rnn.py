
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, a, hidden=None):
        x = torch.cat([z, a], dim=-1)
        h, hidden = self.rnn(x, hidden)
        log_pi = F.log_softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h).view(*h.shape[:2], self.num_gaussians, self.latent_dim)
        logsigma = self.fc_logsigma(h).view(*h.shape[:2], self.num_gaussians, self.latent_dim)
        return log_pi, mu, logsigma, hidden

    def mdn_loss(self, log_pi, mu, logsigma, z_next):
        B, T, D = z_next.shape
        K = self.num_gaussians
        z_expand = z_next.unsqueeze(2).expand(B, T, K, D)
        sigma = torch.exp(logsigma)
        const = torch.log(torch.tensor(2.0 * torch.pi, device=z_next.device))
        comp_log_prob = -0.5 * (((z_expand - mu) / sigma) ** 2 + 2 * logsigma + const)
        comp_log_prob = comp_log_prob.sum(-1)  # (B,T,K)
        log_prob = torch.logsumexp(log_pi + comp_log_prob, dim=-1)  # (B,T)
        return -log_prob.mean()
