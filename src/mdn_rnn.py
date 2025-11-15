import torch
import torch.nn as nn

class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, action_dim=3, h_dim=256, n_gaussians=5):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.input_dim = z_dim + action_dim
        self.num_gaussians = n_gaussians
        self.hidden = None

        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.h_dim,
            batch_first=True
        )

        # mixture density outputs
        self.fc = nn.Linear(self.h_dim, n_gaussians * (2 * z_dim + 1))

    def init_hidden(self, batch_size=1):
        """Return (h0, c0) with correct shape."""
        h0 = torch.zeros(1, batch_size, self.h_dim)
        c0 = torch.zeros(1, batch_size, self.h_dim)
        self.hidden = (h0, c0)
        return self.hidden

    def forward(self, inputs, hidden=None):
        """
        inputs: tuple (z, a)
            z: (1, z_dim)
            a: (1, action_dim)
        """

        z, a = inputs
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)

        # concat → (1, 35)
        x = torch.cat([z, a], dim=-1)

        # Add sequence dimension → (1, 1, 35)
        x = x.unsqueeze(1)

        # hidden: (1,1,256)
        if hidden is None:
            hidden = self.hidden

        out, hidden = self.rnn(x, hidden)

        # out shape: (1,1,256)
        out = out[:, -1, :]  # → (1,256)

        mdn_params = self.fc(out)  # → (1, n_gaussians*(2*z_dim+1))
        return mdn_params, hidden
