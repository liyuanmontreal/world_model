import torch
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import Controller

class WorldModelSimulator:
    def __init__(self):
        self.vae = VAE()
        self.rnn = MDNRNN()
        self.controller = Controller()

    def imagine(self, start_image):
        with torch.no_grad():
            recon, mu, logvar = self.vae(start_image)
            z = self.vae.reparameterize(mu, logvar)
            h = torch.zeros(1, 1, 256)
            a = torch.zeros(1, 3)
            for t in range(10):
                action = self.controller(z, h.squeeze(0))
                pi, mu, sigma, h = self.rnn(z, action, (h, h))
                z = mu.mean(1)
            print("Imagined trajectory in latent space complete.")
