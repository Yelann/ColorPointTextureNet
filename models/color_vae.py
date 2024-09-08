<<<<<<< Updated upstream
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F


class ColorVAE(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=32, hidden_dim=64, pixel_num=128):
        super(ColorVAE, self).__init__()
        self.PIXEL_NUM = pixel_num
    
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()

        latent_pixel = int(pixel_num / 8)

        self.fc_mu = nn.Linear(hidden_dim * 4 * latent_pixel * latent_pixel, embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4 * latent_pixel * latent_pixel, embedding_dim)
        
        # Decoder layers
        self.fc_decode = nn.Linear(embedding_dim, hidden_dim * 4 * latent_pixel * latent_pixel)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # Define a loss function for VAE
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.float(), x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def forward(self, uv_imgs, masks):
        h = uv_imgs.reshape(uv_imgs.shape[0], uv_imgs.shape[3], uv_imgs.shape[1], uv_imgs.shape[2]).float()

        # Encode
        h = self.encoder(h)
        h_flatten = self.flatten(h)
        mu = self.fc_mu(h_flatten)
        logvar = self.fc_logvar(h_flatten)

        # Reparam
        z = self.reparameterize(mu, logvar)

        if self.training:
            # Decode
            h2 = self.fc_decode(z)
            h2 = h2.view(h.shape)
            h2 = self.decoder(h2)

            # Loss
            pred_uv_imgs = h2.view(uv_imgs.shape[0], self.PIXEL_NUM, self.PIXEL_NUM, 3)
            pred_uv_imgs = pred_uv_imgs * masks

            loss = self.loss_function(pred_uv_imgs, uv_imgs, mu, logvar) / 1e5
            return pred_uv_imgs, z, loss
        else:
            return torch.zeros(0).cuda(), z, torch.zeros(1).cuda()

=======
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class ColorVAE(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=48, hidden_dim=64, pixel_num=128):
        super(ColorVAE, self).__init__()
        self.PIXEL_NUM = pixel_num
        self.EMB = embedding_dim
        self.HID = hidden_dim
        self.LAT = int(pixel_num / 8) 

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()

        latent_pixel = self.LAT

        self.fc_mu = nn.Linear(hidden_dim * 4 * latent_pixel * latent_pixel, embedding_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4 * latent_pixel * latent_pixel, embedding_dim)
        
        # Decoder layers
        self.fc_decode = nn.Linear(embedding_dim, hidden_dim * 4 * latent_pixel * latent_pixel)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    # Define a loss function for VAE
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.float(), x.float(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    
    def forward(self, uv_imgs, masks, use_mask = True):
        h = uv_imgs.reshape(uv_imgs.shape[0], uv_imgs.shape[3], uv_imgs.shape[1], uv_imgs.shape[2]).float()

        # Encode
        h = self.encoder(h)
        h_flatten = self.flatten(h)
        mu = self.fc_mu(h_flatten)
        logvar = self.fc_logvar(h_flatten)

        # Reparam
        z = self.reparameterize(mu, logvar)

        # Decode
        h2 = self.fc_decode(z)
        h2 = h2.view(h.shape)
        h2 = self.decoder(h2)

        # Loss
        pred_uv_imgs = h2.view(uv_imgs.shape[0], self.PIXEL_NUM, self.PIXEL_NUM, 3)

        if use_mask:
            pred_uv_imgs = pred_uv_imgs * masks
            loss = self.loss_function(pred_uv_imgs, uv_imgs, mu, logvar) / 1e5
            return pred_uv_imgs, z, loss
        else:
            return pred_uv_imgs, z, torch.zeros(1).cuda()


    def get_emb(self, uv_imgs):
        h = uv_imgs.reshape(uv_imgs.shape[0], uv_imgs.shape[3], uv_imgs.shape[1], uv_imgs.shape[2]).float()

        # Encode
        h = self.encoder(h)
        h_flatten = self.flatten(h)
        mu = self.fc_mu(h_flatten)
        logvar = self.fc_logvar(h_flatten)

        # Reparam
        z = self.reparameterize(mu, logvar)
        return z


    def get_random_emb(self, num):
        z = torch.randn(num, self.EMB).cuda()
        return z


    def get_random_result(self, num):
        z = torch.randn(num, self.EMB).cuda()

        # Decode
        h2 = self.fc_decode(z)
        h2 = h2.reshape(num, self.HID * 4, self.LAT, self.LAT)
        h2 = self.decoder(h2)

        # Loss
        rand_uv_img = h2.view(num, self.PIXEL_NUM, self.PIXEL_NUM, 3)
        return rand_uv_img, z

>>>>>>> Stashed changes
