"""
src.models.autoencoder
======================
Nairobi Urban Flood Digital Twin — Spatial Convolutional Autoencoder (CAE)

PURPOSE
-------
1. Compress 2D spatial terrain matrices (DEM, Slope, TWI) into a compact
   128-dim latent vector, and reconstruct them back — a genuine
   reconstruction autoencoder (input domain == output domain), trained on
   random crops of the single Nairobi terrain grid since there is only
   one full scene available (src.models.train.pretrain_terrain_autoencoder).
2. Its encoder's convolutional weights are transferred into
   src.models.lstm_surrogate.FrameFeatureEncoder before ConvLSTM training,
   so the terrain-understanding half of the surrogate is genuinely
   pretrained rather than learned from scratch inside the small
   flood-sample dataset.

MEMORY CONTRACT
---------------
Lightweight parameter size (~250k params), low memory footprint (< 100 MB RAM).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_out_dim(n: int, times: int = 3, stride: int = 2) -> int:
    """Output size after `times` stride-2, padding-1, kernel-3 convolutions."""
    for _ in range(times):
        n = (n + 2 * 1 - 3) // stride + 1
    return n


class SpatialEncoder(nn.Module):
    """Encodes (B, in_channels, H, W) spatial maps into a latent vector. Resolution-agnostic."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, target_h: int = 198, target_w: int = 252, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(dropout)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(dropout)

        h_out, w_out = _conv_out_dim(target_h), _conv_out_dim(target_w)
        self.flat_h, self.flat_w = h_out, w_out
        self.fc_latent = nn.Linear(128 * h_out * w_out, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x_flat = x.reshape(x.size(0), -1)
        latent = self.fc_latent(x_flat)
        return latent


class SpatialDecoder(nn.Module):
    """Decodes a latent vector back to a spatial map (B, out_channels, H, W). Resolution-agnostic."""

    def __init__(self, latent_dim: int = 128, out_channels: int = 1, target_h: int = 198, target_w: int = 252, dropout: float = 0.1):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.flat_h, self.flat_w = _conv_out_dim(target_h), _conv_out_dim(target_w)

        self.fc_dec = nn.Linear(latent_dim, 128 * self.flat_h * self.flat_w)

        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout2d(dropout)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(dropout)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 128, self.flat_h, self.flat_w)
        x = self.drop3(F.relu(self.bn3(self.deconv3(x))))
        x = self.drop2(F.relu(self.bn2(self.deconv2(x))))
        x = F.relu(self.bn1(self.deconv1(x)))
        out = F.relu(self.final_conv(x))
        if out.shape[-2:] != (self.target_h, self.target_w):
            out = F.interpolate(out, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        return out


class SpatialAutoencoder(nn.Module):
    """Complete Spatial Convolutional Autoencoder — reconstructs its own input."""

    def __init__(self, in_channels: int = 3, out_channels: int | None = None, latent_dim: int = 128,
                 target_h: int = 198, target_w: int = 252, dropout: float = 0.1):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.encoder = SpatialEncoder(in_channels, latent_dim, target_h, target_w, dropout)
        self.decoder = SpatialDecoder(latent_dim, out_channels, target_h, target_w, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
