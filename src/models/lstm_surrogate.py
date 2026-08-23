"""
src.models.lstm_surrogate
=========================
Nairobi Urban Flood Digital Twin — Hybrid Conv-LSTM Predictive Surrogate Model

PURPOSE
-------
1. Process sequential spatiotemporal inputs (Batch, Seq_Len=7, Channels=4, H=198, W=252)
   where Channels = [Rainfall, DEM, Slope, TWI].
2. Extract spatial embeddings per frame using a Conv2D feature encoder whose
   terrain-facing weights (channels 1:4 of conv1, plus conv2/conv3/bn1-3) are
   transferred from a SpatialEncoder pretrained on terrain reconstruction
   (see src.models.train.pretrain_terrain_autoencoder) — this is the
   two-stage "autoencoder compresses terrain context, LSTM models the
   temporal dynamics" design described in the proposal, implemented as
   transfer learning rather than two disconnected networks.
3. Model non-linear temporal runoff dependencies across storm lifecycles using LSTM.
4. Decode final sequence output into predicted 2D spatial flood depth map (Batch, 1, 198, 252).

PERFORMANCE TARGET
------------------
Inference execution speed < 0.1 seconds per prediction on standard CPU.
Memory footprint < 200 MB.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.models.autoencoder import SpatialEncoder, SpatialDecoder


class FrameFeatureEncoder(SpatialEncoder):
    """
    Encodes a single frame (B, in_channels, H, W) to a feature vector.
    Structurally identical to SpatialEncoder — subclassed (rather than
    duplicated) so a terrain-pretrained SpatialEncoder's conv/bn weights
    can be loaded directly via load_pretrained_terrain_weights().
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, target_h: int = 198, target_w: int = 252, dropout: float = 0.1):
        super().__init__(in_channels=in_channels, latent_dim=hidden_dim, target_h=target_h, target_w=target_w, dropout=dropout)

    def load_pretrained_terrain_weights(self, terrain_encoder: SpatialEncoder, rain_channel_idx: int = 0) -> None:
        """
        Transfer conv2/bn2/conv3/bn3/bn1 fully from a terrain-only
        SpatialEncoder (in_channels=3: dem, slope, twi), and conv1's
        terrain-channel filters partially — the extra rainfall input
        channel keeps its own (random) initialization since the
        pretrained model never saw it.
        """
        with torch.no_grad():
            src_conv1 = terrain_encoder.conv1.weight.data  # (32, 3, 3, 3)
            terrain_channels = [i for i in range(self.conv1.in_channels) if i != rain_channel_idx]
            self.conv1.weight.data[:, terrain_channels, :, :] = src_conv1
            self.conv1.bias.data.copy_(terrain_encoder.conv1.bias.data)

            self.bn1.load_state_dict(terrain_encoder.bn1.state_dict())
            self.conv2.load_state_dict(terrain_encoder.conv2.state_dict())
            self.bn2.load_state_dict(terrain_encoder.bn2.state_dict())
            self.conv3.load_state_dict(terrain_encoder.conv3.state_dict())
            self.bn3.load_state_dict(terrain_encoder.bn3.state_dict())


class ConvLSTMSurrogateModel(nn.Module):
    """
    Hybrid Spatiotemporal Conv-LSTM Model for Real-Time Flood Depth Forecasting.
    """

    def __init__(
        self,
        in_channels: int = 4,
        seq_len: int = 7,
        hidden_dim: int = 128,
        target_h: int = 198,
        target_w: int = 252,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = FrameFeatureEncoder(in_channels=in_channels, hidden_dim=hidden_dim, target_h=target_h, target_w=target_w, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.decoder = SpatialDecoder(latent_dim=hidden_dim, out_channels=1, target_h=target_h, target_w=target_w, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (Batch, Seq_Len, Channels, H, W)
        Output: (Batch, 1, H, W) -> Flood depth map in meters
        """
        b, seq, c, h, w = x.shape

        # Encode each frame sequentially
        frame_features = []
        for t in range(seq):
            frame_t = x[:, t, :, :, :]  # (B, C, H, W)
            feat_t = self.encoder(frame_t)  # (B, hidden_dim)
            frame_features.append(feat_t)

        # Stack into sequence tensor (B, Seq_Len, hidden_dim)
        seq_feats = torch.stack(frame_features, dim=1)

        # Pass through LSTM head
        lstm_out, (h_n, c_n) = self.lstm(seq_feats)  # lstm_out: (B, Seq_Len, hidden_dim)

        # Take final time-step representation
        final_state = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Decode into 2D flood depth map
        depth_map = self.decoder(final_state)  # (B, 1, H, W)
        return depth_map
