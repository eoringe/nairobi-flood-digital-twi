"""
src.models.segmentation
=======================
U-Net segmentation model for binary flood probability prediction.

Input:  7-day Sentinel-1 SAR time-series + 7 static predictor channels
        → (14 channels, 198×252 spatial)
Output: Flood probability map (0-1) → (1 channel, 198×252)

Architecture: Standard U-Net with 4 downsampling levels
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net segmentation network.

    Args:
        in_channels: Number of input channels (7 SAR + 7 static = 14)
        out_channels: Number of output channels (1 for flood probability)
        features: Base number of features (64)
    """
    def __init__(self, in_channels: int = 14, out_channels: int = 1, features: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = DoubleConv(features * 2, features)

        # Output
        self.out = nn.Conv2d(features, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)
        x = self.pool4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Output: flood probability (0-1)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


class SegmentationLoss(nn.Module):
    """
    Combined loss for flood segmentation: BCE + Dice.

    Weights:
        bce_weight: 0.7 (handles calibration)
        dice_weight: 0.3 (handles class imbalance)
    """
    def __init__(self, bce_weight: float = 0.7, dice_weight: float = 0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss: 1 - (2|X∩Y|) / (|X| + |Y|)"""
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = 1.0 - (2 * intersection + smooth) / (union + smooth)
        return dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def create_segmentation_model(device: torch.device) -> tuple[UNet, SegmentationLoss]:
    """Factory function to create model and loss."""
    model = UNet(in_channels=14, out_channels=1, features=64).to(device)
    criterion = SegmentationLoss(bce_weight=0.7, dice_weight=0.3)
    return model, criterion
