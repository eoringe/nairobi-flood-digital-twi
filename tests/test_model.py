"""
tests.test_model
================
Unit tests for Spatial Autoencoder, ConvLSTM Surrogate model, and sub-second inference engine.
"""

import torch
import numpy as np

from src.models.autoencoder import SpatialAutoencoder
from src.models.lstm_surrogate import ConvLSTMSurrogateModel
from src.models.predict import FloodSurrogatePredictor


def test_spatial_autoencoder_shape():
    """Verify Autoencoder forward pass shape (B, 3, 198, 252) -> (B, 1, 198, 252)."""
    model = SpatialAutoencoder(in_channels=3, out_channels=1, latent_dim=128, target_h=198, target_w=252)
    dummy_x = torch.randn(2, 3, 198, 252)
    out = model(dummy_x)
    assert out.shape == (2, 1, 198, 252)


def test_conv_lstm_surrogate_shape():
    """Verify ConvLSTM Surrogate forward pass shape (B, 7, 4, 198, 252) -> (B, 1, 198, 252)."""
    model = ConvLSTMSurrogateModel(in_channels=4, seq_len=7, hidden_dim=128, target_h=198, target_w=252)
    dummy_seq = torch.randn(2, 7, 4, 198, 252)
    out = model(dummy_seq)
    assert out.shape == (2, 1, 198, 252)


def test_sub_second_inference_latency():
    """Verify surrogate inference latency is strictly < 1.0 seconds per forecast scenario."""
    predictor = FloodSurrogatePredictor()
    res = predictor.predict_scenario(rainfall_mm_day=90.0)
    assert res["latency_sec"] < 1.0
    assert res["max_depth_m"] >= 0.0
    assert res["depth_grid"].shape == (198, 252)
