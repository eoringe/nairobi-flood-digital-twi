"""
src.models.train_segmentation_v2
================================
Trains the flood-forecast U-Net on `segmentation_dataset_v2.npz`.

Replaces the previous training path, which collapsed to predicting all-zeros
because 488 of its 492 training scenes were entirely background (see
src/ingestion/build_segmentation_dataset_v2.py for the root cause).

WHY THIS ONE DOESN'T RUN OUT OF MEMORY
--------------------------------------
The old dataset stored 77 channels per sample, 70 of them duplicate copies of
the same 13 layers, giving a 6 GB file that had to be streamed off disk. Here
rainfall is 7 scalars per sample and terrain is one shared (6, H, W) array, so
the whole dataset is 1.9 MB. Everything is pushed to the GPU once at startup
and each batch is expanded to (13, H, W) on-device. There is no DataLoader, no
memory-mapping, and no host-to-device traffic in the training loop.

LOSS
----
The positive class is ~0.9% of pixels, so plain BCE is minimised by predicting
zero everywhere. Focal Tversky handles this directly: Tversky weights false
negatives above false positives (BETA > ALPHA), and the focal exponent
concentrates gradient on examples still being got wrong.

USAGE
-----
    python -m src.models.train_segmentation_v2
    python -m src.models.train_segmentation_v2 --epochs 80 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATA_FILE = Path("data/processed/arrays/segmentation_dataset_v2_forecast.npz")
OUT_DIR = Path("models/time_series")
MODEL_FILE = OUT_DIR / "segmentation_model_v2.pth"
METRICS_FILE = OUT_DIR / "segmentation_metrics_v2.json"

#: Per-channel divisors bringing every input to roughly unit scale.
#: dem/slope/twi/built_up are already normalised 0-1 upstream; HAND is in metres
#: and permanent water is a 0-100 occurrence percentage.
STATIC_SCALE = np.array([1.0, 1.0, 1.0, 50.0, 1.0, 100.0], dtype=np.float32)
#: Rainfall arrives as mm/day; 50 mm/day is an extreme daily total for Nairobi.
RAIN_SCALE = 50.0

# Focal Tversky parameters. ALPHA weights false negatives, BETA weights false
# positives, so ALPHA > BETA makes a missed flood cost more than a false alarm --
# the right asymmetry for early warning, and the reason precision will read lower
# than recall in the results.
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
TVERSKY_GAMMA = 0.75
#: Share of the loss taken by pos-weighted BCE. The Tversky term carries almost
#: no gradient on the ~86% of samples with no flood; BCE covers those.
BCE_WEIGHT = 0.5
#: Positive-class weight inside BCE, roughly the inverse of the ~0.9% base rate
#: damped toward 1 so it does not overwhelm the Tversky term.
BCE_POS_WEIGHT = 20.0


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Four-level U-Net. Decoder upsamples with bilinear interpolation resized to
    the encoder feature map, which sidesteps the output_padding arithmetic that
    made the previous ConvTranspose2d stack brittle at 198x252 (an odd height
    that does not halve cleanly four times).
    """

    def __init__(self, in_ch: int = 13, out_ch: int = 1, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.dec4 = ConvBlock(b * 16 + b * 8, b * 8)
        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    @staticmethod
    def _up(x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._up(b, e4))
        d3 = self.dec3(self._up(d4, e3))
        d2 = self.dec2(self._up(d3, e2))
        d1 = self.dec1(self._up(d2, e1))
        return self.head(d1)  # logits; loss applies sigmoid


class FocalTverskyLoss(nn.Module):
    """
    Pos-weighted BCE + batch-level Focal Tversky.

    Two details matter, and getting either wrong stalls training:

    1. Tversky is aggregated over the WHOLE BATCH, not per sample. About 86% of
       samples contain no flood at all; for those, per-sample Tversky has tp=0,
       so the numerator collapses to the smoothing constant while fp accumulates
       small probabilities over all 49,896 pixels. The index then sits near zero
       no matter what the model predicts, pinning the loss near 1 and drowning
       out the samples that do contain floods. Measured on this dataset, the
       per-sample form separated a perfect prediction from total collapse by
       only 0.08 - too flat to optimise. Batch aggregation widens that to 0.67.

    2. In the Tversky index ALPHA weights false negatives and BETA weights false
       positives. For flood early warning a missed flood costs more than a false
       alarm, so ALPHA > BETA. (An earlier version had these reversed.)

    BCE supplies a well-behaved per-pixel gradient on the all-negative samples,
    where the Tversky term alone carries almost no information.
    """

    def __init__(self, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, gamma=TVERSKY_GAMMA,
                 bce_weight=BCE_WEIGHT, pos_weight=BCE_POS_WEIGHT):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        tp = (p * target).sum()
        fn = ((1 - p) * target).sum()
        fp = (p * (1 - target)).sum()
        tversky = (tp + 1.0) / (tp + self.alpha * fn + self.beta * fp + 1.0)
        focal = (1 - tversky) ** self.gamma
        return self.bce_weight * bce + (1 - self.bce_weight) * focal


class GpuDataset:
    """
    Holds the whole dataset on-device and expands batches to (B, C, H, W).

    Every rainfall-derived feature is a scalar (CHIRPS was fetched as a single
    point, so rainfall has no spatial variation), and terrain is shared across
    all samples. Storing them that way and broadcasting on the GPU is what keeps
    the file at ~2 MB and removes host-to-device traffic from the training loop.
    """

    def __init__(self, npz, idx: np.ndarray, device):
        scale = npz["scalar_scale"]
        self.scalars = torch.from_numpy(npz["rain_seq"][idx] / scale).float().to(device)
        self.y = torch.from_numpy(npz["y"][idx]).to(device)  # uint8
        static = npz["static"] / STATIC_SCALE[:, None, None]
        self.static = torch.from_numpy(static).float().to(device)
        self.n = len(idx)
        self.k = self.scalars.shape[1]
        self.h, self.w = self.static.shape[-2:]
        self.in_channels = self.k + self.static.shape[0]

    def batch(self, sel):
        b = len(sel)
        maps = self.scalars[sel][:, :, None, None].expand(b, self.k, self.h, self.w)
        static_maps = self.static.unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([maps, static_maps], dim=1), self.y[sel].float()


@torch.no_grad()
def evaluate(model, ds, batch_size, threshold=0.5):
    model.eval()
    tp = fp = fn = 0.0
    for i in range(0, ds.n, batch_size):
        sel = torch.arange(i, min(i + batch_size, ds.n), device=ds.y.device)
        x, y = ds.batch(sel)
        pred = (torch.sigmoid(model(x)) > threshold).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--data", type=str, default=str(DATA_FILE),
                    help="dataset .npz (forecast = Model A, nwp = Model B)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output filenames, e.g. 'forecast' or 'nwp'")
    args = ap.parse_args()

    data_file = Path(args.data)
    suffix = f"_{args.tag}" if args.tag else ""
    model_file = OUT_DIR / f"segmentation_model_v2{suffix}.pth"
    metrics_file = OUT_DIR / f"segmentation_metrics_v2{suffix}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] device={device}")

    npz = np.load(data_file, allow_pickle=False)
    params = json.loads(str(npz["params"][0]))
    print(f"[DATA] {data_file.name}  label params: {params}")

    train = GpuDataset(npz, npz["train_idx"], device)
    val = GpuDataset(npz, npz["val_idx"], device)
    test = GpuDataset(npz, npz["test_idx"], device)
    print(f"[DATA] train={train.n}  val={val.n}  test={test.n}  grid={train.h}x{train.w}")
    print(f"[DATA] positive pixel rate: train={train.y.float().mean():.4f} "
          f"val={val.y.float().mean():.4f} test={test.y.float().mean():.4f}")
    if torch.cuda.is_available():
        print(f"[DATA] GPU memory held by data: {torch.cuda.memory_allocated()/1e6:.0f} MB")

    model = UNet(in_ch=train.in_channels, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = FocalTverskyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"[MODEL] U-Net base={args.base}, {n_params/1e6:.2f}M params\n")

    print(f"{'Epoch':<7}{'TrainLoss':>11}{'ValF1':>9}{'ValIoU':>9}{'ValPrec':>9}{'ValRec':>9}")
    print("-" * 54)

    best_f1, history = 0.0, []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(train.n, device=device)
        total = 0.0
        for i in range(0, train.n, args.batch_size):
            sel = perm[i:i + args.batch_size]
            x, y = train.batch(sel)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(sel)
        train_loss = total / train.n
        scheduler.step()

        m = evaluate(model, val, args.batch_size)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **m})

        tag = ""
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), model_file)
            tag = "  <- saved"
        print(f"{epoch+1:<7}{train_loss:>11.4f}{m['f1']:>9.4f}{m['iou']:>9.4f}"
              f"{m['precision']:>9.4f}{m['recall']:>9.4f}{tag}")

    model.load_state_dict(torch.load(model_file))
    test_m = evaluate(model, test, args.batch_size)

    print("\n" + "=" * 54)
    print("TEST (held-out storm seasons, never seen in training)")
    print("=" * 54)
    for k in ("f1", "iou", "precision", "recall"):
        print(f"  {k:<10}{test_m[k]:.4f}")

    metrics_file.write_text(json.dumps({
        "label_params": params,
        "model": {"base": args.base, "n_params": n_params,
                  "in_channels": train.in_channels},
        "train": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        "history": history,
        "best_val_f1": best_f1,
        "test_metrics": test_m,
    }, indent=2))
    print(f"\n[SAVE] {model_file}\n[SAVE] {metrics_file}")


if __name__ == "__main__":
    main()
