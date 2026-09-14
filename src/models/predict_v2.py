"""
src.models.predict_v2
=====================
FloodPredictor - inference for the deployed U-Net (Model B).

WHAT THIS REPLACES
------------------
`src.models.predict.FloodSurrogatePredictor` loads `conv_lstm_surrogate.pth`,
a ConvLSTM trained on the dataset whose rainfall join was defective: six of 703
samples were flood-positive and every label was a scene-level flag broadcast
across all 49,896 cells. Predictions from that model cannot be defended and it
must not back a demonstration.

This module loads `segmentation_model_v2_nwp_drainage.pth` - Model B trained on
drainage-based labels, test F1 0.937 on held-out storm seasons.

EXTENT, NOT DEPTH
-----------------
The output is a per-cell flood *probability* in [0, 1]. It is not a depth in
metres and must never be displayed as one. Satellites cannot measure water
depth; the earlier depth figures came from a physics formula rather than from
observation, which is why depth regression was abandoned (RESULTS.md 4.9).

Storm intensity is expressed as flooded *area*: a 30 mm storm inundates roughly
1.4% of the grid, a 120 mm storm roughly 12.3%. Bigger storm, more ground - not
deeper water.

SCENARIO INPUTS
---------------
Model B expects 21 channels: 14 scalar features broadcast to maps, plus 7 static
terrain layers. A dashboard user supplies only an accumulated rainfall depth, so
the remaining scalars are filled as follows, and every assumption is returned in
the result dict so the interface can disclose it:

  rain_fcst_t+0..2   the scenario total, distributed across the forecast window
  rain_t-7..t-1      antecedent rainfall; defaults to the seasonal mean unless
                     the caller supplies an observed series
  doy_sin, doy_cos   from the scenario date, defaulting to today
  rain_sum_14d/30d   extrapolated from the antecedent daily mean

USAGE
-----
    from src.models.predict_v2 import FloodPredictor
    p = FloodPredictor()
    result = p.predict(rainfall_mm=62.3)
    result["probability_grid"]    # (198, 252) float32 in [0, 1]
    result["flooded_fraction"]    # share of grid above threshold
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.models.train_segmentation_v2 import UNet, STATIC_SCALE

MODELS_DIR = Path("models/time_series")
DATA_DIR = Path("data/processed/arrays")

#: Model B on drainage labels - the configuration the system deploys.
DEFAULT_WEIGHTS = MODELS_DIR / "segmentation_model_v2_nwp_drainage.pth"
DEFAULT_DATASET = DATA_DIR / "segmentation_dataset_v2_nwp_drainage.npz"
DEFAULT_METRICS = MODELS_DIR / "segmentation_metrics_v2_nwp_drainage.json"
#: Isotonic calibration fitted on the validation seasons (src/models/calibrate_v2.py).
DEFAULT_CALIBRATION = MODELS_DIR / "calibration_v2_nwp_drainage.json"

#: Probability above which a cell is reported flooded.
FLOOD_THRESHOLD = 0.5

#: Mean daily rainfall across Nairobi storm seasons, used when the caller gives
#: no antecedent series. Measured from the CHIRPS record, not assumed.
DEFAULT_ANTECEDENT_MM_DAY = 3.74


class FloodPredictor:
    """
    Inference wrapper around the trained U-Net.

    Attributes
    ----------
    model_version : str
        Filename of the loaded weights, recorded against every prediction so a
        displayed result can be traced to the model that produced it.
    metrics : dict
        Held-out test metrics for the loaded model, so the interface can state
        the accuracy of what it is showing.
    """

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS,
                 dataset_path: Path = DEFAULT_DATASET) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = weights_path.name
        self.metrics: dict = {}
        self.label_params: dict = {}
        self._cache: dict[tuple[int, str], np.ndarray] = {}
        self._lock = threading.Lock()

        # The static layers and scalar scaling are read from the dataset the
        # model was trained on, so inference cannot silently diverge from
        # training in channel order or normalisation.
        npz = np.load(dataset_path, allow_pickle=False)
        self.static = npz["static"] / STATIC_SCALE[:, None, None]
        self.scalar_scale = npz["scalar_scale"]
        self.scalar_names = [str(n) for n in npz["scalar_names"]]
        self.susceptibility = npz["susceptibility"]
        self.label_params = json.loads(str(npz["params"][0]))
        self.n_static, self.h, self.w = self.static.shape
        self.n_scalars = len(self.scalar_names)

        self.static_t = torch.from_numpy(self.static).float().unsqueeze(0).to(self.device)

        in_ch = self.n_scalars + self.n_static
        self.model = UNet(in_ch=in_ch, base=32).to(self.device)
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"FloodPredictor: loaded {weights_path.name} ({in_ch} channels)")
        else:
            logger.error(
                f"FloodPredictor: {weights_path} not found - predictions will be "
                f"from an untrained network. Copy the trained weights into "
                f"{MODELS_DIR}/ before demonstrating."
            )
        self.model.eval()

        if DEFAULT_METRICS.exists():
            self.metrics = json.load(open(DEFAULT_METRICS)).get("test_metrics", {})

        # Raw outputs are overconfident in the 0.6-0.9 range (LIMITATIONS.md
        # section 10). The calibration map is monotone, so it changes the
        # stated percentages without reordering which cells are most at risk.
        self.calibration: dict | None = None
        if DEFAULT_CALIBRATION.exists():
            self.calibration = json.load(open(DEFAULT_CALIBRATION))
            self._cal_x = np.asarray(self.calibration["x"], dtype=np.float32)
            self._cal_y = np.asarray(self.calibration["y"], dtype=np.float32)
            self.model_version += " + isotonic calibration"
            logger.info(f"FloodPredictor: calibrated probabilities ({self.calibration['fitted_on']})")
        else:
            logger.warning("FloodPredictor: no calibration file - showing raw, overconfident "
                           "probabilities. Run `python -m src.models.calibrate_v2`.")

    # ------------------------------------------------------------ features --
    def _build_scalars(self, rainfall_mm: float, antecedent: np.ndarray | None,
                       scenario_date: date | None) -> tuple[np.ndarray, dict]:
        """Assemble the scalar feature vector in the exact order used in training."""
        forecast_days = int(self.label_params.get("forecast_days", 3))
        seq_len = int(self.label_params.get("seq_len", 7))
        d = scenario_date or date.today()

        if antecedent is None:
            antecedent = np.full(seq_len, DEFAULT_ANTECEDENT_MM_DAY, dtype=np.float32)
            antecedent_source = f"seasonal mean ({DEFAULT_ANTECEDENT_MM_DAY} mm/day)"
        else:
            antecedent = np.asarray(antecedent, dtype=np.float32)[-seq_len:]
            antecedent_source = "caller-supplied observations"

        doy = d.timetuple().tm_yday
        feats: list[float] = list(antecedent)
        feats += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
        mean_ante = float(antecedent.mean())
        feats += [mean_ante * 14.0, mean_ante * 30.0]

        if "rain_fcst_t+0" in self.scalar_names:      # Model B receives the forecast
            feats += [rainfall_mm / forecast_days] * forecast_days

        assert len(feats) == self.n_scalars, (
            f"built {len(feats)} scalars but the model expects {self.n_scalars}"
        )
        assumptions = {
            "antecedent_source": antecedent_source,
            "scenario_date": d.isoformat(),
            "forecast_days": forecast_days,
        }
        return np.asarray(feats, dtype=np.float32), assumptions

    # ------------------------------------------------------------- predict --
    @torch.no_grad()
    def predict(self, rainfall_mm: float, antecedent: np.ndarray | None = None,
                scenario_date: date | None = None,
                threshold: float = FLOOD_THRESHOLD) -> dict:
        """
        Predict flood extent for an accumulated rainfall depth.

        Parameters
        ----------
        rainfall_mm : float
            Rainfall accumulated over the forecast window, in millimetres.
        antecedent : array-like, optional
            Observed daily rainfall for the preceding week. The seasonal mean is
            used when omitted.
        threshold : float
            Probability above which a cell is reported flooded.

        Returns
        -------
        dict with `probability_grid` (H, W) in [0, 1], `flooded_fraction`,
        `flooded_area_km2`, the model version, its test metrics, and the
        assumptions used to fill unspecified inputs.
        """
        t0 = time.perf_counter()

        scalars, assumptions = self._build_scalars(rainfall_mm, antecedent, scenario_date)
        scaled = torch.from_numpy(scalars / self.scalar_scale).float().to(self.device)
        maps = scaled[None, :, None, None].expand(1, self.n_scalars, self.h, self.w)
        x = torch.cat([maps, self.static_t], dim=1)

        probability = torch.sigmoid(self.model(x))[0, 0].cpu().numpy().astype(np.float32)
        if self.calibration is not None:
            probability = np.interp(probability, self._cal_x, self._cal_y).astype(np.float32)
        assumptions["calibrated"] = self.calibration is not None
        return self.postprocess_results(
            probability, rainfall_mm, threshold, assumptions,
            latency_sec=time.perf_counter() - t0,
        )

    def postprocess_results(self, probability: np.ndarray, rainfall_mm: float,
                            threshold: float, assumptions: dict,
                            latency_sec: float) -> dict:
        """Summarise a probability field into the values the interface displays."""
        binary = probability > threshold
        cell_km2 = 0.067 * 0.079          # ~67 m x 79 m at this latitude

        return {
            "probability_grid": probability,
            "flooded_mask": binary,
            "flooded_fraction": float(binary.mean()),
            "flooded_area_km2": round(float(binary.sum()) * cell_km2, 2),
            "mean_probability": float(probability.mean()),
            "peak_probability": float(probability.max()),
            "rainfall_mm": rainfall_mm,
            "threshold": threshold,
            "model_version": self.model_version,
            "test_metrics": self.metrics,
            "assumptions": assumptions,
            "latency_sec": round(latency_sec, 4),
            "units": "dimensionless probability in [0, 1] - NOT a depth in metres",
        }

    # --------------------------------------------------------------- cache --
    def probability_for(self, rainfall_mm: float, scenario_date: date | None = None) -> np.ndarray:
        """
        Probability grid for a rainfall depth, cached.

        One CPU inference costs ~1.5 s and batching does not reduce it, so a
        12-hour outlook computed directly would take ~20 s. Rainfall is rounded
        to 1 mm (the response changes by at most ~0.5 percentage points of area
        per mm, near the 30 mm onset) and the date is part of the key, because
        season shifts the result near that onset.

        Antecedent rainfall uses the seasonal mean. Measured sensitivity: at
        50 mm, antecedent 0 vs 20 mm/day moves flooded area from 4.04% to 4.09%.
        """
        d = scenario_date or date.today()
        key = (int(round(max(0.0, rainfall_mm))), d.isoformat())
        grid = self._cache.get(key)
        if grid is not None:
            return grid
        # Serialised: concurrent forward passes only contend for the same cores.
        with self._lock:
            grid = self._cache.get(key)
            if grid is None:
                grid = self.predict(rainfall_mm=float(key[0]), scenario_date=d)["probability_grid"]
                self._cache[key] = grid
        return grid

    def is_cached(self, rainfall_mm: float, scenario_date: date | None = None) -> bool:
        d = scenario_date or date.today()
        return (int(round(max(0.0, rainfall_mm))), d.isoformat()) in self._cache

    # ------------------------------------------------------------ validate --
    def validate_spatial_plausibility(self, probability: np.ndarray,
                                      threshold: float = FLOOD_THRESHOLD) -> dict:
        """
        Check that predicted flooding concentrates on susceptible ground.

        A prediction that ignores terrain and spreads uniformly would score the
        same mean susceptibility as the grid at large. A plausible one should
        sit well above it. This is a sanity check on an individual prediction,
        not a substitute for the validation in RESULTS.md 4.8.
        """
        mask = probability > threshold
        if not mask.any():
            return {"plausible": False, "reason": "no cells predicted flooded"}

        flooded_susc = float(self.susceptibility[mask].mean())
        grid_susc = float(self.susceptibility.mean())
        ratio = flooded_susc / max(grid_susc, 1e-9)
        return {
            "plausible": ratio > 1.5,
            "flooded_mean_susceptibility": round(flooded_susc, 4),
            "grid_mean_susceptibility": round(grid_susc, 4),
            "concentration_ratio": round(ratio, 2),
            "reason": ("flooding concentrates on susceptible ground"
                       if ratio > 1.5 else
                       "flooding is not concentrated on susceptible ground"),
        }
