"""Audit: compare terrain-driven vs hardcoded-river contribution."""
import numpy as np

terrain = np.load("data/processed/arrays/static_terrain_features.npy")
dem = terrain[0]
slope = terrain[1]
twi = terrain[2]

# Pure terrain-driven pooling (what the DEM says)
slope_safe = np.maximum(slope, 0.02)
terrain_pooling = (twi ** 1.2) * np.exp(-2.5 * slope_safe) * np.exp(-1.5 * dem)
t_max = np.max(terrain_pooling)
if t_max > 0:
    terrain_pooling /= t_max

# Check where terrain says water pools (top 10% of cells)
thresh = np.percentile(terrain_pooling, 90)
terrain_flood_cells = np.sum(terrain_pooling > thresh)
print("Terrain-driven pooling:")
print("  Cells above 90th percentile:", terrain_flood_cells)
print("  Max pooling value:", terrain_pooling.max())
print("  Mean pooling value:", terrain_pooling.mean())
print()

# Check the neural network contribution
from src.models.predict import FloodSurrogatePredictor
predictor = FloodSurrogatePredictor()

# Run prediction at 45mm
res = predictor.predict_scenario(rainfall_mm_day=45.0)
grid = res["depth_grid"]

print("At 45mm rainfall:")
print("  Grid max depth:", grid.max())
print("  Cells with depth > 0.2m:", np.sum(grid > 0.2))
print("  Cells with depth > 1.0m:", np.sum(grid > 1.0))
print("  Cells with depth > 1.8m:", np.sum(grid > 1.8))
print()

# Check what the hydro mask looks like
hm = predictor.hydro_mask
print("Hydro mask stats:")
print("  Mean:", hm.mean())
print("  Cells above 0.5:", np.sum(hm > 0.5))
print("  Cells above 0.3:", np.sum(hm > 0.3))
print("  Cells above 0.1:", np.sum(hm > 0.1))
print("  Total cells:", hm.size)
print("  Fraction above 0.3:", np.sum(hm > 0.3) / hm.size)
print()

# Compare terrain-only vs hydro_mask
print("Terrain pooling mean:", terrain_pooling.mean())
print("Hydro mask mean:", hm.mean())
print("Correlation between terrain pooling and hydro mask:", np.corrcoef(terrain_pooling.flatten(), hm.flatten())[0, 1])
