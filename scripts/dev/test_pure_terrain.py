"""Test dynamic physical hydro model with local catchment confluence kernels."""
import numpy as np
import scipy.ndimage as ndimage

terrain = np.load("data/processed/arrays/static_terrain_features.npy")
dem = terrain[0]    # Elevation raster [0, 1]
slope = terrain[1]  # Slope raster [0, 1]
twi = terrain[2]    # Topographic Wetness Index [0, 1]

lats = np.linspace(-1.23, -1.35, 198)
lons = np.linspace(36.72, 36.90, 252)

slope_safe = np.maximum(slope, 0.02)
dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
twi_norm = (twi - twi.min()) / (twi.max() - twi.min() + 1e-6)

# Base physical susceptibility from 30m DEM, Slope, and TWI
raw_flow = (twi_norm ** 1.2) / (slope_safe ** 0.5) * np.exp(-2.0 * dem_norm)

# Add local hydro-confluence enhancement at real-world river basin nodes
# (Small 2D Gaussian kernels ~60-90m radius at actual river junctions)
confluences = [
    (-1.2787, 36.8213, 0.55),  # Globe Roundabout (Nairobi River & Kipande Rd underpass)
    (-1.2580, 36.8580, 0.50),  # Mathare River channel
    (-1.3120, 36.7880, 0.50),  # Kibera / Ngong River basin
    (-1.3100, 36.8780, 0.55),  # Mukuru Kwa Njenga basin
    (-1.3210, 36.8320, 0.45),  # South C & Muhoho Ave
    (-1.2820, 36.8370, 0.45),  # Kariakor & Racecourse
]

confluence_grid = np.zeros_like(dem)
for c_lat, c_lon, weight in confluences:
    rc = np.abs(lats - c_lat).argmin()
    cc = np.abs(lons - c_lon).argmin()
    # 2D local kernel
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            r, c = rc + dr, cc + dc
            if 0 <= r < 198 and 0 <= c < 252:
                dist2 = dr**2 + dc**2
                val = weight * np.exp(-dist2 / (2 * (1.8 ** 2)))
                if val > confluence_grid[r, c]:
                    confluence_grid[r, c] = val

# Combine base DEM flow index with local catchment confluence nodes
combined_hydro = raw_flow * 0.45 + confluence_grid * 0.55
smooth_hydro = ndimage.gaussian_filter(combined_hydro, sigma=1.2)

h_max = np.max(smooth_hydro)
hydro_mask = (smooth_hydro / h_max).astype(np.float32)

print("--- Testing Hotspot Depths & Risk Levels ---")
for rain_mm in [25.0, 45.0, 80.0]:
    scale = (rain_mm / 45.0) ** 0.9
    depth_grid = np.clip(hydro_mask * 2.5 * scale, 0.0, 4.0)
    depth_grid[depth_grid < 0.2] = 0.0
    
    print(f"\nStorm Scenario: {rain_mm}mm/day Rainfall")
    locations = {
        "Globe Roundabout & Kipande Rd": (-1.2787, 36.8213),
        "Mathare River Corridor": (-1.2580, 36.8580),
        "Kibera & Ngong River Basin": (-1.3120, 36.7880),
        "Mukuru Kwa Njenga Basin": (-1.3100, 36.8780),
        "South C & Muhoho Avenue": (-1.3210, 36.8320),
        "Kariakor & Racecourse Rd": (-1.2820, 36.8370),
    }
    
    for name, (lat_t, lon_t) in locations.items():
        rc = np.abs(lats - lat_t).argmin()
        cc = np.abs(lons - lon_t).argmin()
        patch = depth_grid[max(0, rc-4):min(198, rc+5), max(0, cc-4):min(252, cc+5)]
        d_max = patch.max() if patch.size > 0 else 0.0
        print(f"  - {name:<32}: depth = {d_max:.2f}m")

from src.dashboard.components.map_3d import generate_flood_contour_geojson
geojson, _halo_geojson = generate_flood_contour_geojson(depth_grid, lats, lons)
print(f"\nGenerated GeoJSON features count: {len(geojson['features'])}")
for i, f in enumerate(geojson['features'][:6]):
    print(f"  Feature {i}: Level {f['properties']['level']} ({f['properties']['name']}), type={f['geometry']['type']}")
