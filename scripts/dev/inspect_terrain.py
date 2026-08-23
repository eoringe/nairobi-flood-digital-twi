"""Analyze DEM, Slope, TWI features at key Nairobi hotspot coordinates."""
import numpy as np

terrain = np.load("data/processed/arrays/static_terrain_features.npy")
dem = terrain[0]
slope = terrain[1]
twi = terrain[2]

lats = np.linspace(-1.23, -1.35, 198)
lons = np.linspace(36.72, 36.90, 252)

locations = {
    "Globe Roundabout & Kipande Rd": (-1.2787, 36.8213),
    "Mathare River Corridor": (-1.2580, 36.8580),
    "Kibera & Ngong River Basin": (-1.3120, 36.7880),
    "Mukuru Kwa Njenga Basin": (-1.3100, 36.8780),
    "South C & Muhoho Avenue": (-1.3210, 36.8320),
    "Kariakor & Racecourse Rd": (-1.2820, 36.8370),
}

print(f"{'Hotspot Name':<32} | {'Row':<4} {'Col':<4} | {'DEM':<6} | {'Slope':<6} | {'TWI':<6}")
print("-" * 65)
for name, (lat_t, lon_t) in locations.items():
    rc = np.abs(lats - lat_t).argmin()
    cc = np.abs(lons - lon_t).argmin()
    
    # Take 3x3 patch average around exact point
    dem_val = dem[rc-1:rc+2, cc-1:cc+2].mean()
    slope_val = slope[rc-1:rc+2, cc-1:cc+2].mean()
    twi_val = twi[rc-1:rc+2, cc-1:cc+2].mean()
    
    print(f"{name:<32} | {rc:<4} {cc:<4} | {dem_val:<6.3f} | {slope_val:<6.3f} | {twi_val:<6.3f}")
