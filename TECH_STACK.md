# Tech Stack

What the Nairobi Flood Digital Twin actually runs on.

---

## Frontend

| Tool | Used for |
|---|---|
| **Plotly Dash** | The web dashboard — layout, sliders, panels |
| **Dash Bootstrap Components** | Page grid and styling |
| **Pydeck (deck.gl / WebGL)** | The 3D map: flood polygons and building extrusions |
| **Plotly** | The rainfall-response chart |
| **CARTO Dark** | Basemap tiles |

The map updates in place when the slider moves — only the flood shapes are sent, not the whole map.

---

## Backend

| Tool | Used for |
|---|---|
| **Python 3.13** | Everything |
| **Flask** | Web server (built into Dash) |
| **PyTorch** | The U-Net flood model |
| **NumPy / SciPy** | Arrays, smoothing, distance maps |
| **Shapely / Rasterio / GeoPandas** | Turning predictions into map polygons |
| **SQLite** | Database — 19 tables, no install needed |
| **Google Colab (T4 GPU)** | Model training |

**Model:** U-Net, 7.85 M parameters. Takes rainfall + terrain, returns the probability each 70 m cell floods. Test F1 **0.937**.

**Database note:** SQLite is used so the project runs with nothing installed. The same SQL works on PostgreSQL/PostGIS — set `DATABASE_URL` to switch.

---

## Data acquisition

| Data | Source | How we got it |
|---|---|---|
| Daily rainfall, 2015–2026 | **CHIRPS 2.0** (UC Santa Barbara) | Downloaded NetCDF files |
| Elevation | **SRTM** | Downloaded tiles, merged into one map |
| Slope, wetness (TWI) | Derived from SRTM | Calculated locally |
| Height above drainage, flow accumulation | **MERIT Hydro** | Google Earth Engine |
| Built-up land | **ESA WorldCover** | Google Earth Engine |
| Permanent water | **JRC Global Surface Water** | Google Earth Engine |
| Building footprints | **Google Open Buildings** | Downloaded CSV |
| Place names (332) | **OpenStreetMap** | Overpass API, cached to file |
| Live weather | **Open-Meteo** | Called from the dashboard |

Everything is clipped to one 198 × 252 grid over Nairobi (~70 m per cell).

**Tried and dropped:** Sentinel-1 radar and Sentinel-2 optical imagery. Radar picked up dry surfaces instead of water; optical saw no floods at all because of clouds and small street-scale flooding. See `LIMITATIONS.md` §4.

---

## Honest notes

- **Building heights are estimated** from footprint size — the source data has no heights.
- **Rainfall is one value for all of Nairobi** — CHIRPS was fetched as a single point, not a grid.
- **Not built:** the Kafka streams, Redis cache and separate API gateway shown in the architecture diagram. At this scale they would have nothing to do; they are future work.
- **Credit:** place names © OpenStreetMap contributors (ODbL).
