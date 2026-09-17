# Tech Stack

What the Nairobi Flood Digital Twin actually runs on.

---

## Frontend

| Tool | Used for |
|---|---|
| **Plotly Dash** | The web dashboard: layout, sliders, panels |
| **Dash Bootstrap Components** | Page grid and styling |
| **Pydeck (deck.gl / WebGL)** | The 3D map: flood polygons, buildings, routes |
| **Plotly** | Rainfall outlook and rainfall-response charts |
| **CARTO Dark / Voyager** | Basemap tiles for dark and light mode |

- The map updates in place. When the forecast hour, rainfall or route changes, only those layers are sent, not the whole map.
- **Light and dark mode** share one set of colour tokens. The dashboard follows the computer's setting until the user picks one, and remembers the choice.

---

## Backend

| Tool | Used for |
|---|---|
| **Python 3.13** | Everything |
| **Flask** | Web server (built into Dash) |
| **PyTorch** | The U-Net flood model |
| **NumPy / SciPy** | Arrays, smoothing, and route finding (compiled Dijkstra) |
| **Shapely / Rasterio / GeoPandas** | Turning predictions into map polygons |
| **SQLite** | Database: 19 tables, no install needed |
| **Google Colab (T4 GPU)** | Model training |

**Model:** U-Net, 7.85 M parameters. Takes rainfall + terrain and returns the probability that each 70 m cell floods. Test F1 **0.937** (0.923–0.945 across test seasons). Probabilities are **calibrated**, so "70%" means about 70% (calibration error 0.092 → 0.020).

**People at risk:** WorldPop 2025 population living in cells predicted flooded, an upper-bound exposure count.

**Flood outlook:** hourly rainfall → 72-hour total for each of the next 12 hours → model → the first hour each place floods ("expect moderate flooding in Mathare in about 2 hours").

**Location search:** type any place, road, landmark or coordinates. It searches an offline index first (instant, works without internet), then the Photon geocoder for rarer names. Places outside the mapped area are listed but marked as not routable.

**Flood-aware routing:** a graph of Nairobi's roads (167k junctions). Roads under predicted flooding cost more (moderate ×3, high ×25, critical closed). The route re-plans when the forecast changes. About 0.1–0.5 s per route.

**Database note:** SQLite is used so the project runs with nothing installed. The same SQL works on PostgreSQL/PostGIS; set `DATABASE_URL` to switch.

---

## Data acquisition

| Data | Source | How we got it |
|---|---|---|
| Daily rainfall, 2015–2026 | **CHIRPS 2.0** (UC Santa Barbara) | Downloaded NetCDF files |
| Hourly rainfall forecast | **Open-Meteo** forecast API | Called by the dashboard, refreshed every 20 min |
| Hourly rainfall, past storms | **Open-Meteo** archive API | Used for the April 2024 replays |
| Elevation | **SRTM** | Downloaded tiles, merged into one map |
| Slope, wetness (TWI) | Derived from SRTM | Calculated locally |
| Height above drainage, flow accumulation | **MERIT Hydro** | Google Earth Engine |
| Built-up land | **ESA WorldCover** | Google Earth Engine |
| Permanent water | **JRC Global Surface Water** | Google Earth Engine |
| Building footprints | **Google Open Buildings** | Downloaded CSV |
| Place names (332) | **OpenStreetMap** | Overpass API, cached to file |
| Road network (29,473 roads) | **OpenStreetMap** | Overpass API, cached to `nairobi_roads.json.gz` |
| Searchable locations (~19,800: shops, schools, hospitals, stages, roads) | **OpenStreetMap** | Overpass API, cached to `nairobi_search_index.json.gz` |
| Location search fallback | **Photon** (komoot) | Called only when the offline index has too few matches |
| Population, 2025 (100 m) | **WorldPop** (CC BY 4.0) | Downloaded GeoTIFF, resampled to the model grid |

Everything is clipped to one 198 × 252 grid over Nairobi (~70 m per cell).

**Tried and dropped:** Sentinel-1 radar and Sentinel-2 optical imagery. Radar picked up dry surfaces instead of water. Optical saw no floods at all, because of clouds and small street-scale flooding. See `LIMITATIONS.md` §4.

---

## Honest notes

- **"In 2 hours" comes from the rainfall forecast,** not from simulating water flow. The model itself is daily. See `LIMITATIONS.md` §14.
- **Routes are lower-risk suggestions,** not verified street by street, and there is no live traffic. See §13.
- **Building heights are estimated** from footprint size; the source data has no heights.
- **Rainfall is one value for all of Nairobi.** CHIRPS and Open-Meteo are both fetched as a single point, not a grid.
- **Not built:** the Kafka streams, Redis cache and separate API gateway shown in the architecture diagram. At this scale they would have nothing to do; they are future work.
- **Search privacy:** a query the offline index cannot answer is sent to Photon (photon.komoot.io). Only the typed text is sent.
- **Credit:** place names and roads © OpenStreetMap contributors (ODbL).
