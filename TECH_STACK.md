# Technical Stack & Dependency Blueprint

## 1. Geospatial Processing & Ingestion Pipeline
*   **earthengine-api:** Python interface for cloud-based remote sensing data, Sentinel-1 SAR imagery time-series processing, and elevation ingestion.
*   **geopandas & shapely:** High-performance vector geometry handling for grid-clipping, watershed boundaries, and structural coordinates.
*   **xarray & netcdf4:** Multi-dimensional array handling for temporal, gridded gauge-precipitation datasets (CHIRPS/NLDAS style mapping).

## 2. Machine Learning Core (Surrogate Modeling)
*   **torch (PyTorch):** Deep learning framework for setting up convolutional spatial map encoders/decoders and recursive time-series forecasting layers (LSTM/NARX).
*   **scikit-learn:** Data normalization, min-max target scaling (e.g., target ranges 0.2 to 1.0 to handle unflooded cell masking), and cross-validation matrix tools.
*   **networkx:** Constructing Directed Acyclic Graphs (DAG) to implement object-based structural variation and tracking of connected water bodies over consecutive timesteps.

## 3. Web UI & 3D Geospatial Twin Canvas
*   **dash (Plotly Dash):** Overarching reactive Python framework for structural dashboard grids, state callbacks, and multi-page application lifecycle.
*   **dash-bootstrap-components:** Grid system layout engine to ensure mobile-responsive viewport UI management.
*   **pydeck:** The high-performance hardware-accelerated WebGL core used to map 3D geospatial overlays, LOD2 building tile sets, and dynamic flood inundation layers seamlessly above a basemap service string.

## 4. Systems Infrastructure
*   **python-dotenv:** Safe environment variable configuration handling for access tokens and project routing paths.