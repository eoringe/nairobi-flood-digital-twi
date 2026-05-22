# Local Machine Hardware & Memory Strategy

## 1. Active Resource Profile
*   **Target Machine:** Local Development Workstation
*   **Memory Limit:** 16GB System RAM
*   **Optimization Mandate:** Absolute memory efficiency during predictive model execution, tensor evaluation, and grid array processing.

## 2. Mandatory Memory Guardrails
*   **Spatial Domain Chunking:** Catchment vectors and satellite data surfaces must be explicitly sliced into discrete subdomains (e.g., $198 \times 252$ structural pixel matrix windows) rather than loading the complete expanse of Nairobi simultaneously[cite: 2].
*   **Disk-Stream Data Loader loops:** Ingested datasets must be processed via streaming chunks using PyTorch DataLoaders with `pin_memory=False` and conservative batch boundaries (e.g., batch sizes of 32 or 64). Global variables must never hold raw target images natively.
*   **Static Cell Masking:** Topographical wetness profiles and baseline elevation thresholds must filter out permanent non-flooded areas prior to model training[cite: 2]. Tensors will optimize processing by mathematically operating only on coordinate matrices with true flood hazard potential[cite: 2].
*   **Offloaded Frontend Pipelines:** The operational browser client will render mapping outputs client-side via hardware-accelerated WebGL layers using Pydeck/Deck.gl, freeing backend system RAM[cite: 1, 4].