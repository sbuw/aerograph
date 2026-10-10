# Planetary Climate Teleconnection Graph & Spectral Wavelets (2D)

## 1. Overview
Real-time geophysical web engine modeling Earth's atmospheric circulation as a dynamic weighted graph. 

The atmosphere is discretized into a 2D coordinate grid. The backend ingests surface pressure and 500 hPa geopotential height fields, computes lagged cross-correlation matrices across spatial nodes, identifies global teleconnections (e.g., NAO, ENSO), and applies Spectral Graph Wavelets (SGWT) over the normalized graph Laplacian to isolate energetic anomaly cascades.

The front end renders a 2D canvas/map featuring teleconnection arcs, anomaly heatmaps, and spectral graph metrics.

## 2. Core Capabilities
* **Automated Data Pipeline:** Scheduled ingestion of global grids from Open-Meteo / NOAA GFS every 6 hours.
* **Matrix Engine:** Vectorized Pearson cross-correlation across lag windows $\tau \in [0, 7]$ days with significance filtering ($p < 0.01$).
* **Spectral Decomposition:** Normalized graph Laplacian formulation $L = I - D^{-1/2} A D^{-1/2}$, computing leading eigenvalues and eigenvectors.
* **Spectral Graph Wavelets (SGWT):** Wavelet localization of pressure disturbances over non-Euclidean graph topologies using Chebyshev polynomial approximations.
* **Interactive 2D Dashboard:** Dark-mode map (Leaflet / Canvas 2D) rendering dynamic arcs, interactive node inspection, and spectrum plots.