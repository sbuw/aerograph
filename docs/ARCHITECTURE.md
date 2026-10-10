# System Architecture

## 1. End-to-End Data & Processing Pipeline

```text
               [ Open-Meteo Historical & Reanalysis API ]
                                   │
                                   ▼
                       [ scripts/fetch_data.py ]
                                   │
                    data/cache/grid.npz (T=120, 13x72)
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
 [ engine/teleconnection.py ]                  [ engine/forecast.py ]
 • Lagged Cross-Correlation (tau: -7..+7)      • Macro Indices: NAO & AO
 • Great-Circle Distance Filter (d >= 1000km)  • Lead-Lag 5-day Autoregression
 • Effective DoF (Quenouille/Davis Neff)       • Node Predictability (R² Skill)
             │                                           │
             ▼                                           │
    data/teleconnections.db (SQLite)                     │
 (graph_nodes, computation_epochs, edges)               │
             │                                           │
             ▼                                           │
     [ engine/spectral.py ]                              │
 • Normalized Graph Laplacian L                          │
 • Sparse Eigensolver (Fiedler value λ₂)                 │
 • Chebyshev SGWT Wavelet Polynomials                    │
             │                                           │
             └─────────────────────┬─────────────────────┘
                                   ▼
                          [ api/main.py ]
                         (FastAPI REST API)
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
[ engine/export.py ]                              [ client/ (Vite/Leaflet/uPlot) ]
• 7 High-Res Charts (300 DPI)                     • 2D Canvas Arcs Overlay (Bézier)
• REPORT.md & Glassmorphism REPORT.html           • Dynamic SGWT Heatmap Layer
• Tabular CSV & JSON Datasets                     • Interactive HUD Pills (NAO/AO)
• Timestamped ZIP Export Stream                   • Forecast Drawer & Trajectory SVG
```

## 2. Component Specifications

### 2.1 Ingestion & Cache Layer
* `scripts/fetch_data.py`: Batches latitude slices across coordinate grids ($20^\circ\text{N} - 80^\circ\text{N}$, $-180^\circ - +175^\circ$, step $5^\circ$), fetching synoptic pressure fields ($Z_{500}$ / surface pressure) over 60-day windows.
* `data/cache/grid.npz`: Compressed NumPy tensor cache containing standardized pressure grids, spatial coordinate metadata, and epoch timestamps.
* `data/cache/spectral.npz`: Cached graph Laplacian spectra, smallest 20 eigenvalues, Fiedler eigenvector, node degree vectors, and pre-evaluated SGWT wavelet response slices.

### 2.2 Mathematical Engine
* `engine/teleconnection.py`: Vectorized lagged cross-correlation matrix across $\tau \in [-7, +7]$ days. Filters pairs by great-circle orthodromic distance ($d \ge 1\,000\text{ km}$) and computes effective degrees of freedom ($N_{\mathrm{eff}}$) via Quenouille/Davis autocorrelation correction.
* `engine/spectral.py`: Constructs the symmetric normalized graph Laplacian $\mathcal{L} = I - D^{-1/2} A D^{-1/2}$ with pseudo-inverse regularization for isolated nodes. Solves sparse eigensystems via `scipy.sparse.linalg.eigsh` and evaluates Spectral Graph Wavelet Transform (SGWT) atoms via fast 30th-order Chebyshev polynomial recurrence (Clenshaw algorithm).
* `engine/forecast.py`: Computes macro-scale atmospheric teleconnection indices (North Atlantic Oscillation / NAO, Arctic Oscillation / AO via Laplacian harmonic projection) and builds lead-lag autoregressive models forecasting 5-day trajectory anomalies.

### 2.3 Persistence & Storage
* `data/schema.sql`: Embedded SQLite schema defining tables `graph_nodes`, `computation_epochs`, and `teleconnection_edges` with foreign keys and compound indices on `(epoch_id, ABS(correlation))` and `(source_node_id, target_node_id)`.
* `data/teleconnections.db`: Local SQLite database storing topology snapshots and validated significant teleconnection pairs.

### 2.4 API & Export Layer
* `api/main.py`: Asynchronous FastAPI application serving REST endpoints with in-memory caching of the sparse Laplacian $\mathcal{L}$ for sub-10ms dynamic wavelet calculations.
* `engine/export.py` & `scripts/export_report.py`: Automated report generator producing 7 publication-grade charts (300 DPI, dark palette), Markdown report, self-contained glassmorphism HTML dashboard with print-to-PDF styles, and tabular CSV/JSON files packed into a ZIP archive.

### 2.5 Web Client Layer
* `client/`: Single-page Vite application featuring Leaflet CartoDB Dark Matter basemap, custom HTML5 Canvas 2D overlay rendering thousands of quadratic Bézier arcs and SGWT heatmaps, `uPlot` correlation lag spectrum charts, live HUD stat pills, and side drawer forecast tabs with interactive SVG confidence bounds.