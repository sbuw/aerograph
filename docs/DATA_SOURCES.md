# Data Sources Specification

## 1. Primary: Open-Meteo Historical & Forecast APIs
* **Endpoint (Archive):** `https://archive-api.open-meteo.com/v1/archive`
* **Endpoint (Forecast):** `https://api.open-meteo.com/v1/forecast`
* **Auth:** Free tier, no API key required for non-commercial research.
* **Target Variables:**
  * `geopotential_height_500hPa` (mid-tropospheric circulation indicator).
  * `surface_pressure` (mean sea level pressure anomalies).
* **Grid Resolution:**
  * Spatial: $5^\circ \times 5^\circ$ regular global or Northern Hemisphere grid ($20^\circ\text{N}$ to $80^\circ\text{N}$).
  * Total nodes: $N \approx 1,000 - 2,592$.
  * Temporal window: Rolling 60–90 days at 12h or 24h sample rates.

## 2. Secondary (Production Fallback): NOAA GFS on AWS Open Data
* **Format:** GRIB2 / NetCDF via public S3 bucket (`noaa-gfs-bdp-pds`).
* **Target:** $1^\circ \times 1^\circ$ resolution grids for operational runs.