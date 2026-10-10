# Tech Stack & Dependencies

## Backend & Computation
* **Python 3.11+**
* **Linear Algebra & Graph Theory:**
  * `numpy` (>= 1.26.0) — Vectorized array operations and tensor slicing.
  * `scipy` (>= 1.12.0) — Sparse linear algebra (`scipy.sparse.linalg.eigsh`), statistical p-values.
  * `zarr` (>= 2.16.0) — Chunked, compressed multidimensional storage.
* **Networking & Pipeline:**
  * `httpx` (>= 0.27.0) — Async HTTP client.
  * `openmeteo-requests` — FlatBuffers-optimized Open-Meteo API wrapper.
* **API Layer:**
  * `fastapi` (>= 0.110.0) — REST and WebSocket framework.
  * `uvicorn[standard]` — High-performance ASGI runtime.
  * `aiosqlite` — Non-blocking SQLite access.
  * `pydantic` (>= 2.6.0) — Request/response schemas.

## Frontend
* **Runtime & Build:** Node.js 20+, Vite.
* **Rendering & GIS:**
  * `leaflet` (>= 1.9.4) — Base map layer.
  * HTML5 Canvas 2D — Custom rendering for tens of thousands of arcs and particle waves.
  * Tiles: CartoDB Dark Matter (`https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png`).
* **Charts:**
  * `uplot` — Lightweight, zero-dependency timeline and spectrum plotting.

## Database
* **SQLite 3** — Standard embedded engine.