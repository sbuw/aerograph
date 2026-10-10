"""
FastAPI Backend Service for Planetary Climate Teleconnections & Spectral Wavelets.
Serves topology snapshots, edge arcs, Laplacian spectra, and dynamic SGWT wavelet fields.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiosqlite
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy import sparse

from engine.export import generate_climate_report
from engine.spectral import (
    build_normalized_laplacian,
    chebyshev_sgwt_filter,
    compute_wavelet_atom,
)
from engine.teleconnection import compute_teleconnections

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_main")

DB_PATH = Path("data/teleconnections.db")
GRID_PATH = Path("data/cache/grid.npz")
SPECTRAL_PATH = Path("data/cache/spectral.npz")


class NodeDTO(BaseModel):
    id: int
    lat: float
    lon: float
    grid_i: int
    grid_j: int
    is_active: bool
    degree: float


class EdgeDTO(BaseModel):
    source_id: int
    target_id: int
    source_lat: float
    source_lon: float
    target_lat: float
    target_lon: float
    correlation: float
    abs_correlation: float
    lag_days: int
    p_value: float
    distance_km: float


class EdgesResponse(BaseModel):
    epoch_id: int
    total_returned: int
    min_r: float
    edges: List[EdgeDTO]


class SpectrumResponse(BaseModel):
    spectral_radius: float
    fiedler_value: float
    eigenvalues: List[float]
    active_nodes_count: int
    total_nodes: int


class WaveletResponse(BaseModel):
    node_id: int
    node_lat: float
    node_lon: float
    scale: float
    chebyshev_order: int
    peak_value: float
    l2_energy: float
    psi: List[float]


class NaoIndexDTO(BaseModel):
    value: float
    phase: str
    description: str
    azores_coords: List[float]
    iceland_coords: List[float]


class AoIndexDTO(BaseModel):
    value: float
    phase: str
    description: str


class ClimateIndicesResponse(BaseModel):
    nao: NaoIndexDTO
    ao: AoIndexDTO
    timestamp: str


class NodeForecastResponse(BaseModel):
    node_id: int
    lat: float
    lon: float
    current_val: float
    horizon_days: int
    forecast_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    num_predictors: int
    r2: float
    synoptic_verdict: str


# In-memory application state
state: Dict[str, Any] = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload graph Laplacian, spectral metrics, and coordinates on startup."""
    logger.info("Initializing API in-memory caches...")
    
    # 1. Load spectral cache
    if SPECTRAL_PATH.exists():
        with np.load(SPECTRAL_PATH) as s_data:
            state["eigenvalues"] = s_data["eigenvalues"].tolist()
            state["spectral_radius"] = float(s_data["spectral_radius"])
            state["fiedler_value"] = float(s_data["fiedler_value"])
            state["active_mask"] = s_data["active_mask"]
            state["degrees"] = s_data["degrees"]
        logger.info("Loaded spectral metrics from '%s'", SPECTRAL_PATH)
    else:
        logger.warning("Spectral cache '%s' not found. Run engine/spectral.py first.", SPECTRAL_PATH)

    # 2. Reconstruct or load sparse Laplacian L for real-time SGWT evaluation
    if GRID_PATH.exists():
        logger.info("Reconstructing sparse Laplacian for on-the-fly SGWT evaluation...")
        tele_res = compute_teleconnections(GRID_PATH, min_r=0.65, max_p=0.01)
        L, degrees, active_mask = build_normalized_laplacian(tele_res.adjacency_sparse)
        state["laplacian"] = L
        state["node_metadata"] = tele_res.nodes  # (N, 5)
        state["degrees"] = degrees
        state["active_mask"] = active_mask
        state["lambda_max"] = state.get("spectral_radius", 2.0)
        logger.info("Laplacian cached in RAM (shape: %s)", L.shape)
    else:
        logger.warning("Grid cache '%s' not found.", GRID_PATH)

    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Planetary Climate Teleconnection Graph & Spectral Wavelets API",
    version="1.0.0",
    description="Vectorized teleconnection graphs, Laplacian harmonics, and Spectral Graph Wavelets (2D)",
    lifespan=lifespan,
)

# CORS setup for Vite / Leaflet client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint verifying database and in-memory caches."""
    db_ok = DB_PATH.exists()
    return {
        "status": "online",
        "database_exists": db_ok,
        "laplacian_cached": "laplacian" in state,
        "spectral_cached": "eigenvalues" in state,
    }


@app.get("/api/nodes", response_model=List[NodeDTO])
async def get_nodes():
    """Return complete list of 936 spatial grid nodes with coordinates and active status."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")

    active_mask = state.get("active_mask", np.ones(936, dtype=bool))
    degrees = state.get("degrees", np.zeros(936, dtype=np.float32))

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, lat, lon, grid_i, grid_j FROM graph_nodes ORDER BY id ASC;")
        rows = await cursor.fetchall()

    nodes: List[NodeDTO] = []
    for r in rows:
        node_id = r["id"]
        is_act = bool(active_mask[node_id]) if node_id < len(active_mask) else True
        deg = float(degrees[node_id]) if node_id < len(degrees) else 0.0
        nodes.append(
            NodeDTO(
                id=node_id,
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                grid_i=int(r["grid_i"]),
                grid_j=int(r["grid_j"]),
                is_active=is_act,
                degree=deg,
            )
        )
    return nodes


@app.get("/api/edges", response_model=EdgesResponse)
async def get_edges(
    min_r: float = Query(0.65, ge=0.0, le=1.0, description="Minimum absolute correlation threshold"),
    limit: int = Query(2000, ge=1, le=10000, description="Max edges to return"),
    epoch: str = Query("latest", description="Epoch ID or 'latest'"),
):
    """Retrieve pruned teleconnection arcs for map canvas rendering."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        # Resolve epoch ID
        if epoch == "latest":
            cur_ep = await db.execute("SELECT id FROM computation_epochs ORDER BY id DESC LIMIT 1;")
            ep_row = await cur_ep.fetchone()
            if not ep_row:
                raise HTTPException(status_code=404, detail="No computation epochs found in database")
            epoch_id = ep_row["id"]
        else:
            try:
                epoch_id = int(epoch)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid epoch format")

        query = """
            SELECT 
                e.source_node_id, e.target_node_id, e.correlation, e.lag_days, e.p_value, e.distance_km,
                s.lat AS s_lat, s.lon AS s_lon,
                t.lat AS t_lat, t.lon AS t_lon
            FROM teleconnection_edges e
            JOIN graph_nodes s ON e.source_node_id = s.id
            JOIN graph_nodes t ON e.target_node_id = t.id
            WHERE e.epoch_id = ? AND ABS(e.correlation) >= ?
            ORDER BY ABS(e.correlation) DESC
            LIMIT ?;
        """
        cursor = await db.execute(query, (epoch_id, min_r, limit))
        rows = await cursor.fetchall()

    edges: List[EdgeDTO] = []
    for r in rows:
        edges.append(
            EdgeDTO(
                source_id=r["source_node_id"],
                target_id=r["target_node_id"],
                source_lat=r["s_lat"],
                source_lon=r["s_lon"],
                target_lat=r["t_lat"],
                target_lon=r["t_lon"],
                correlation=r["correlation"],
                abs_correlation=abs(r["correlation"]),
                lag_days=r["lag_days"],
                p_value=r["p_value"],
                distance_km=r["distance_km"],
            )
        )

    return EdgesResponse(
        epoch_id=epoch_id,
        total_returned=len(edges),
        min_r=min_r,
        edges=edges,
    )


@app.get("/api/node/{node_id}/teleconnections")
async def get_node_teleconnections(node_id: int):
    """Retrieve all incident teleconnection arcs for a clicked node."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not initialized")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        # Verify node
        cur_node = await db.execute("SELECT id, lat, lon FROM graph_nodes WHERE id = ?;", (node_id,))
        node_row = await cur_node.fetchone()
        if not node_row:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        query = """
            SELECT 
                e.source_node_id, e.target_node_id, e.correlation, e.lag_days, e.p_value, e.distance_km,
                s.lat AS s_lat, s.lon AS s_lon,
                t.lat AS t_lat, t.lon AS t_lon
            FROM teleconnection_edges e
            JOIN graph_nodes s ON e.source_node_id = s.id
            JOIN graph_nodes t ON e.target_node_id = t.id
            WHERE (e.source_node_id = ? OR e.target_node_id = ?)
            ORDER BY ABS(e.correlation) DESC;
        """
        cursor = await db.execute(query, (node_id, node_id))
        rows = await cursor.fetchall()

    incident_edges = []
    for r in rows:
        # Determine peer node
        is_source = (r["source_node_id"] == node_id)
        peer_id = r["target_node_id"] if is_source else r["source_node_id"]
        peer_lat = r["t_lat"] if is_source else r["s_lat"]
        peer_lon = r["t_lon"] if is_source else r["s_lon"]
        # Signed lag relative to clicked node (positive if clicked leads peer)
        rel_lag = r["lag_days"] if is_source else -r["lag_days"]

        incident_edges.append(
            {
                "peer_id": peer_id,
                "peer_lat": peer_lat,
                "peer_lon": peer_lon,
                "correlation": r["correlation"],
                "abs_correlation": abs(r["correlation"]),
                "lag_days": rel_lag,
                "p_value": r["p_value"],
                "distance_km": r["distance_km"],
                "is_source": is_source,
            }
        )

    return {
        "node_id": node_id,
        "lat": node_row["lat"],
        "lon": node_row["lon"],
        "teleconnections_count": len(incident_edges),
        "edges": incident_edges,
    }


@app.get("/api/spectrum", response_model=SpectrumResponse)
async def get_spectrum():
    """Retrieve Laplacian eigenvalues and Fiedler algebraic connectivity."""
    if "eigenvalues" not in state:
        raise HTTPException(status_code=503, detail="Spectral data not loaded")

    return SpectrumResponse(
        spectral_radius=state["spectral_radius"],
        fiedler_value=state["fiedler_value"],
        eigenvalues=state["eigenvalues"],
        active_nodes_count=int(np.sum(state["active_mask"])),
        total_nodes=len(state["active_mask"]),
    )


@app.get("/api/wavelet", response_model=WaveletResponse)
async def get_wavelet_field(
    node_id: int = Query(652, ge=0, le=935, description="Center node index"),
    scale: float = Query(5.0, gt=0.0, le=50.0, description="Wavelet scale s"),
    order: int = Query(30, ge=5, le=60, description="Chebyshev truncation order M"),
):
    """
    Dynamically compute localized Spectral Graph Wavelet atom psi_{s, i} = g(sL) * delta_i
    using the Chebyshev polynomial recurrence on the sparse Laplacian L.
    Returns 936 values across the global spatial grid for direct 2D canvas heatmap rendering.
    """
    if "laplacian" not in state:
        raise HTTPException(status_code=503, detail="Laplacian not initialized in memory")

    L: sparse.csr_matrix = state["laplacian"]
    nodes_meta: np.ndarray = state["node_metadata"]
    lambda_max: float = state.get("lambda_max", 2.0)

    # Compute wavelet field
    psi = compute_wavelet_atom(
        L=L,
        node_id=node_id,
        scale=scale,
        lambda_max=lambda_max,
        order=order,
    )

    node_row = nodes_meta[node_id]
    peak_val = float(np.max(np.abs(psi)))
    l2_energy = float(np.sum(psi**2))

    return WaveletResponse(
        node_id=node_id,
        node_lat=float(node_row[1]),
        node_lon=float(node_row[2]),
        scale=scale,
        chebyshev_order=order,
        peak_value=peak_val,
        l2_energy=l2_energy,
        psi=[round(float(v), 6) for v in psi],
    )


@app.get("/api/export/report")
async def export_report(force_regenerate: bool = Query(False, description="Force new report generation")):
    """
    Generate or retrieve the comprehensive climate teleconnections analytical report.
    Returns a timestamped ZIP archive containing REPORT.md, REPORT.html, charts/, and data/.
    """
    try:
        today_tag = datetime.now().strftime("%Y-%m-%d")
        existing_zip = Path("exports") / f"climate_report_{today_tag}.zip"
        if not force_regenerate and existing_zip.exists() and existing_zip.stat().st_size > 1000:
            zip_path = existing_zip
        else:
            zip_path_str = generate_climate_report(output_dir="exports")
            zip_path = Path(zip_path_str)

        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="Export report file could not be generated")

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_path.name,
        )
    except Exception as exc:
        logger.exception("Failed to generate export report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(exc)}")


@app.get("/api/climate-indices", response_model=ClimateIndicesResponse)
async def get_climate_indices():
    """Retrieve macro-atmospheric teleconnection indices (NAO, AO) and current phases."""
    try:
        from engine.forecast import compute_climate_indices
        return compute_climate_indices()
    except Exception as exc:
        logger.exception("Failed to compute climate indices: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/node/{node_id}/forecast", response_model=NodeForecastResponse)
async def get_node_forecast(
    node_id: int,
    horizon_days: int = Query(5, ge=1, le=14, description="Forecast horizon in days"),
):
    """Retrieve multi-day lead-lag autoregressive pressure trajectory forecast for a grid node."""
    try:
        from engine.forecast import predict_node_trajectory
        return predict_node_trajectory(node_id=node_id, horizon_days=horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to generate forecast for node %s: %s", node_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

