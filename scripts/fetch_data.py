"""
Data ingestion script for Open-Meteo climate pressure grids.
Batches coordinate queries to download 60-day historical pressure grids,
constructs regular (Time x Lat x Lon) tensors, and caches them in data/cache/grid.npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_data")


def create_coordinate_grid(
    lat_min: float = 20.0,
    lat_max: float = 80.0,
    lat_step: float = 5.0,
    lon_min: float = -180.0,
    lon_max: float = 175.0,
    lon_step: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Generate regular 2D coordinate grid arrays and node index mapping.
    
    Returns:
        lats: 1D array of latitude values (shape: n_lats)
        lons: 1D array of longitude values (shape: n_lons)
        nodes: List of metadata dicts {id, lat, lon, grid_i, grid_j}
    """
    lats = np.arange(lat_min, lat_max + 1e-4, lat_step, dtype=np.float32)
    lons = np.arange(lon_min, lon_max + 1e-4, lon_step, dtype=np.float32)

    nodes: List[Dict[str, Any]] = []
    node_id = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            nodes.append(
                {
                    "id": node_id,
                    "lat": float(lat),
                    "lon": float(lon),
                    "grid_i": int(i),
                    "grid_j": int(j),
                }
            )
            node_id += 1

    return lats, lons, nodes


def fetch_open_meteo_batch(
    coords: List[Tuple[float, float]],
    past_days: int = 60,
    variables: Tuple[str, ...] = (
        "geopotential_height_500hPa",
        "surface_pressure",
        "pressure_msl",
    ),
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Fetch weather variables for a batch of coordinates from Open-Meteo Forecast API.
    
    Args:
        coords: List of (lat, lon) pairs.
        past_days: Rolling historical window in days.
        variables: Hourly variables to request.
        max_retries: Max retry attempts on failure.
        backoff_factor: Exponential backoff delay multiplier.
        
    Returns:
        List of location payload dicts from Open-Meteo.
    """
    lats_str = ",".join(f"{lat:.2f}" for lat, _ in coords)
    lons_str = ",".join(f"{lon:.2f}" for _, lon in coords)
    vars_str = ",".join(variables)

    params = {
        "latitude": lats_str,
        "longitude": lons_str,
        "past_days": str(past_days),
        "forecast_days": "0",
        "hourly": vars_str,
    }
    query_str = urllib.parse.urlencode(params)
    url = f"https://api.open-meteo.com/v1/forecast?{query_str}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ClimateTeleconnectionEngine/1.0 (Research Ingestion)"},
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=30) as resp:
                elapsed = time.time() - t0
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                if isinstance(data, dict):
                    # Single point returns dict; normalize to list
                    data = [data]
                logger.info(
                    "Fetched batch of %d coords in %.2fs (HTTP %d, payload %.1f KB)",
                    len(coords),
                    elapsed,
                    resp.status,
                    len(body) / 1024.0,
                )
                return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            last_error = err
            wait_time = backoff_factor**attempt
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt,
                max_retries,
                err,
                wait_time,
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to fetch batch after {max_retries} attempts: {last_error}"
    )


def sample_step_from_rate(sample_rate: str) -> int:
    """Map sample rate string (e.g. '1h', '12h', '24h') to hourly step integer."""
    rates = {
        "1h": 1,
        "6h": 6,
        "12h": 12,
        "24h": 24,
    }
    if sample_rate not in rates:
        raise ValueError(
            f"Unsupported sample rate '{sample_rate}'. Choose from {list(rates.keys())}"
        )
    return rates[sample_rate]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch fetch climate pressure grids via Open-Meteo into data/cache/grid.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/grid.npz"),
        help="Target .npz cache file path (default: data/cache/grid.npz)",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=60,
        help="Historical rolling window in days (default: 60)",
    )
    parser.add_argument(
        "--sample-rate",
        type=str,
        default="12h",
        choices=["1h", "6h", "12h", "24h"],
        help="Sample rate step for output tensor (default: 12h)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Number of spatial coordinates per Open-Meteo HTTP request (default: 250)",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=20.0,
        help="Minimum latitude in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=80.0,
        help="Maximum latitude in degrees (default: 80.0)",
    )
    parser.add_argument(
        "--lat-step",
        type=float,
        default=5.0,
        help="Latitude step in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=-180.0,
        help="Minimum longitude in degrees (default: -180.0)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=175.0,
        help="Maximum longitude in degrees (default: 175.0)",
    )
    parser.add_argument(
        "--lon-step",
        type=float,
        default=5.0,
        help="Longitude step in degrees (default: 5.0)",
    )

    args = parser.parse_args()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing coordinate grid...")
    lats, lons, nodes = create_coordinate_grid(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lat_step=args.lat_step,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lon_step=args.lon_step,
    )
    n_lats = len(lats)
    n_lons = len(lons)
    total_nodes = len(nodes)
    logger.info(
        "Grid initialized: %d lats [%.1f..%.1f] x %d lons [%.1f..%.1f] = %d spatial nodes",
        n_lats,
        lats[0],
        lats[-1],
        n_lons,
        lons[0],
        lons[-1],
        total_nodes,
    )

    # Chunk coordinate points into batches
    coords_list = [(node["lat"], node["lon"]) for node in nodes]
    batch_size: int = args.batch_size
    num_batches = (total_nodes + batch_size - 1) // batch_size
    logger.info(
        "Fetching %d-day pressure fields in %d batches (batch size: %d)...",
        args.past_days,
        num_batches,
        batch_size,
    )

    all_results: List[Dict[str, Any]] = []
    t_start = time.time()
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, total_nodes)
        batch_coords = coords_list[start_i:end_i]
        logger.info(
            "Batch %d/%d (nodes %d to %d)...",
            batch_idx + 1,
            num_batches,
            start_i,
            end_i - 1,
        )
        batch_res = fetch_open_meteo_batch(
            coords=batch_coords,
            past_days=args.past_days,
            variables=(
                "geopotential_height_500hPa",
                "surface_pressure",
                "pressure_msl",
            ),
        )
        if len(batch_res) != len(batch_coords):
            raise ValueError(
                f"Batch {batch_idx + 1}: expected {len(batch_coords)} responses, received {len(batch_res)}"
            )
        all_results.extend(batch_res)

    total_fetch_time = time.time() - t_start
    logger.info(
        "All %d location responses received in %.2fs (avg %.2fs/batch)",
        len(all_results),
        total_fetch_time,
        total_fetch_time / num_batches,
    )

    # Validate timestamps and extract time dimension
    first_hourly = all_results[0]["hourly"]
    raw_times: List[str] = first_hourly["time"]
    raw_t = len(raw_times)
    logger.info("Raw hourly timestamps per node: %d", raw_t)

    step = sample_step_from_rate(args.sample_rate)
    time_indices = np.arange(0, raw_t, step)
    sampled_times = np.array([raw_times[idx] for idx in time_indices], dtype=str)
    t_steps = len(time_indices)
    logger.info(
        "Sampled time dimension at '%s' (step=%d): %d steps (%s to %s)",
        args.sample_rate,
        step,
        t_steps,
        sampled_times[0],
        sampled_times[-1],
    )

    # Allocate tensors (Time x Lat x Lon)
    z500_tensor = np.empty((t_steps, n_lats, n_lons), dtype=np.float32)
    sp_tensor = np.empty((t_steps, n_lats, n_lons), dtype=np.float32)
    mslp_tensor = np.empty((t_steps, n_lats, n_lons), dtype=np.float32)

    # Fill tensors
    logger.info("Populating (Time x Lat x Lon) tensors from responses...")
    for node, result in zip(nodes, all_results):
        grid_i = node["grid_i"]
        grid_j = node["grid_j"]
        hourly = result["hourly"]

        raw_z500 = np.array(hourly["geopotential_height_500hPa"], dtype=np.float32)
        raw_sp = np.array(hourly["surface_pressure"], dtype=np.float32)
        raw_mslp = np.array(hourly["pressure_msl"], dtype=np.float32)

        z500_tensor[:, grid_i, grid_j] = raw_z500[time_indices]
        sp_tensor[:, grid_i, grid_j] = raw_sp[time_indices]
        mslp_tensor[:, grid_i, grid_j] = raw_mslp[time_indices]

    # Quick NaN / Inf verification
    for name, arr in [
        ("z500", z500_tensor),
        ("surface_pressure", sp_tensor),
        ("pressure_msl", mslp_tensor),
    ]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            logger.warning(
                "Tensor '%s' contains %d NaN values! Interpolating/zeroing...",
                name,
                nan_count,
            )
            np.nan_to_num(arr, copy=False)

    # Construct node metadata table for SQLite graph_nodes compatibility
    # Columns: node_id, lat, lon, grid_i, grid_j
    node_matrix = np.array(
        [[n["id"], n["lat"], n["lon"], n["grid_i"], n["grid_j"]] for n in nodes],
        dtype=np.float32,
    )

    # Save to compressed .npz archive
    logger.info("Compressing and saving tensors to '%s'...", output_path)
    np.savez_compressed(
        output_path,
        z500=z500_tensor,
        surface_pressure=sp_tensor,
        pressure_msl=mslp_tensor,
        lats=lats,
        lons=lons,
        timestamps=sampled_times,
        nodes=node_matrix,
        sample_rate=np.array(args.sample_rate, dtype=str),
        past_days=np.array(args.past_days, dtype=np.int32),
    )

    file_size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    logger.info(
        "Successfully saved '%s' (%.2f MB on disk)",
        output_path,
        file_size_mb,
    )

    print("\n" + "=" * 60)
    print(" VERIFICATION SUMMARY: DATASET SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f" Output File:         {output_path.resolve()}")
    print(f" Disk Size:           {file_size_mb:.2f} MB")
    print(f" Spatial Nodes:       {total_nodes} ({n_lats} lats x {n_lons} lons)")
    print(f" Time Dimension (T):  {t_steps} steps ({args.past_days} days @ {args.sample_rate})")
    print(f" z500 Tensor Shape:   {z500_tensor.shape} (dtype: {z500_tensor.dtype})")
    print(f" Surface Pressure:    {sp_tensor.shape} (dtype: {sp_tensor.dtype})")
    print(f" Pressure MSL:        {mslp_tensor.shape} (dtype: {mslp_tensor.dtype})")
    print(f" Lats Shape:          {lats.shape} [{lats[0]:.1f}° to {lats[-1]:.1f}°]")
    print(f" Lons Shape:          {lons.shape} [{lons[0]:.1f}° to {lons[-1]:.1f}°]")
    print(f" Timestamps Shape:    {sampled_times.shape} [{sampled_times[0]} -> {sampled_times[-1]}]")
    print(f" Nodes Metadata:      {node_matrix.shape}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
