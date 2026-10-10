"""
Planetary Climate Teleconnection Engine.
Vectorized lagged cross-correlation, Haversine spatial distance filtering,
effective degrees of freedom (Neff) calculation, and symmetric adjacency matrix construction.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse, stats

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("teleconnection_engine")


@dataclass(slots=True, frozen=True)
class TeleconnectionEdge:
    """Represents a validated teleconnection link between two spatial nodes."""
    source_node_id: int
    target_node_id: int
    source_lat: float
    source_lon: float
    target_lat: float
    target_lon: float
    correlation: float
    abs_correlation: float
    lag_days: float
    p_value: float
    distance_km: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "source_lat": self.source_lat,
            "source_lon": self.source_lon,
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "correlation": self.correlation,
            "abs_correlation": self.abs_correlation,
            "lag_days": self.lag_days,
            "p_value": self.p_value,
            "distance_km": self.distance_km,
        }


@dataclass(slots=True)
class TeleconnectionResult:
    """Encapsulates the teleconnection graph adjacency, pruned edges, and matrices."""
    adjacency_sparse: sparse.csr_matrix
    adjacency_dense: np.ndarray
    edges: List[TeleconnectionEdge]
    nodes: np.ndarray  # shape: (N, 5) -> [node_id, lat, lon, grid_i, grid_j]
    edge_count: int
    node_count: int
    max_abs_r: np.ndarray
    best_r: np.ndarray
    best_lag_days: np.ndarray
    distance_km: np.ndarray
    p_values: np.ndarray


def haversine_distance_matrix(lats_deg: np.ndarray, lons_deg: np.ndarray) -> np.ndarray:
    """
    Compute pairwise great-circle distance matrix on Earth using vectorized Haversine formula.
    
    Args:
        lats_deg: 1D array of node latitudes in degrees (N,).
        lons_deg: 1D array of node longitudes in degrees (N,).
        
    Returns:
        Pairwise distance matrix in kilometers of shape (N, N).
    """
    lats = np.radians(lats_deg.astype(np.float64))
    lons = np.radians(lons_deg.astype(np.float64))

    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]

    a = (
        np.sin(dlat * 0.5) ** 2
        + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon * 0.5) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    earth_radius_km = 6371.0088
    return (earth_radius_km * c).astype(np.float32)


def compute_effective_degrees_of_freedom(
    X_norm: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Compute pairwise effective degrees of freedom (N_eff) using the Quenouille / Davis formula
    to correct for temporal autocorrelation (red noise) in atmospheric time series.
    
    Formula:
        N_eff = T * (1 - rho_i(1) * rho_j(1)) / (1 + rho_i(1) * rho_j(1))
        
    Args:
        X_norm: Standardized anomaly matrix of shape (T, N).
        T: Number of temporal observations.
        
    Returns:
        Symmetric N_eff matrix of shape (N, N), bounded in [3.0, T].
    """
    # Lag-1 sample autocorrelation per node
    numerator = np.sum(X_norm[:-1] * X_norm[1:], axis=0)
    denominator = np.sum(X_norm**2, axis=0) + 1e-12
    rho1 = np.clip(numerator / denominator, -0.99, 0.99)

    prod_rho = rho1[:, None] * rho1[None, :]
    N_eff = T * (1.0 - prod_rho) / (1.0 + prod_rho)
    return np.clip(N_eff, 3.0, float(T)).astype(np.float32)


def compute_lagged_cross_correlations(
    X_norm: np.ndarray,
    max_lag_steps: int,
    T: int,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of two-sided lagged cross-correlation across lags tau in [-max_lag, +max_lag].
    
    Guarantees exact mathematical symmetry for peak absolute correlation:
        max_abs_r[i, j] == max_abs_r[j, i]
        
    Returns:
        max_abs_r: Symmetric matrix of peak |r| of shape (N, N).
        best_r: Signed correlation at the peak lag of shape (N, N).
        best_lag_steps: Lag (in integer steps) where peak occurred of shape (N, N).
    """
    # Instantaneous correlation (tau = 0)
    R0 = (X_norm.T @ X_norm) / float(T)
    max_abs_r = np.abs(R0)
    best_r = R0.copy()
    best_lag_steps = np.zeros((N, N), dtype=np.int16)

    # Positive and negative lags
    for tau in range(1, max_lag_steps + 1):
        X_lead = X_norm[:-tau]
        X_lag = X_norm[tau:]
        denom = float(T - tau)
        
        # R_tau[i, j]: i leads j by +tau steps
        R_tau = (X_lead.T @ X_lag) / denom
        abs_R_tau = np.abs(R_tau)

        # 1. Update where +tau is strictly higher than previous peak
        mask_pos = abs_R_tau > max_abs_r
        max_abs_r[mask_pos] = abs_R_tau[mask_pos]
        best_r[mask_pos] = R_tau[mask_pos]
        best_lag_steps[mask_pos] = tau

        # 2. Update where -tau (i lags j by tau, which is R_tau.T) is higher
        R_neg = R_tau.T
        abs_R_neg = abs_R_tau.T
        mask_neg = abs_R_neg > max_abs_r
        max_abs_r[mask_neg] = abs_R_neg[mask_neg]
        best_r[mask_neg] = R_neg[mask_neg]
        best_lag_steps[mask_neg] = -tau

    # Enforce exact symmetry: max_abs_r = max(max_abs_r, max_abs_r.T)
    max_abs_r = np.maximum(max_abs_r, max_abs_r.T)
    return max_abs_r, best_r, best_lag_steps


def compute_teleconnections(
    npz_path: str | Path = "data/cache/grid.npz",
    variable_name: str = "z500",
    min_r: float = 0.65,
    max_p: float = 0.01,
    min_dist_km: float = 1000.0,
    max_lag_days: float = 7.0,
) -> TeleconnectionResult:
    """
    Load climate grid tensor, standardize anomaly series, compute two-sided lagged cross-correlations,
    apply Haversine distance filtering and Quenouille-corrected p-value significance,
    and build the symmetric adjacency matrix and validated edge list.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Input grid archive not found: {path.resolve()}")

    logger.info("Loading '%s' from '%s'...", variable_name, path)
    with np.load(path) as data:
        tensor = data[variable_name]  # (T, n_lats, n_lons)
        nodes = data["nodes"]        # (N, 5) -> [node_id, lat, lon, grid_i, grid_j]
        sample_rate_str = str(data["sample_rate"])

    T, n_lats, n_lons = tensor.shape
    N = n_lats * n_lons
    logger.info(
        "Tensor shape: %s (T=%d, lats=%d, lons=%d, N=%d nodes, sample_rate=%s)",
        tensor.shape,
        T,
        n_lats,
        n_lons,
        N,
        sample_rate_str,
    )

    # Flatten spatial grid to (T, N)
    X = tensor.reshape(T, N).astype(np.float32)

    # 1. Standardize anomalies per node
    logger.info("Standardizing pressure anomalies...")
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # 2. Derive lag step conversion
    hours_per_step = 12.0
    if sample_rate_str == "1h":
        hours_per_step = 1.0
    elif sample_rate_str == "6h":
        hours_per_step = 6.0
    elif sample_rate_str == "24h":
        hours_per_step = 24.0

    steps_per_day = 24.0 / hours_per_step
    max_lag_steps = int(round(max_lag_days * steps_per_day))
    logger.info(
        "Lag window: %.1f days = %d discrete time steps (sampling: %.1fh/step)",
        max_lag_days,
        max_lag_steps,
        hours_per_step,
    )

    # 3. Vectorized cross-correlation
    logger.info("Computing vectorized lagged cross-correlations across [-7..+7] days...")
    max_abs_r, best_r, best_lag_steps = compute_lagged_cross_correlations(
        X_norm, max_lag_steps, T, N
    )
    best_lag_days = (best_lag_steps / steps_per_day).astype(np.float32)

    # 4. Pairwise spherical Haversine distance
    logger.info("Computing Haversine distance matrix...")
    lats_deg = nodes[:, 1]
    lons_deg = nodes[:, 2]
    dist_matrix = haversine_distance_matrix(lats_deg, lons_deg)

    # 5. Effective degrees of freedom and p-values
    logger.info("Calculating Neff via Quenouille/Davis formula and Student's t p-values...")
    N_eff = compute_effective_degrees_of_freedom(X_norm, T)

    # Student's t statistic: t = |r| * sqrt((N_eff - 2) / (1 - r^2))
    df = np.maximum(N_eff - 2.0, 1.0)
    r_sq = np.clip(max_abs_r**2, 0.0, 0.9999)
    t_stat = max_abs_r * np.sqrt(df / (1.0 - r_sq + 1e-12))
    p_values = (2.0 * stats.t.sf(t_stat, df=df)).astype(np.float32)

    # 6. Apply filtering masks
    logger.info(
        "Applying teleconnection criteria: |r| >= %.2f, p < %.4f, dist >= %.0f km...",
        min_r,
        max_p,
        min_dist_km,
    )
    dist_mask = dist_matrix >= min_dist_km
    significance_mask = (max_abs_r >= min_r) & (p_values < max_p) & dist_mask

    # Build symmetric adjacency matrix A = A^T
    # Nonzero entries receive peak absolute correlation
    A_dense = np.zeros((N, N), dtype=np.float32)
    A_dense[significance_mask] = max_abs_r[significance_mask]
    # Guarantee zero diagonal
    np.fill_diagonal(A_dense, 0.0)

    # Convert to sparse CSR matrix
    A_sparse = sparse.csr_matrix(A_dense)

    # 7. Extract edge list for undirected pairs (upper triangle i < j)
    triu_mask = np.triu(significance_mask, k=1)
    i_indices, j_indices = np.where(triu_mask)

    logger.info("Building edge objects for %d significant pairs...", len(i_indices))
    edges: List[TeleconnectionEdge] = []
    for i, j in zip(i_indices, j_indices):
        edges.append(
            TeleconnectionEdge(
                source_node_id=int(i),
                target_node_id=int(j),
                source_lat=float(lats_deg[i]),
                source_lon=float(lons_deg[i]),
                target_lat=float(lats_deg[j]),
                target_lon=float(lons_deg[j]),
                correlation=float(best_r[i, j]),
                abs_correlation=float(max_abs_r[i, j]),
                lag_days=float(best_lag_days[i, j]),
                p_value=float(p_values[i, j]),
                distance_km=float(dist_matrix[i, j]),
            )
        )

    # Sort edges descending by absolute correlation
    edges.sort(key=lambda e: e.abs_correlation, reverse=True)

    return TeleconnectionResult(
        adjacency_sparse=A_sparse,
        adjacency_dense=A_dense,
        edges=edges,
        nodes=nodes,
        edge_count=len(edges),
        node_count=N,
        max_abs_r=max_abs_r,
        best_r=best_r,
        best_lag_days=best_lag_days,
        distance_km=dist_matrix,
        p_values=p_values,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run planetary teleconnection lagged cross-correlation engine"
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("data/cache/grid.npz"),
        help="Path to grid.npz tensor archive (default: data/cache/grid.npz)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="z500",
        choices=["z500", "surface_pressure", "pressure_msl"],
        help="Target atmospheric variable (default: z500)",
    )
    parser.add_argument(
        "--min-r",
        type=float,
        default=0.65,
        help="Minimum absolute correlation threshold (default: 0.65)",
    )
    parser.add_argument(
        "--max-p",
        type=float,
        default=0.01,
        help="Maximum p-value threshold (default: 0.01)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=1000.0,
        help="Minimum distance in km to filter trivial spatial neighbors (default: 1000.0)",
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        default=7.0,
        help="Maximum lag in days for two-sided cross-correlation (default: 7.0)",
    )

    args = parser.parse_args()

    result = compute_teleconnections(
        npz_path=args.grid,
        variable_name=args.variable,
        min_r=args.min_r,
        max_p=args.max_p,
        min_dist_km=args.min_dist,
        max_lag_days=args.max_lag,
    )

    print("\n" + "=" * 78)
    print(" PLANETARY TELECONNECTION GRAPH ENGINE - EXECUTION REPORT")
    print("=" * 78)
    print(f" Variable Analyzed:       {args.variable}")
    print(f" Total Spatial Nodes:     {result.node_count}")
    print(f" Total Validated Edges:   {result.edge_count} undirected pairs")
    print(f" Adjacency Matrix Shape:  {result.adjacency_dense.shape} (Symmetric: A = A^T)")
    print(f" Nonzero Density:         {result.adjacency_sparse.nnz / (result.node_count**2) * 100.0:.2f}%")
    print(f" Correlation Threshold:   |r| >= {args.min_r:.2f}")
    print(f" Significance Threshold:  p < {args.max_p:.4f} (Quenouille Neff adjusted)")
    print(f" Distance Filter:         dist >= {args.min_dist:.0f} km (Haversine)")
    print(f" Lag Window:              [-{args.max_lag:.1f} .. +{args.max_lag:.1f}] days")
    print("-" * 78)
    print(" TOP 5 STRONGEST GLOBAL TELECONNECTIONS:")
    print("-" * 78)

    for rank, edge in enumerate(result.edges[:5], start=1):
        direction_str = (
            f"source leads by {edge.lag_days:+.1f}d"
            if edge.lag_days > 0
            else (f"target leads by {-edge.lag_days:+.1f}d" if edge.lag_days < 0 else "synchronous (0d)")
        )
        print(
            f" #{rank}: ({edge.source_lat:5.1f}°N, {edge.source_lon:6.1f}°E) [Node {edge.source_node_id:3d}]"
            f" <---> "
            f"({edge.target_lat:5.1f}°N, {edge.target_lon:6.1f}°E) [Node {edge.target_node_id:3d}]"
        )
        print(
            f"     |r| = {edge.abs_correlation:.4f} (signed r = {edge.correlation:+.4f}) | "
            f"Lag: {edge.lag_days:+.1f}d ({direction_str}) | "
            f"p = {edge.p_value:.2e} | Dist = {edge.distance_km:.1f} km"
        )
        print()

    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    main()
