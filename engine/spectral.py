"""
Spectral Graph Theory & Spectral Graph Wavelet (SGWT) Engine.
Constructs the normalized symmetric graph Laplacian L with pseudo-inverse handling for isolated nodes,
solves sparse eigenvalue decomposition (eigsh), computes algebraic connectivity (Fiedler value),
and implements Chebyshev polynomial approximations for Spectral Graph Wavelets.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from engine.teleconnection import compute_teleconnections, TeleconnectionResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spectral_engine")


@dataclass(slots=True)
class SpectralDecompositionResult:
    """Encapsulates graph Laplacian spectrum, Fiedler connectivity, and wavelet fields."""
    laplacian_sparse: sparse.csr_matrix
    spectral_radius: float
    fiedler_value: float
    eigenvalues: np.ndarray        # shape: (k,)
    eigenvectors: np.ndarray       # shape: (N, k)
    degrees: np.ndarray            # shape: (N,)
    active_mask: np.ndarray        # shape: (N,)
    wavelet_responses: np.ndarray  # shape: (num_scales, num_nodes, N)
    scales: np.ndarray             # shape: (num_scales,)
    test_node_ids: np.ndarray      # shape: (num_test_nodes,)
    decomp_time_sec: float
    wavelet_time_sec: float


def build_normalized_laplacian(
    adjacency_sparse: sparse.csr_matrix,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Construct symmetric normalized graph Laplacian L = I_active - D^(-1/2) A D^(-1/2)
    with pseudo-inverse handling for isolated nodes (d_i = 0 -> (D^-1/2)_i = 0).
    
    Args:
        adjacency_sparse: Symmetric adjacency matrix A (N, N).
        
    Returns:
        L: Sparse CSR normalized Laplacian matrix (N, N).
        degrees: 1D array of node degrees (N,).
        active_mask: 1D boolean array indicating nodes with degree > 0.
    """
    N = adjacency_sparse.shape[0]
    degrees = np.array(adjacency_sparse.sum(axis=1), dtype=np.float32).flatten()
    active_mask = degrees > 0.0

    d_inv_sqrt = np.zeros(N, dtype=np.float32)
    d_inv_sqrt[active_mask] = 1.0 / np.sqrt(degrees[active_mask])

    D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
    I_active = sparse.diags(active_mask.astype(np.float32), format="csr")

    # L = I_active - D^(-1/2) * A * D^(-1/2)
    L = I_active - (D_inv_sqrt @ adjacency_sparse @ D_inv_sqrt)
    return L, degrees, active_mask


def compute_laplacian_spectrum(
    L: sparse.csr_matrix,
    k: int = 20,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    """
    Compute spectral radius (lambda_max) and k smallest eigenvalues/eigenvectors of L.
    
    Returns:
        spectral_radius: Maximum eigenvalue lambda_max.
        fiedler_value: Algebraic connectivity (first strictly positive eigenvalue).
        eigenvalues: Sorted smallest k eigenvalues.
        eigenvectors: Corresponding orthonormal eigenvectors (N, k).
        elapsed_sec: Time taken for the spectral solver.
    """
    t0 = time.time()
    
    # 1. Largest eigenvalue (spectral radius)
    eval_max, _ = eigsh(L, k=1, which="LA")
    spectral_radius = float(eval_max[0])

    # 2. k smallest eigenvalues & eigenvectors
    evals, evecs = eigsh(L, k=k, which="SA")
    
    # Sort in ascending order
    sort_idx = np.argsort(evals)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Clean small numerical zeros
    evals = np.where(np.abs(evals) < 1e-10, 0.0, evals)

    # 3. Fiedler value (first strictly positive eigenvalue lambda > 1e-5)
    positive_evals = evals[evals > 1e-5]
    fiedler_value = float(positive_evals[0]) if len(positive_evals) > 0 else 0.0

    elapsed_sec = time.time() - t0
    return spectral_radius, fiedler_value, evals, evecs, elapsed_sec


def default_wavelet_kernel(x: np.ndarray) -> np.ndarray:
    """
    Band-pass continuous wavelet kernel: g(x) = C * x * exp(-x).
    Normalized with C = e so that peak value equals 1.0 at x = 1.0.
    """
    x_pos = np.maximum(x, 0.0)
    return np.e * x_pos * np.exp(-x_pos)


def compute_chebyshev_coefficients(
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    scale: float,
    lambda_max: float,
    order: int = 30,
    quad_nodes: int = 80,
) -> np.ndarray:
    """
    Compute Chebyshev expansion coefficients c_m for scaled filter kernel h(lambda) = g(s * lambda)
    using Gauss-Chebyshev quadrature over [-1, 1].
    
    Mapping:
        lambda = (lambda_max / 2) * (y + 1), for y in [-1, 1].
    """
    K = max(quad_nodes, order + 20)
    k_idx = np.arange(K, dtype=np.float64)
    # Chebyshev nodes
    y_k = np.cos(np.pi * (k_idx + 0.5) / K)

    # Map y in [-1, 1] to lambda in [0, lambda_max]
    lam_k = (lambda_max / 2.0) * (y_k + 1.0)
    h_k = kernel_fn(scale * lam_k)

    # Quadrature summation for coefficients c_m
    coeffs = np.zeros(order + 1, dtype=np.float64)
    for m in range(order + 1):
        coeffs[m] = (2.0 / K) * np.sum(h_k * np.cos(m * np.pi * (k_idx + 0.5) / K))

    return coeffs


def chebyshev_sgwt_filter(
    L: sparse.csr_matrix,
    signal: np.ndarray,
    scale: float,
    lambda_max: float,
    order: int = 30,
    kernel_fn: Callable[[np.ndarray], np.ndarray] = default_wavelet_kernel,
) -> np.ndarray:
    """
    Apply Spectral Graph Wavelet filter T_g = g(sL) to an input graph signal using Clenshaw /
    vectorized Chebyshev polynomial recurrence on the shifted operator L_tilde.
    
    Complexity: O(order * |E|) sparse matrix-vector multiplications.
    
    Args:
        L: Sparse normalized Laplacian matrix (N, N).
        signal: Input vector (N,) or unit impulse delta_i.
        scale: Wavelet scaling factor s > 0.
        lambda_max: Upper spectral bound (spectral radius of L).
        order: Truncation order M of Chebyshev polynomials (default: 30).
        kernel_fn: Band-pass kernel function g(x).
        
    Returns:
        Filtered wavelet response vector (N,).
    """
    N = L.shape[0]
    coeffs = compute_chebyshev_coefficients(kernel_fn, scale, lambda_max, order=order)

    # Shifted Laplacian L_tilde = (2 / lambda_max) * L - I
    # Spectrum of L_tilde lies strictly within [-1, 1]
    scale_factor = 2.0 / lambda_max
    I_full = sparse.eye(N, format="csr", dtype=np.float32)
    L_tilde = (scale_factor * L - I_full).tocsr()

    # Clenshaw backward recurrence:
    # y_{M+2} = 0, y_{M+1} = 0
    # for m = M down to 1:
    #     y_m = 2 * L_tilde * y_{m+1} - y_{m+2} + c_m * signal
    # y_0 = L_tilde * y_1 - y_2 + 0.5 * c_0 * signal
    y_m2 = np.zeros(N, dtype=np.float64)
    y_m1 = np.zeros(N, dtype=np.float64)
    v = signal.astype(np.float64)

    for m in range(order, 0, -1):
        y_m = 2.0 * (L_tilde @ y_m1) - y_m2 + (coeffs[m] * v)
        y_m2 = y_m1
        y_m1 = y_m

    # Final step for m = 0 (factor 1/2 on c_0)
    result = (L_tilde @ y_m1) - y_m2 + (0.5 * coeffs[0] * v)
    return result.astype(np.float32)


def compute_wavelet_atom(
    L: sparse.csr_matrix,
    node_id: int,
    scale: float,
    lambda_max: float,
    order: int = 30,
) -> np.ndarray:
    """Compute localized wavelet atom psi_{s, i} = g(sL) * delta_i centered at node_id."""
    N = L.shape[0]
    delta = np.zeros(N, dtype=np.float32)
    delta[node_id] = 1.0
    return chebyshev_sgwt_filter(L, delta, scale=scale, lambda_max=lambda_max, order=order)


def run_spectral_analysis(
    grid_path: str | Path = "data/cache/grid.npz",
    output_path: str | Path = "data/cache/spectral.npz",
    k_eigenvalues: int = 20,
    chebyshev_order: int = 30,
    scales: Optional[List[float]] = None,
    test_node_ids: Optional[List[int]] = None,
) -> SpectralDecompositionResult:
    """
    Full pipeline: runs teleconnections, builds normalized Laplacian, solves sparse spectrum,
    computes localized SGWT wavelet fields, and saves results to data/cache/spectral.npz.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if scales is None:
        scales = [2.0, 5.0, 10.0]  # High, medium, low frequency wavelet bands
    if test_node_ids is None:
        # Default to top teleconnection hubs (Node 652: Alaska, Node 179: Morocco/Azores)
        test_node_ids = [652, 179]

    # 1. Run teleconnection graph engine
    logger.info("Computing teleconnection graph adjacency from '%s'...", grid_path)
    tele_res = compute_teleconnections(
        npz_path=grid_path,
        min_r=0.65,
        max_p=0.01,
        min_dist_km=1000.0,
        max_lag_days=7.0,
    )
    A = tele_res.adjacency_sparse
    N = A.shape[0]

    # 2. Build normalized Laplacian with isolated node pseudo-inverse
    logger.info("Constructing symmetric normalized Laplacian L...")
    L, degrees, active_mask = build_normalized_laplacian(A)
    logger.info(
        "Laplacian built: %d nodes, %d active connected nodes (%d isolated)",
        N,
        int(np.sum(active_mask)),
        int(N - np.sum(active_mask)),
    )

    # 3. Compute spectrum via eigsh
    logger.info("Solving sparse eigenvalue decomposition for k=%d smallest modes...", k_eigenvalues)
    lambda_max, fiedler_val, evals, evecs, decomp_time = compute_laplacian_spectrum(
        L, k=k_eigenvalues
    )
    logger.info(
        "Decomposition finished in %.3fs: spectral radius=%.4f, Fiedler value=%.5f",
        decomp_time,
        lambda_max,
        fiedler_val,
    )

    # 4. Compute Chebyshev SGWT wavelets
    logger.info(
        "Computing SGWT wavelet fields (Chebyshev order M=%d, scales=%s, nodes=%s)...",
        chebyshev_order,
        scales,
        test_node_ids,
    )
    t_wav0 = time.time()
    num_scales = len(scales)
    num_nodes = len(test_node_ids)
    wavelet_cube = np.zeros((num_scales, num_nodes, N), dtype=np.float32)

    for s_idx, s in enumerate(scales):
        for n_idx, node_id in enumerate(test_node_ids):
            psi = compute_wavelet_atom(
                L,
                node_id=node_id,
                scale=s,
                lambda_max=lambda_max,
                order=chebyshev_order,
            )
            wavelet_cube[s_idx, n_idx, :] = psi

    wavelet_time = time.time() - t_wav0
    logger.info("Computed %d wavelet fields in %.3fs", num_scales * num_nodes, wavelet_time)

    # 5. Save to .npz cache
    logger.info("Saving spectral fields to '%s'...", output_file)
    np.savez_compressed(
        output_file,
        eigenvalues=evals.astype(np.float32),
        eigenvectors=evecs.astype(np.float32),
        spectral_radius=np.array(lambda_max, dtype=np.float32),
        fiedler_value=np.array(fiedler_val, dtype=np.float32),
        degrees=degrees.astype(np.float32),
        active_mask=active_mask.astype(bool),
        wavelet_responses=wavelet_cube,
        scales=np.array(scales, dtype=np.float32),
        test_node_ids=np.array(test_node_ids, dtype=np.int32),
        decomp_time_sec=np.array(decomp_time, dtype=np.float32),
        wavelet_time_sec=np.array(wavelet_time, dtype=np.float32),
    )

    file_size_kb = output_file.stat().st_size / 1024.0
    logger.info("Successfully written '%s' (%.1f KB on disk)", output_file, file_size_kb)

    return SpectralDecompositionResult(
        laplacian_sparse=L,
        spectral_radius=lambda_max,
        fiedler_value=fiedler_val,
        eigenvalues=evals,
        eigenvectors=evecs,
        degrees=degrees,
        active_mask=active_mask,
        wavelet_responses=wavelet_cube,
        scales=np.array(scales, dtype=np.float32),
        test_node_ids=np.array(test_node_ids, dtype=np.int32),
        decomp_time_sec=decomp_time,
        wavelet_time_sec=wavelet_time,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run planetary spectral graph theory & SGWT engine"
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("data/cache/grid.npz"),
        help="Input grid.npz tensor archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/spectral.npz"),
        help="Target spectral.npz output archive",
    )
    parser.add_argument(
        "-k",
        "--eigenvalues",
        type=int,
        default=20,
        help="Number of smallest eigenvalues to solve (default: 20)",
    )
    parser.add_argument(
        "-M",
        "--chebyshev-order",
        type=int,
        default=30,
        help="Chebyshev polynomial approximation order (default: 30)",
    )

    args = parser.parse_args()

    res = run_spectral_analysis(
        grid_path=args.grid,
        output_path=args.output,
        k_eigenvalues=args.eigenvalues,
        chebyshev_order=args.chebyshev_order,
    )

    print("\n" + "=" * 78)
    print(" SPECTRAL GRAPH & SGWT WAVELET ENGINE - EXECUTION REPORT")
    print("=" * 78)
    print(f" Spectral Radius (lambda_max): {res.spectral_radius:.5f} (Theory bound: <= 2.0)")
    print(f" Fiedler Value (lambda_2):     {res.fiedler_value:.5f} (Global Algebraic Connectivity)")
    print(f" Total Spatial Nodes (N):     {len(res.degrees)}")
    print(f" Active Connected Nodes:      {int(np.sum(res.active_mask))}/{len(res.degrees)}")
    print(f" Eigendecomposition Time:     {res.decomp_time_sec:.3f} s (via scipy.sparse.linalg.eigsh)")
    print(f" SGWT Wavelet Transform Time: {res.wavelet_time_sec:.3f} s (Chebyshev M={args.chebyshev_order})")
    print(f" Cache Archive File:          {args.output.resolve()}")
    print("-" * 78)
    print(" SMALLEST 10 LAPLACIAN EIGENVALUES:")
    print("-" * 78)
    for idx, lam in enumerate(res.eigenvalues[:10], start=1):
        label = " (trivial mode)" if lam < 1e-5 else (" (Fiedler value)" if idx == 5 else "")
        print(f"  lambda_{idx:02d} = {lam:.6f}{label}")

    print("-" * 78)
    print(" SGWT LOCALIZED WAVELET RESPONSES (ENERGY SPREAD):")
    print("-" * 78)
    for s_idx, s in enumerate(res.scales):
        for n_idx, node_id in enumerate(res.test_node_ids):
            psi = res.wavelet_responses[s_idx, n_idx]
            energy = np.sum(psi**2)
            peak_val = np.max(np.abs(psi))
            peak_node = int(np.argmax(np.abs(psi)))
            print(
                f"  Scale s={s:4.1f} | Node {node_id:3d}: "
                f"Peak = {peak_val:.4f} at Node {peak_node:3d} | "
                f"L2 Energy = {energy:.4f}"
            )

    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    main()
