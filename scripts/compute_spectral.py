"""
CLI script for graph Laplacian spectral decomposition and SGWT wavelet pre-caching.
Constructs normalized symmetric Laplacian L, solves sparse eigensystem (eigsh),
computes algebraic connectivity (Fiedler value), and pre-evaluates Chebyshev SGWT wavelets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.spectral import main as spectral_main

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compute_spectral")


def main() -> int:
    return spectral_main()


if __name__ == "__main__":
    sys.exit(main())
