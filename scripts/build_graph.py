"""
CLI script to construct teleconnection graph, filter edges, and populate SQLite database.
Computes lagged cross-correlations, Haversine distance filtering (d >= 1000 km),
Quenouille/Davis Neff degrees of freedom, and inserts results into data/teleconnections.db.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.init_db import initialize_database

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_graph")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct planetary teleconnection graph and populate SQLite database."
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("data/cache/grid.npz"),
        help="Path to input grid tensor (.npz) cached by fetch_data.py",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/teleconnections.db"),
        help="Path to output SQLite database",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/schema.sql"),
        help="Path to SQLite schema file",
    )
    parser.add_argument(
        "--spectral",
        type=Path,
        default=Path("data/cache/spectral.npz"),
        help="Path to spectral metrics cache (.npz)",
    )
    args = parser.parse_args()

    logger.info("Starting teleconnection graph construction...")
    return initialize_database(
        db_path=args.db,
        schema_path=args.schema,
        grid_path=args.grid,
        spectral_path=args.spectral,
    )


if __name__ == "__main__":
    sys.exit(main())
