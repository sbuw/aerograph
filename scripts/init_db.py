"""
Database initialization and data migration script.
Initializes SQLite database from data/schema.sql, inserts spatial nodes,
epochs, and pruned teleconnection edges into data/teleconnections.db.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from engine.spectral import run_spectral_analysis
from engine.teleconnection import compute_teleconnections

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_db")

SCHEMA_PATH = Path("data/schema.sql")
DB_PATH = Path("data/teleconnections.db")
GRID_PATH = Path("data/cache/grid.npz")
SPECTRAL_PATH = Path("data/cache/spectral.npz")


def initialize_database(
    db_path: Path = DB_PATH,
    schema_path: Path = SCHEMA_PATH,
    grid_path: Path = GRID_PATH,
    spectral_path: Path = SPECTRAL_PATH,
) -> int:
    """Initialize SQLite database schema and populate nodes, epoch, and edges."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid cache file not found: {grid_path}")

    # 1. Connect and initialize tables
    logger.info("Initializing database schema at '%s'...", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)

    # 2. Populate graph_nodes
    logger.info("Loading grid nodes from '%s'...", grid_path)
    with np.load(grid_path) as g_data:
        nodes_arr = g_data["nodes"]  # shape (N, 5): id, lat, lon, grid_i, grid_j
        timestamps = g_data["timestamps"]
        past_days = int(g_data["past_days"])

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM graph_nodes;")
    node_count_existing = cursor.fetchone()[0]

    nodes_tuples = [
        (int(row[0]), float(row[1]), float(row[2]), int(row[3]), int(row[4]))
        for row in nodes_arr
    ]
    cursor.executemany(
        """
        INSERT OR REPLACE INTO graph_nodes (id, lat, lon, grid_i, grid_j)
        VALUES (?, ?, ?, ?, ?)
        """,
        nodes_tuples,
    )
    logger.info("Inserted/updated %d spatial nodes in 'graph_nodes'", len(nodes_tuples))

    # 3. Load or compute teleconnections and spectral metrics
    logger.info("Computing teleconnection edges...")
    tele_res = compute_teleconnections(grid_path, min_r=0.65, max_p=0.01)

    if spectral_path.exists():
        logger.info("Loading spectral metrics from '%s'...", spectral_path)
        with np.load(spectral_path) as s_data:
            spectral_radius = float(s_data["spectral_radius"])
            fiedler_value = float(s_data["fiedler_value"])
    else:
        logger.info("Computing spectral metrics...")
        spec_res = run_spectral_analysis(grid_path=grid_path, output_path=spectral_path)
        spectral_radius = spec_res.spectral_radius
        fiedler_value = spec_res.fiedler_value

    # 4. Insert computation epoch
    latest_ts = str(timestamps[-1])
    logger.info(
        "Inserting computation epoch: timestamp=%s, window=%dd, edges=%d, spectral_radius=%.4f, fiedler=%.5f",
        latest_ts,
        past_days,
        tele_res.edge_count,
        spectral_radius,
        fiedler_value,
    )
    cursor.execute(
        """
        INSERT INTO computation_epochs (timestamp, window_days, node_count, edge_count, spectral_radius, fiedler_value)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            latest_ts,
            past_days,
            tele_res.node_count,
            tele_res.edge_count,
            spectral_radius,
            fiedler_value,
        ),
    )
    epoch_id = cursor.lastrowid
    logger.info("Epoch created with ID: %d", epoch_id)

    # 5. Bulk insert teleconnection edges
    logger.info("Inserting %d teleconnection edges into 'teleconnection_edges'...", len(tele_res.edges))
    edge_tuples = [
        (
            epoch_id,
            e.source_node_id,
            e.target_node_id,
            e.correlation,
            int(round(e.lag_days)),
            e.p_value,
            e.distance_km,
        )
        for e in tele_res.edges
    ]

    cursor.executemany(
        """
        INSERT INTO teleconnection_edges (epoch_id, source_node_id, target_node_id, correlation, lag_days, p_value, distance_km)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        edge_tuples,
    )

    conn.commit()
    conn.close()

    # Verification query
    conn_verify = sqlite3.connect(db_path)
    cur_v = conn_verify.cursor()
    cur_v.execute("SELECT COUNT(*) FROM graph_nodes;")
    total_nodes = cur_v.fetchone()[0]
    cur_v.execute("SELECT COUNT(*) FROM computation_epochs;")
    total_epochs = cur_v.fetchone()[0]
    cur_v.execute("SELECT COUNT(*) FROM teleconnection_edges WHERE epoch_id = ?;", (epoch_id,))
    total_edges = cur_v.fetchone()[0]
    conn_verify.close()

    print("\n" + "=" * 65)
    print(" SQLITE DATABASE INITIALIZATION REPORT")
    print("=" * 65)
    print(f" Database File:        {db_path.resolve()}")
    print(f" Database Size:        {db_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f" Table graph_nodes:    {total_nodes} rows")
    print(f" Table epochs:         {total_epochs} rows (Epoch ID: {epoch_id})")
    print(f" Table edges:          {total_edges} rows for epoch {epoch_id}")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(initialize_database())
