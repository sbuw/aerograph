-- data/schema.sql
-- SQLite schema for graph run metadata and topology indices

CREATE TABLE IF NOT EXISTS computation_epochs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    window_days INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    edge_count INTEGER NOT NULL,
    spectral_radius REAL,
    fiedler_value REAL, -- Algebraic connectivity (second-smallest Laplacian eigenvalue)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id INTEGER PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    grid_i INTEGER NOT NULL,
    grid_j INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teleconnection_edges (
    epoch_id INTEGER NOT NULL,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER NOT NULL,
    correlation REAL NOT NULL,
    lag_days INTEGER NOT NULL,
    p_value REAL NOT NULL,
    distance_km REAL NOT NULL,
    FOREIGN KEY(epoch_id) REFERENCES computation_epochs(id),
    FOREIGN KEY(source_node_id) REFERENCES graph_nodes(id),
    FOREIGN KEY(target_node_id) REFERENCES graph_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_edges_epoch ON teleconnection_edges(epoch_id);
CREATE INDEX IF NOT EXISTS idx_edges_src_dst ON teleconnection_edges(source_node_id, target_node_id);