"""
Unit and integration tests for AeroGraph Forecasting Engine and API.
Uses Python standard unittest library.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from engine.forecast import (
    compute_climate_indices,
    predict_node_trajectory,
    compute_all_nodes_predictability,
)


class TestClimateForecastingEngine(unittest.TestCase):
    """Test suite verifying mathematical rigor and edge-case resilience of forecast.py."""

    def test_compute_climate_indices_real_data(self):
        """Verify macro climate indices compute valid floats on real Z500 cache."""
        res = compute_climate_indices()
        self.assertIn("nao", res)
        self.assertIn("ao", res)
        self.assertIn("timestamp", res)

        nao = res["nao"]
        ao = res["ao"]

        # Validate NAO
        self.assertIsInstance(nao["value"], float)
        self.assertFalse(math.isnan(nao["value"]))
        self.assertFalse(math.isinf(nao["value"]))
        self.assertIn(nao["phase"], ["+", "-"])
        self.assertEqual(len(nao["azores_coords"]), 2)
        self.assertEqual(len(nao["iceland_coords"]), 2)

        # Validate AO
        self.assertIsInstance(ao["value"], float)
        self.assertFalse(math.isnan(ao["value"]))
        self.assertFalse(math.isinf(ao["value"]))
        self.assertIn(ao["phase"], ["+", "-"])

    def test_predict_node_trajectory_connected_node(self):
        """Test prediction for node 780 with incoming teleconnection predictors."""
        res = predict_node_trajectory(780, horizon_days=5)
        self.assertEqual(res["node_id"], 780)
        self.assertEqual(res["horizon_days"], 5)
        self.assertGreater(res["num_predictors"], 0)
        self.assertGreaterEqual(res["r2"], 0.40)
        self.assertEqual(len(res["forecast_values"]), 5)
        self.assertEqual(len(res["confidence_lower"]), 5)
        self.assertEqual(len(res["confidence_upper"]), 5)

        for i in range(5):
            self.assertIsInstance(res["forecast_values"][i], float)
            self.assertIsInstance(res["confidence_lower"][i], float)
            self.assertIsInstance(res["confidence_upper"][i], float)
            self.assertLessEqual(res["confidence_lower"][i], res["forecast_values"][i])
            self.assertGreaterEqual(res["confidence_upper"][i], res["forecast_values"][i])

        self.assertIn("Ожидается", res["synoptic_verdict"])

    def test_predict_node_trajectory_zero_leading_connections(self):
        """Test prediction for node 0 having 0 leading connections (tau > 0, |r| >= 0.65)."""
        res = predict_node_trajectory(0, horizon_days=5)
        self.assertEqual(res["node_id"], 0)
        self.assertEqual(res["num_predictors"], 0)
        self.assertEqual(res["r2"], 0.45)
        self.assertEqual(len(res["forecast_values"]), 5)
        self.assertIn("затухающей инерции", res["synoptic_verdict"])

        for val in res["forecast_values"] + res["confidence_lower"] + res["confidence_upper"]:
            self.assertFalse(math.isnan(val))
            self.assertFalse(math.isinf(val))

    def test_predict_node_trajectory_completely_isolated_node(self):
        """Test prediction for node 45 having 0 teleconnection arcs at all."""
        res = predict_node_trajectory(45, horizon_days=5)
        self.assertEqual(res["node_id"], 45)
        self.assertEqual(res["num_predictors"], 0)
        self.assertEqual(res["r2"], 0.45)
        self.assertEqual(len(res["forecast_values"]), 5)

    def test_predict_node_trajectory_max_connections_node(self):
        """Test prediction for hub node 719 with maximum leading connections."""
        res = predict_node_trajectory(719, horizon_days=5)
        self.assertEqual(res["node_id"], 719)
        self.assertGreaterEqual(res["num_predictors"], 15)

    def test_predict_node_trajectory_short_horizon_edge_cases(self):
        """Test horizon_days=1 and horizon_days=2 to ensure no IndexError on index 2."""
        for h in [1, 2, 3]:
            res = predict_node_trajectory(780, horizon_days=h)
            self.assertEqual(len(res["forecast_values"]), h)
            self.assertIn(f"к {h}-м суткам", res["synoptic_verdict"])

    def test_predict_node_trajectory_invalid_node_id(self):
        """Test that invalid node IDs properly raise ValueError."""
        with self.assertRaises(ValueError):
            predict_node_trajectory(-1)
        with self.assertRaises(ValueError):
            predict_node_trajectory(9999)

    def test_predict_node_trajectory_invalid_horizon(self):
        """Test that invalid horizons properly raise ValueError."""
        with self.assertRaises(ValueError):
            predict_node_trajectory(780, horizon_days=0)
        with self.assertRaises(ValueError):
            predict_node_trajectory(780, horizon_days=-5)
        with self.assertRaises(ValueError):
            predict_node_trajectory(780, horizon_days=31)

    def test_compute_all_nodes_predictability(self):
        """Test global predictability statistics across all 936 nodes."""
        preds = compute_all_nodes_predictability()
        self.assertEqual(len(preds), 936)

        for p in preds:
            self.assertIsInstance(p["id"], int)
            self.assertIsInstance(p["lat"], float)
            self.assertIsInstance(p["lon"], float)
            self.assertIsInstance(p["num_predictors"], int)
            self.assertIsInstance(p["total_weight"], float)
            self.assertIsInstance(p["r2"], float)
            self.assertFalse(math.isnan(p["r2"]))
            self.assertGreaterEqual(p["r2"], 0.35)
            self.assertLessEqual(p["r2"], 0.92)


if __name__ == "__main__":
    unittest.main()
