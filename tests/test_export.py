"""
Unit and integration tests for AeroGraph Analytical Export & Reporting Engine.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

from engine.export import generate_climate_report


class TestExportEngine(unittest.TestCase):
    """Test suite verifying analytical export generation, chart validity, and datasets."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="aerograph_test_export_")
        cls.zip_path = Path(generate_climate_report(output_dir=cls.test_dir))
        cls.extract_dir = Path(cls.test_dir) / "unpacked"
        cls.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(cls.zip_path, "r") as zf:
            zf.extractall(cls.extract_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_zip_exists_and_non_empty(self):
        """Verify zip archive is created and is reasonably sized (> 1 MB)."""
        self.assertTrue(self.zip_path.exists())
        self.assertGreater(self.zip_path.stat().st_size, 1024 * 1024)

    def test_report_files_and_links(self):
        """Verify REPORT.md and REPORT.html contain valid references to all 7 charts."""
        md_path = self.extract_dir / "REPORT.md"
        html_path = self.extract_dir / "REPORT.html"

        self.assertTrue(md_path.exists())
        self.assertTrue(html_path.exists())

        md_text = md_path.read_text(encoding="utf-8")
        html_text = html_path.read_text(encoding="utf-8")

        md_links = re.findall(r'!\[.*?\]\((.*?)\)', md_text)
        html_links = re.findall(r'<img\s+[^>]*src=[\'"](.*?)[\'"]', html_text)

        self.assertEqual(len(md_links), 7)
        self.assertEqual(len(html_links), 7)

        for link in md_links:
            target = self.extract_dir / link
            self.assertTrue(target.exists(), f"Broken MD chart link: {link}")

        for link in html_links:
            target = self.extract_dir / link
            self.assertTrue(target.exists(), f"Broken HTML chart link: {link}")

    def test_all_seven_charts(self):
        """Verify all 7 charts exist, are > 10 KB, and open properly with PIL."""
        charts_dir = self.extract_dir / "charts"
        expected_charts = [
            "01_teleconnection_map.png",
            "02_laplacian_spectrum.png",
            "03_distance_vs_correlation.png",
            "04_lag_distribution.png",
            "05_wavelet_energy_decay.png",
            "06_climate_indices_phases.png",
            "07_forecast_lead_time_skill.png",
        ]
        for name in expected_charts:
            chart_file = charts_dir / name
            self.assertTrue(chart_file.exists(), f"Missing chart {name}")
            self.assertGreater(chart_file.stat().st_size, 10 * 1024, f"Chart {name} smaller than 10KB")
            with Image.open(chart_file) as img:
                img.verify()
            with Image.open(chart_file) as img:
                self.assertEqual(img.format, "PNG")
                self.assertGreater(img.size[0], 2000)
                self.assertGreater(img.size[1], 1000)

    def test_csv_datasets_integrity(self):
        """Verify tabular CSV files have valid headers, row counts, and no NaN or empty cells."""
        expected_csvs = [
            "data/nodes_predictability.csv",
            "data/nodes_summary.csv",
            "data/significant_edges.csv",
        ]
        for rel in expected_csvs:
            csv_path = self.extract_dir / rel
            self.assertTrue(csv_path.exists())
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertGreater(len(header), 0)
                row_count = 0
                for row in reader:
                    row_count += 1
                    self.assertEqual(len(row), len(header), f"Row mismatch in {rel}")
                    for cell in row:
                        val = cell.strip()
                        self.assertNotEqual(val, "", f"Empty cell in {rel}")
                        self.assertNotIn(val.lower(), ["nan", "null", "none"], f"NaN/null cell in {rel}")
                self.assertGreater(row_count, 100)

    def test_json_datasets_integrity(self):
        """Verify climate_indices.json and summary_metrics.json."""
        indices_path = self.extract_dir / "data" / "climate_indices.json"
        metrics_path = self.extract_dir / "data" / "summary_metrics.json"

        self.assertTrue(indices_path.exists())
        self.assertTrue(metrics_path.exists())

        with open(indices_path, "r", encoding="utf-8") as f:
            ci = json.load(f)
        self.assertIn("nao", ci)
        self.assertIn("ao", ci)
        self.assertIsInstance(ci["nao"]["value"], float)
        self.assertIsInstance(ci["ao"]["value"], float)

        with open(metrics_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        self.assertIn("spectral_radius", sm)
        self.assertIn("fiedler_value", sm)
        self.assertIn("edge_count", sm)
        self.assertGreater(sm["edge_count"], 1000)


if __name__ == "__main__":
    unittest.main()
