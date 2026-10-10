"""
QA Audit Script for AeroGraph Export Dump & Report Generation.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from PIL import Image

def run_audit(zip_path: Path) -> dict:
    print(f"=== Starting QA Audit on {zip_path} ===")
    assert zip_path.exists(), f"Archive {zip_path} does not exist!"

    temp_dir = Path("exports/qa_unpacked")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    results = {
        "files": {},
        "links_ok": True,
        "charts_ok": True,
        "csvs_ok": True,
        "jsons_ok": True,
        "issues": [],
    }

    # 1. Collect all files and sizes
    for p in sorted(temp_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(temp_dir)).replace("\\", "/")
            size = p.stat().st_size
            results["files"][rel] = size
            print(f"File: {rel:<40} Size: {size:>10} bytes ({size/1024:.2f} KB)")

    # 2. Check REPORT.md and REPORT.html links
    md_file = temp_dir / "REPORT.md"
    html_file = temp_dir / "REPORT.html"

    if not md_file.exists():
        results["issues"].append("REPORT.md missing")
        results["links_ok"] = False
    else:
        md_text = md_file.read_text(encoding="utf-8")
        md_links = re.findall(r'!\[.*?\]\((.*?)\)', md_text)
        print(f"\nREPORT.md chart links found ({len(md_links)}): {md_links}")
        for link in md_links:
            target = temp_dir / link
            if not target.exists():
                results["issues"].append(f"Broken link in REPORT.md: {link}")
                results["links_ok"] = False

    if not html_file.exists():
        results["issues"].append("REPORT.html missing")
        results["links_ok"] = False
    else:
        html_text = html_file.read_text(encoding="utf-8")
        html_links = re.findall(r'<img\s+[^>]*src=[\'"](.*?)[\'"]', html_text)
        print(f"REPORT.html chart links found ({len(html_links)}): {html_links}")
        for link in html_links:
            target = temp_dir / link
            if not target.exists():
                results["issues"].append(f"Broken link in REPORT.html: {link}")
                results["links_ok"] = False

    # Check chart coverage
    charts_dir = temp_dir / "charts"
    chart_files = sorted([f.name for f in charts_dir.glob("*.png")]) if charts_dir.exists() else []
    print(f"\nPhysical charts in charts/ ({len(chart_files)}): {chart_files}")

    expected_charts = [
        "01_teleconnection_map.png",
        "02_laplacian_spectrum.png",
        "03_distance_vs_correlation.png",
        "04_lag_distribution.png",
        "05_wavelet_energy_decay.png",
        "06_climate_indices_phases.png",
        "07_forecast_lead_time_skill.png",
    ]
    for ec in expected_charts:
        if ec not in chart_files:
            results["issues"].append(f"Missing expected chart: {ec}")
            results["charts_ok"] = False

    # 3. Check Chart images
    print("\n--- Charts Quality Check ---")
    for cf_name in chart_files:
        cf_path = charts_dir / cf_name
        size_kb = cf_path.stat().st_size / 1024.0
        if size_kb < 10.0:
            results["issues"].append(f"Chart {cf_name} is too small: {size_kb:.2f} KB (< 10 KB)")
            results["charts_ok"] = False
        try:
            with Image.open(cf_path) as img:
                img.verify()
            with Image.open(cf_path) as img:
                w, h = img.size
                fmt = img.format
                print(f"Chart {cf_name:<35}: {size_kb:>7.1f} KB, format={fmt}, res={w}x{h}")
        except Exception as e:
            results["issues"].append(f"Chart {cf_name} failed PIL check: {e}")
            results["charts_ok"] = False

    # 4. Check CSV files
    print("\n--- CSV Data Quality Check ---")
    csv_paths = list(temp_dir.glob("data/*.csv"))
    for cp in sorted(csv_paths):
        rel = str(cp.relative_to(temp_dir)).replace("\\", "/")
        with open(cp, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                results["issues"].append(f"Empty CSV file: {rel}")
                results["csvs_ok"] = False
                continue

            rows = 0
            nans = 0
            empties = 0
            bad_row_lens = 0
            for r_idx, row in enumerate(reader):
                rows += 1
                if len(row) != len(header):
                    bad_row_lens += 1
                for cell in row:
                    val = cell.strip()
                    if val == "":
                        empties += 1
                    if val.lower() in ("nan", "none", "null", "inf", "-inf"):
                        nans += 1

            print(f"CSV {rel:<30}: {rows:>5} rows, {len(header)} cols, headers={header}")
            if bad_row_lens > 0:
                results["issues"].append(f"CSV {rel} has {bad_row_lens} rows with mismatched column count")
                results["csvs_ok"] = False
            if nans > 0:
                results["issues"].append(f"CSV {rel} has {nans} NaN/null values")
                results["csvs_ok"] = False
            if empties > 0:
                results["issues"].append(f"CSV {rel} has {empties} empty cell values")
                results["csvs_ok"] = False

    # 5. Check JSON files
    print("\n--- JSON Data Quality Check ---")
    json_paths = list(temp_dir.glob("data/*.json"))
    for jp in sorted(json_paths):
        rel = str(jp.relative_to(temp_dir)).replace("\\", "/")
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"JSON {rel:<30}: valid JSON, {len(data)} root keys ({list(data.keys())})")
            if "climate_indices.json" in rel:
                if "nao" not in data or "ao" not in data:
                    results["issues"].append("climate_indices.json missing 'nao' or 'ao'")
                    results["jsons_ok"] = False
                else:
                    print(f"  NAO: {data['nao']}")
                    print(f"  AO:  {data['ao']}")
        except Exception as e:
            results["issues"].append(f"JSON {rel} parse error: {e}")
            results["jsons_ok"] = False

    return results

if __name__ == "__main__":
    zip_path = Path("exports/climate_report_2026-10-11.zip")
    res = run_audit(zip_path)
    print("\n=== Audit Summary ===")
    print("Links OK: ", res["links_ok"])
    print("Charts OK:", res["charts_ok"])
    print("CSVs OK:  ", res["csvs_ok"])
    print("JSONs OK: ", res["jsons_ok"])
    print("Issues:   ", res["issues"])
