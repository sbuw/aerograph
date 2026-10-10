"""
HTTP test client verifying all FastAPI endpoints across Phases 1-5.
Tests nodes, edges, spectrum, wavelets, climate indices, forecast, and report export.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"


def test_endpoint(title: str, path: str, is_binary: bool = False):
    url = f"{BASE_URL}{path}"
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "TestClient/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        dt = (time.time() - t0) * 1000.0
        raw = resp.read()

        print(f"\n{'='*70}\n {title}\n{'='*70}")
        print(f"URL:     {url}")
        print(f"Status:  {resp.status} OK | Latency: {dt:.1f} ms | Payload: {len(raw)/1024:.1f} KB")

        if is_binary:
            content_type = resp.headers.get("Content-Type", "")
            content_disp = resp.headers.get("Content-Disposition", "")
            print(f"Binary Stream: Content-Type={content_type}, Disposition={content_disp}")
            return

        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, list):
            print(f"Items:   {len(data)} items returned")
            print(f"Sample:  {data[0]}")
        elif isinstance(data, dict):
            preview = {
                k: (f"[{len(v)} items]" if isinstance(v, list) else v)
                for k, v in data.items()
                if k != "psi"
            }
            print(f"Summary: {preview}")
            if "edges" in data and len(data["edges"]) > 0:
                print(f"Sample edge: {data['edges'][0]}")
            if "indices" in data:
                print(f"Indices: {[idx.get('code') for idx in data['indices']]}")
            if "forecast" in data and len(data["forecast"]) > 0:
                print(f"Sample forecast step: {data['forecast'][0]}")
            if "psi" in data:
                print(f"Psi vector: {len(data['psi'])} values (first 5: {data['psi'][:5]})")


if __name__ == "__main__":
    time.sleep(1)
    test_endpoint("1. Health Check", "/api/health")
    test_endpoint("2. Grid Nodes (N=936)", "/api/nodes")
    test_endpoint("3. Global Teleconnection Arcs", "/api/edges?min_r=0.65&limit=5")
    test_endpoint("4. Incident Arcs for Clicked Node 652", "/api/node/652/teleconnections")
    test_endpoint("5. Laplacian Spectrum & Fiedler Value", "/api/spectrum")
    test_endpoint("6. Dynamic Chebyshev SGWT Wavelet (Node 652, Scale 5.0)", "/api/wavelet?node_id=652&scale=5.0")
    test_endpoint("7. Planetary Climate Indices (NAO & AO)", "/api/climate-indices")
    test_endpoint("8. Lead-Lag Trajectory Forecast (Node 652)", "/api/node/652/forecast")
    test_endpoint("9. Analytical Report Export (ZIP Stream)", "/api/export/report", is_binary=True)
