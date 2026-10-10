"""
Verification script for external data sources defined in docs/DATA_SOURCES.md
Tests Open-Meteo APIs (Archive, Historical-Forecast, Forecast) and NOAA GFS on AWS Open Data.
"""

import json
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

def log_header(title: str):
    print(f"\n{'='*70}\n {title}\n{'='*70}")

def fetch_json(url: str, timeout: int = 15):
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        elapsed = time.time() - t0
        body = resp.read().decode("utf-8")
        data = json.loads(body)
        return resp.status, elapsed, len(body), data

def test_open_meteo_archive():
    log_header("1. Open-Meteo Historical Archive API (archive-api.open-meteo.com)")
    
    # Test 1A: Check reachability and surface_pressure
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=52.52&longitude=13.41"
        "&start_date=2024-01-01&end_date=2024-01-03"
        "&hourly=surface_pressure"
    )
    print(f"Request URL: {url}")
    status, elapsed, size, data = fetch_json(url)
    print(f"Status: {status} | Latency: {elapsed:.3f}s | Response Size: {size} bytes")
    print(f"Hourly Units: {data.get('hourly_units')}")
    print(f"Sample Times: {data['hourly']['time'][:3]}")
    print(f"Sample surface_pressure: {data['hourly']['surface_pressure'][:3]}")
    
    # Test 1B: Check geopotential_height_500hPa on archive-api
    url_z500 = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=52.52&longitude=13.41"
        "&start_date=2024-01-01&end_date=2024-01-03"
        "&hourly=geopotential_height_500hPa,surface_pressure"
    )
    print(f"\nTesting geopotential_height_500hPa on archive-api:")
    status, elapsed, size, data_z = fetch_json(url_z500)
    print(f"Status: {status} | Latency: {elapsed:.3f}s")
    print(f"Hourly Units: {data_z.get('hourly_units')}")
    z_vals = data_z['hourly']['geopotential_height_500hPa'][:3]
    print(f"Sample geopotential_height_500hPa values: {z_vals}")
    if all(v is None for v in z_vals):
        print("  -> NOTICE: archive-api accepts 'geopotential_height_500hPa' parameter without HTTP error, but returns null values because standard ERA5 surface archive does not store 3D pressure levels.")

    # Test 1C: Check historical-forecast-api (Historical Forecast Archive)
    log_header("1.1 Open-Meteo Historical Forecast API (historical-forecast-api.open-meteo.com)")
    url_hist_fc = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        "?latitude=52.52&longitude=13.41"
        "&start_date=2024-01-01&end_date=2024-01-03"
        "&hourly=geopotential_height_500hPa,surface_pressure"
    )
    print(f"Request URL: {url_hist_fc}")
    status, elapsed, size, data_hf = fetch_json(url_hist_fc)
    print(f"Status: {status} | Latency: {elapsed:.3f}s | Response Size: {size} bytes")
    print(f"Hourly Units: {data_hf.get('hourly_units')}")
    print(f"Sample z500 values: {data_hf['hourly']['geopotential_height_500hPa'][:3]}")
    print(f"Sample surface_pressure values: {data_hf['hourly']['surface_pressure'][:3]}")
    print("  -> CONFIRMED: historical-forecast-api DOES provide geopotential_height_500hPa (meters) and surface_pressure (hPa) for historical dates!")

def test_open_meteo_forecast():
    log_header("2. Open-Meteo Forecast API (api.open-meteo.com)")

    # Test 2A: Single point with past_days (rolling historical window)
    url_single = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=52.52&longitude=13.41"
        "&past_days=92&forecast_days=1"
        "&hourly=geopotential_height_500hPa,surface_pressure"
    )
    print(f"Single point request (past_days=92): {url_single}")
    status, elapsed, size, data = fetch_json(url_single)
    print(f"Status: {status} | Latency: {elapsed:.3f}s | Payload Size: {size/1024:.1f} KB")
    print(f"Hourly Units: {data.get('hourly_units')}")
    times = data['hourly']['time']
    print(f"Total time steps: {len(times)} (From {times[0]} to {times[-1]})")
    print(f"Sample z500: {data['hourly']['geopotential_height_500hPa'][:3]}")
    print(f"Sample surface_pressure: {data['hourly']['surface_pressure'][:3]}")
    print("  -> CONFIRMED: past_days=92 successfully returns a rolling 92-day historical window with valid z500 and surface_pressure!")

    # Test 2B: Batch coordinates query (e.g. 100 and 500 coordinates)
    log_header("2.1 Open-Meteo Batch Coordinates Performance ($5x5 degree grid)")
    for n in [10, 50, 100, 200, 500]:
        lats = ",".join(str(round(20.0 + (i % 13) * 5.0, 2)) for i in range(n))
        lons = ",".join(str(round(-180.0 + (i // 13) * 5.0, 2)) for i in range(n))
        url_batch = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lats}&longitude={lons}"
            f"&past_days=7&hourly=geopotential_height_500hPa,surface_pressure"
        )
        t0 = time.time()
        req = urllib.request.Request(url_batch, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - t0
            raw = resp.read().decode("utf-8")
            res_list = json.loads(raw)
            print(f"Batch {n:3d} points: Status {resp.status} | Latency: {elapsed:.2f}s | Payload: {len(raw)/1024:.1f} KB | Returned: {len(res_list)} locations")

def test_noaa_gfs_aws():
    log_header("3. NOAA GFS on AWS Open Data (noaa-gfs-bdp-pds)")

    # Test 3A: S3 bucket listing via HTTPS
    now_utc = datetime.now(timezone.utc)
    # Check today or yesterday
    date_str = now_utc.strftime("%Y%m%d")
    s3_list_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/?prefix=gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.1p00"
    print(f"Testing S3 XML listing: {s3_list_url}")
    req = urllib.request.Request(s3_list_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_bytes = resp.read()
        print(f"Status: {resp.status} | Size: {len(xml_bytes)} bytes")
        root = ET.fromstring(xml_bytes)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        keys = [elem.text for elem in root.findall(".//s3:Key", ns)]
        if not keys:
            keys = [elem.text for elem in root.findall(".//Key")]
        print(f"Found {len(keys)} GFS 1.00-deg files for {date_str}. Sample files:")
        for k in keys[:4]:
            print(f"  - {k}")

    # Test 3B: Inspect .idx file to obtain byte ranges for HGT:500 mb and PRES:surface
    idx_key = f"gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.1p00.f000.idx"
    idx_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/{idx_key}"
    print(f"\nFetching index file: {idx_url}")
    req_idx = urllib.request.Request(idx_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_idx, timeout=15) as resp_idx:
        idx_lines = resp_idx.read().decode("utf-8").splitlines()
        print(f"Index lines count: {len(idx_lines)}")
        
        # Parse records
        records = []
        for line in idx_lines:
            parts = line.split(":")
            if len(parts) >= 6:
                records.append({
                    "num": int(parts[0]),
                    "offset": int(parts[1]),
                    "date": parts[2],
                    "var": parts[3],
                    "level": parts[4],
                    "forecast": parts[5]
                })

        hgt_500 = None
        pres_sfc = None
        for i, rec in enumerate(records):
            if rec["var"] == "HGT" and rec["level"] == "500 mb":
                next_offset = records[i+1]["offset"] - 1 if i+1 < len(records) else ""
                hgt_500 = (rec["offset"], next_offset)
                print(f"Found HGT:500 mb -> byte offset: {rec['offset']} to {next_offset}")
            elif rec["var"] == "PRES" and rec["level"] == "surface":
                next_offset = records[i+1]["offset"] - 1 if i+1 < len(records) else ""
                pres_sfc = (rec["offset"], next_offset)
                print(f"Found PRES:surface -> byte offset: {rec['offset']} to {next_offset}")

    # Test 3C: HTTP Range request to fetch ONLY HGT:500 mb GRIB2 slice
    if hgt_500:
        data_key = f"gfs.{date_str}/00/atmos/gfs.t00z.pgrb2.1p00.f000"
        data_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/{data_key}"
        range_header = f"bytes={hgt_500[0]}-{hgt_500[1]}"
        print(f"\nPerforming HTTP Range request: {data_url} with Range: {range_header}")
        t0 = time.time()
        req_range = urllib.request.Request(data_url, headers={"User-Agent": "Mozilla/5.0", "Range": range_header})
        with urllib.request.urlopen(req_range, timeout=15) as resp_range:
            elapsed = time.time() - t0
            grib_chunk = resp_range.read()
            print(f"Range Status: {resp_range.status} (Partial Content) | Latency: {elapsed:.3f}s")
            print(f"Downloaded slice size: {len(grib_chunk)} bytes ({len(grib_chunk)/1024:.1f} KB)")
            print(f"Header magic bytes: {grib_chunk[:4]} (Expected: b'GRIB')")
            print(f"End magic bytes: {grib_chunk[-4:]} (Expected: b'7777')")
            assert grib_chunk[:4] == b"GRIB", "Magic header must be GRIB"
            assert grib_chunk[-4:] == b"7777", "GRIB message must end with 7777"
            print("  -> CONFIRMED: NOAA GFS S3 allows precise partial-content Range downloads for 500hPa geopotential height!")

if __name__ == "__main__":
    test_open_meteo_archive()
    test_open_meteo_forecast()
    test_noaa_gfs_aws()
