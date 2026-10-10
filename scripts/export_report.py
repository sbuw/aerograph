"""
CLI Script to generate and export Planetary Climate Teleconnections analytical report.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.export import generate_climate_report

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_cli")


def main() -> None:
    t0 = time.time()
    logger.info("Initializing climate report export CLI runner...")
    
    zip_path_str = generate_climate_report(output_dir="exports")
    zip_path = Path(zip_path_str)
    
    elapsed = time.time() - t0
    file_size_kb = zip_path.stat().st_size / 1024.0
    
    logger.info("Report generation completed in %.2f seconds.", elapsed)
    logger.info("Archive Location: %s", zip_path.resolve())
    logger.info("Archive Size: %.2f KB (%.2f MB)", file_size_kb, file_size_kb / 1024.0)


if __name__ == "__main__":
    main()
