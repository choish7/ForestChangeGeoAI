"""R2 (editorial query Q1): provenance of the AlphaEarth annual images used (asset ID, time stamps, versions).

The observation year is the year of system:time_start; system:time_end is the exclusive end (1 January of the next year).
Feature extraction selects images with ee.Filter.calendarRange(year, year, 'year'), which acts on system:time_start.
  python forest-change-agent/scripts/16_alphaearth_assets.py [--project <GEE project>]
Output: outputs/results/R2_alphaearth_assets.json
"""
import argparse
import json
from pathlib import Path

import ee

OUT = Path(__file__).resolve().parents[1] / "outputs" / "results" / "R2_alphaearth_assets.json"
REGIONS = {"Chuncheon": [127.45, 37.68, 128.05, 38.08], "Inje": [127.95, 37.85, 128.55, 38.30]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ee-choish7838")
    ee.Initialize(project=ap.parse_args().project)
    rows = []
    for region, bbox in REGIONS.items():
        col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
               .filterBounds(ee.Geometry.Rectangle(bbox)).filterDate("2017-01-01", "2026-01-01"))
        info = col.map(lambda i: ee.Feature(None, {
            "id": i.get("system:id"), "ts": ee.Date(i.get("system:time_start")).format("YYYY-MM-dd"),
            "te": ee.Date(i.get("system:time_end")).format("YYYY-MM-dd"), "mv": i.get("MODEL_VERSION"),
            "dv": i.get("DATASET_VERSION"), "psv": i.get("PROCESSING_SOFTWARE_VERSION"), "utm": i.get("UTM_ZONE")})).getInfo()
        rows += [{**f["properties"], "region": region} for f in info["features"]]
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in sorted(rows, key=lambda r: (r["region"], r["ts"])):
        print(r["region"], r["ts"][:4], r["id"], r["ts"], r["te"], "model", r["mv"], "dataset", r["dv"])


if __name__ == "__main__":
    main()
