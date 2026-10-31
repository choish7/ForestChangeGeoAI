"""[4.2-①] 임상도 2019×2025 중첩 → 변화 라벨 폴리곤 + 에이전트 분석단위 생성.

산출: work/change_polygons.parquet, work/regions.parquet
실행: python scripts/01_build_change_labels.py      (중첩 연산 약 4분)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from fcd import config as C
from fcd.data import build_regions, load_forest_map
from fcd.labels import CHANGE_TYPES, classify


def main():
    y0, y1 = C.YEAR_FROM, C.YEAR_TO
    t = time.time()
    a = load_forest_map(y0).drop(columns=["UPDATE_YEAR"])
    b = load_forest_map(y1)
    a.columns = [f"{c}_{y0}" if c != "geometry" else c for c in a.columns]
    b.columns = [f"{c}_{y1}" if c not in ("geometry", "UPDATE_YEAR") else c for c in b.columns]
    print(f"[01] 임상도 {y0}: {len(a):,} / {y1}: {len(b):,} 폴리곤 — 중첩 중…")

    ov = gpd.overlay(a, b, how="intersection", keep_geom_type=True)
    ov = ov.explode(index_parts=False, ignore_index=True)
    ov["area_ha"] = ov.area / 1e4
    ov = ov[ov["area_ha"] >= C.MIN_PIECE_HA].reset_index(drop=True)
    print(f"[01] 중첩 완료 {time.time() - t:.0f}s, 조각 {len(ov):,}개 (≥{C.MIN_PIECE_HA} ha)")

    ov = classify(ov, y0, y1)
    ov["patch_id"] = [f"P{i:06d}" for i in range(len(ov))]
    ov["year_from"], ov["year_to"] = y0, y1
    ov["source"] = f"임상도 {y0}·{y1} 속성비교 (산림청, 1:5,000)"
    ov.to_parquet(C.CHANGE_POLYGONS)

    summary = (ov.groupby("change_type")["area_ha"].agg(["count", "sum"])
                 .rename(columns={"count": "patches", "sum": "area_ha"}).round(1))
    summary["설명"] = summary.index.map(CHANGE_TYPES)
    print(summary.to_string())
    print(f"[01] ✓ {C.CHANGE_POLYGONS}")

    r = build_regions()
    r.to_parquet(C.REGIONS)
    print(f"[01] ✓ 분석단위 {len(r)}개 → {C.REGIONS}")
    print(r.groupby("level").size().to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    main()
