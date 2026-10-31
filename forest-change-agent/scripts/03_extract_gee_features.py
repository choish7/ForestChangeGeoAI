"""[4.2-③] 표본점에서 AlphaEarth 연간 임베딩(2017~2025)·Sentinel-2·지형 특징 추출.

산출: work/features.parquet  (표본 속성 + ae{year}_A00..A63, ndvi/nbr, elev/slope)
옵션: --export-rasters  시범지역 전체 AlphaEarth 래스터를 Google Drive 로 export

실행: python scripts/03_extract_gee_features.py [--export-rasters] [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd

from fcd import config as C
from fcd import gee
from fcd.data import regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-rasters", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 표본 수 제한")
    args = ap.parse_args()

    gee.init()
    s = gpd.read_parquet(C.SAMPLES)
    if args.limit:
        s = s.sample(args.limit, random_state=C.SEED)
    print(f"[03] 표본 {len(s):,}점 × (AlphaEarth {len(C.AE_YEARS)}년×64 + S2 + DEM) 추출")

    t = time.time()
    f = gee.extract_points(s[["sample_id", "lon", "lat"]])
    df = s.drop(columns="geometry").merge(f, on="sample_id", how="inner")
    miss = df.filter(like=f"ae{C.YEAR_TO}_").isna().any(axis=1).mean()
    df.to_parquet(C.FEATURES)
    print(f"[03] ✓ {len(df):,}점, {f.shape[1] - 1}개 밴드, {time.time() - t:.0f}s, "
          f"AlphaEarth 결측 {miss:.1%} → {C.FEATURES}")

    if args.export_rasters:
        city = regions().query("level == 'city'").to_crs(C.CRS_GEE)
        tasks = gee.export_rasters(gee.to_ee_geometry(city))
        print(f"[03] Drive export task {len(tasks)}개 시작: " + ", ".join(t.id for t in tasks))
        print("     진행상황: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
