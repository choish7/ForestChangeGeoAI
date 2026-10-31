"""[4.2-②] 변화 라벨 폴리곤에서 층화 표본점 추출 + 공간블록 분할.

- 경계 오차 회피: 폴리곤 내부 버퍼(-15 m) 안에서만 추출
- 층화: 클래스별 목표 개수, 폴리곤 면적 비례 확률로 추출
- 공간 일반화 검증: 5 km 블록 단위로 20% 를 test_spatial 로 분리

산출: work/samples.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from fcd import config as C
from fcd.data import change_polygons
from fcd.labels import TRAIN_CLASSES

PRE_STATE = [f"{c}_{C.YEAR_FROM}" for c in ("FRTP", "KOFTR", "AGCLS", "DNST")]


def sample_class(polys: gpd.GeoDataFrame, n: int, rng) -> gpd.GeoDataFrame:
    inner = polys.copy()
    inner["geometry"] = inner.buffer(-C.INNER_BUFFER_M)
    inner = inner[~inner.is_empty & inner.is_valid & (inner.area > 100)]
    if inner.empty:
        return inner
    w = inner.area.to_numpy()
    pick = rng.choice(len(inner), size=min(n * 2, max(n, len(inner) * 5)), p=w / w.sum())
    chosen = inner.iloc[pick]
    pts = chosen.sample_points(1, rng=rng)
    out = chosen.drop(columns="geometry").set_geometry(pts.values, crs=inner.crs)
    # 같은 10 m 셀 중복 제거 (AlphaEarth 해상도)
    key = (out.geometry.x // 10).astype(int).astype(str) + "_" + (out.geometry.y // 10).astype(int).astype(str)
    out = out[~key.duplicated()].head(n)
    return out


def main():
    rng = np.random.default_rng(C.SEED)
    cp = change_polygons()
    parts = []
    for cls in TRAIN_CLASSES:
        s = sample_class(cp[cp["change_type"] == cls], C.SAMPLES_PER_CLASS[cls], rng)
        print(f"[02] {cls:<11} 폴리곤 {int((cp.change_type == cls).sum()):>6,} → 표본 {len(s):,}")
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    s = gpd.GeoDataFrame(s, geometry="geometry", crs=C.CRS_WORK)

    bx, by = (s.geometry.x // C.BLOCK_M).astype(int), (s.geometry.y // C.BLOCK_M).astype(int)
    s["block_id"] = "B" + bx.astype(str) + "_" + by.astype(str)
    blocks = s["block_id"].unique()
    test_blocks = set(rng.choice(blocks, size=int(len(blocks) * C.TEST_BLOCK_FRAC), replace=False))
    s["split"] = np.where(s["block_id"].isin(test_blocks), "test_spatial", "train")

    ll = s.to_crs(C.CRS_GEE).geometry
    s["lon"], s["lat"] = ll.x.round(7), ll.y.round(7)
    s["sample_id"] = [f"S{i:05d}" for i in range(len(s))]
    keep = ["sample_id", "patch_id", "change_type", "changed", "change_year", "label_conf",
            *PRE_STATE, "block_id", "split", "lon", "lat", "geometry"]
    s = s[keep]
    s.to_parquet(C.SAMPLES)

    print(pd.crosstab(s["change_type"], s["split"], margins=True).to_string())
    print(f"[02] 공간블록 {len(blocks)}개 (test {len(test_blocks)}개) ✓ {C.SAMPLES}")


if __name__ == "__main__":
    main()
