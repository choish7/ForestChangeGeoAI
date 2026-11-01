# [통합 사본] 원본: scripts/build_forest_mask.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""변화 분류 지도용 산림 마스크: 임상도 2019 또는 2025 에서 임지(입목지·무립목지)인 화소 = 1, 그 밖 = 0.

산림 2019 만 쓰면 무립목지→입목지(산림증가)가, 산림 2025 만 쓰면 산림→비산림(산지전용)이 지워지므로
두 시기 중 한 번이라도 임지(STORUNST 1 입목지, 2 무립목지)였던 곳을 남긴다. 두 시기 모두 비산림(STORUNST 0:
경작지·주거지·수체·초지 등)이거나 임상도 도화 범위 밖이면 0.

출력: output/Chuncheon_forestmask_S2_20m.tif, output/Chuncheon_forestmask_AE_10m.tif (uint8, 1 산림 / 0 비산림)
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_training_label_rasters import AE_GRID, S2_REF, grid_from  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
MAPS = {2019: DATA / "임상도 2019 51110" / "51110.shp", 2025: DATA / "임상도 2025 51110" / "51110.shp"}


def main():
    forest = []
    for y, p in MAPS.items():
        g = gpd.read_file(p, encoding="cp949", columns=["STORUNST"])
        g["STORUNST"] = g["STORUNST"].astype(str).str.replace(r"\.0$", "", regex=True)
        f = g[g["STORUNST"].isin(["1", "2"])]
        print(f"[{y}] 임지 폴리곤 {len(f):,} / {len(g):,}, 면적 {f.area.sum() / 1e4:,.0f} ha (전체 {g.area.sum() / 1e4:,.0f} ha)")
        forest.append(f)
    grids = {"S2_20m": grid_from(S2_REF),
             "AE_10m": (rasterio.crs.CRS.from_string(AE_GRID["crs"]), AE_GRID["transform"], AE_GRID["width"], AE_GRID["height"])}
    for name, (crs, tf, w, h) in grids.items():
        m = np.zeros((h, w), np.uint8)
        for f in forest:
            m |= rasterize(((geom, 1) for geom in f.to_crs(crs).geometry), out_shape=(h, w), transform=tf,
                           fill=0, dtype="uint8")
        out = ROOT / "output" / f"Chuncheon_forestmask_{name}.tif"
        with rasterio.open(out, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=crs, transform=tf,
                           nodata=None, compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
            dst.write(m, 1)
            dst.set_band_description(1, "FOREST_2019_OR_2025")
        px_ha = abs(tf.a * tf.e) / 1e4
        print(f"[{name}] 산림 화소 {int(m.sum()):,} = {m.sum() * px_ha:,.0f} ha → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
