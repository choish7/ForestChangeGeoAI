"""경영실적 ↔ 임상도 비교 합의 결과로 Mini-GFM(Sentinel-2 20 m)·AlphaEarth(10 m) 학습용 라벨 래스터를 만든다.

입력
  output/춘천시_산림변화라벨_2013_2019_2025.gpkg      임상도 변화라벨(조각)
  output/춘천_경영실적_임상도_변화라벨_비교.gpkg      단위별 합의 라벨 (P1 본번 / P2 소반)
  격자: Sentinel-2 = output/Chuncheon_S2_2019_2025_42Bands.tif (2017~2025 54밴드도 같은 영역·20 m·EPSG:32652)
        AlphaEarth = 실제 Chuncheon_AE_*_64Bands_int8.tif 격자(AE_GRID, 4647×4549, 원점 368350/4216210)
                     --ae-ref 로 AlphaEarth GeoTIFF 를 주면 그 파일의 격자 사용

학습 라벨 규칙 (구간별, 임상도 조각을 10 m 안쪽으로 줄여 래스터화)
  임상도 화소 라벨: 0 무변화, 1 벌채·갱신, 2 산지전용, 3 산림증가 (type_shift·ambiguous·nonforest·도화범위차이 제외)
  국유림 안: 변화(1~3) 화소는 합의 1(확정변화)·6(수확외 실적·임상도변화) 단위에서만,
            무변화(0) 화소는 합의 0·4·5(실적없음/솎아베기/피해·기타 + 변화없음) 단위에서만 유지
            → 합의 2(임상도만 변화)·3(수확실적만) 단위는 학습에서 제외
  국유림 밖: 실적으로 검증할 수 없으므로 임상도 라벨 그대로 (SOURCE 밴드로 구분)
  변화 연도: 확정변화 단위는 실적의 최초 수확·갱신 연도, 그 밖은 임상도 갱신년도

밴드 (uint8, 255 = 제외/없음)
  1 TRAIN_P1   2013→2019 학습 라벨 (0~3)          2 TRAIN_P2   2019→2025 학습 라벨 (0~3)
  3 YEAR_P1    P1 변화 연도 − 2000               4 YEAR_P2    P2 변화 연도 − 2000
  5 CONS_P1    본번 단위 합의 라벨 (0~6)          6 CONS_P2    소반 단위 합의 라벨 (0~6)
  7 SOURCE     1 국유림(실적 검증), 2 국유림 밖(임상도 단독)

출력: output/Chuncheon_labels_S2_20m.tif, output/Chuncheon_labels_AE_10m.tif
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "output" / "춘천시_산림변화라벨_2013_2019_2025.gpkg"
COMPARE = ROOT / "output" / "춘천_경영실적_임상도_변화라벨_비교.gpkg"
S2_REF = ROOT / "output" / "Chuncheon_S2_2019_2025_42Bands.tif"
FC_CODE = {"no_change": 0, "harvest": 1, "conversion": 2, "gain": 3}
PERIODS = ("P2013_2019", "P2019_2025")
VERIFIED_CHANGE, VERIFIED_STABLE = (1, 6), (0, 4, 5)
INNER_BUFFER_M = 10
BANDS = ["TRAIN_P1", "TRAIN_P2", "YEAR_P1", "YEAR_P2", "CONS_P1", "CONS_P2", "SOURCE"]
# GEE 가 내보낸 Chuncheon_AE_{year}_64Bands_int8.tif 의 실제 격자 (Colab 에서 확인).
# Sentinel 20 m 격자와 원점이 1화소(10 m) 어긋나므로 S2 격자에서 유도하지 않고 그대로 쓴다.
AE_GRID = {"crs": "EPSG:32652", "transform": Affine(10, 0, 368350, 0, -10, 4216210), "width": 4647, "height": 4549}


def grid_from(path, res=None):
    with rasterio.open(path) as ds:
        if res is None:
            return ds.crs, ds.transform, ds.width, ds.height
        # 같은 영역을 res 해상도로 (GEE 내보내기는 같은 원점에서 scale 배수로 격자를 잡음)
        f = ds.res[0] / res
        t = ds.transform
        return ds.crs, Affine(res, 0, t.c, 0, -res, t.f), int(round(ds.width * f)), int(round(ds.height * f))


def burn(shapes, shape, transform, fill=255):
    shapes = [(g, int(v)) for g, v in shapes if g is not None and not g.is_empty]
    if not shapes:
        return np.full(shape, fill, np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=fill, dtype="uint8")


def build(crs, transform, width, height, pieces, units, stands_all):
    shape = (height, width)
    pieces = pieces.to_crs(crs)
    inner = pieces.copy()
    inner["geometry"] = pieces.buffer(-INNER_BUFFER_M)
    inner = inner[~inner.is_empty]
    nat = burn(zip(stands_all.to_crs(crs).geometry, [1] * len(stands_all)), shape, transform, fill=0)
    covered = burn(zip(pieces.geometry, [1] * len(pieces)), shape, transform, fill=0)
    out = {"SOURCE": np.where(nat == 1, 1, np.where(covered == 1, 2, 255)).astype(np.uint8)}
    stats = {}
    for i, p in enumerate(PERIODS, 1):
        typ = inner[f"{p}_type"]
        sel = inner[typ.isin(FC_CODE)].assign(code=typ.map(FC_CODE))
        sel = sel.sort_values("code")  # 무변화를 먼저 굽고 변화 조각이 위에 오도록
        fc = burn(zip(sel.geometry, sel["code"]), shape, transform)
        yr = pd.to_numeric(sel[f"{p}_year"], errors="coerce")
        chg = sel[(sel["code"] > 0) & yr.notna()]
        fc_year = burn(zip(chg.geometry, (yr[chg.index] - 2000).astype(int)), shape, transform)

        u = units[units["구간"] == p].to_crs(crs)
        cons = burn(zip(u.geometry, u["CONS"]), shape, transform)
        ry = u[(u["CONS"] == 1) & u["수확연도_최초"].notna()]
        rec_year = burn(zip(ry.geometry, (ry["수확연도_최초"] - 2000).astype(int)), shape, transform)

        train = fc.copy()
        in_nat = nat == 1
        train[in_nat & np.isin(fc, [1, 2, 3]) & ~np.isin(cons, VERIFIED_CHANGE)] = 255
        train[in_nat & (fc == 0) & ~np.isin(cons, VERIFIED_STABLE)] = 255
        year = np.where(in_nat & (cons == 1) & (rec_year != 255), rec_year, fc_year).astype(np.uint8)
        year[~np.isin(train, [1, 2, 3])] = 255
        out[f"TRAIN_P{i}"], out[f"YEAR_P{i}"], out[f"CONS_P{i}"] = train, year, cons

        stats[p] = {
            "임상도_화소": {k: int((fc == v).sum()) for k, v in FC_CODE.items()},
            "학습_국유림": {k: int(((train == v) & in_nat).sum()) for k, v in FC_CODE.items()},
            "학습_국유림밖": {k: int(((train == v) & (out["SOURCE"] == 2)).sum()) for k, v in FC_CODE.items()},
            "국유림에서_제외된_변화화소": int((in_nat & np.isin(fc, [1, 2, 3]) & (train == 255)).sum()),
            "국유림에서_제외된_무변화화소": int((in_nat & (fc == 0) & (train == 255)).sum()),
        }
    return out, stats


def write(path, crs, transform, out):
    profile = dict(driver="GTiff", width=out["SOURCE"].shape[1], height=out["SOURCE"].shape[0], count=len(BANDS),
                   dtype="uint8", crs=crs, transform=transform, nodata=255, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        for i, b in enumerate(BANDS, 1):
            dst.write(out[b], i)
            dst.set_band_description(i, b)
        dst.update_tags(TRAIN="0 no_change,1 harvest,2 conversion,3 gain", YEAR="year-2000",
                        CONS="0 확정무변화,1 확정변화,2 임상도만변화,3 수확실적만,4 솎아베기·무변화,5 피해기타·무변화,6 수확외실적·임상도변화",
                        SOURCE="1 국유림(실적 검증),2 국유림 밖(임상도 단독)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae-ref", default=None, help="AlphaEarth GeoTIFF (있으면 그 격자로, 없으면 AE_GRID)")
    args = ap.parse_args()

    pieces = gpd.read_file(LABELS, layer="변화라벨",
                           columns=["P2013_2019_type", "P2013_2019_year", "P2019_2025_type", "P2019_2025_year"])
    units = gpd.read_file(COMPARE, layer="단위별_비교", columns=["구간", "단위ID", "CONS", "수확연도_최초"])
    stands_all = units[units["구간"] == "P2019_2025"]  # 9차 소반 전체 = 국유림 범위

    ae = (grid_from(args.ae_ref) if args.ae_ref else
          (rasterio.crs.CRS.from_string(AE_GRID["crs"]), AE_GRID["transform"], AE_GRID["width"], AE_GRID["height"]))
    grids = {"S2_20m": grid_from(S2_REF), "AE_10m": ae}
    for name, (crs, transform, w, h) in grids.items():
        out, stats = build(crs, transform, w, h, pieces, units, stands_all)
        path = ROOT / "output" / f"Chuncheon_labels_{name}.tif"
        write(path, crs, transform, out)
        print(f"[{name}] {w}×{h} px, {crs} → {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        for p, s in stats.items():
            print(f"  {p}: {s}")


if __name__ == "__main__":
    main()
