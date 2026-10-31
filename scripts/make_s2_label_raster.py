"""42밴드 Sentinel-2 스택(Chuncheon_S2_2019_2025_42Bands.tif) 격자에 맞춘 평가용 라벨 래스터를 만든다.

밴드 (uint8, 255 = 평가 제외)
  1 FC_TYPE   임상도 2019→2025 변화 (forest-change-agent work/change_polygons.parquet, fcd.labels 규칙)
              0 no_change, 1 harvest(벌채·갱신), 2 conversion(산지전용), 3 gain(산림 증가)
              type_shift·ambiguous·nonforest 는 255. 경계 오차를 피하려 폴리곤을 10 m 안쪽으로 줄여 래스터화
  2 FC_YEAR   위 변화의 발생연도(임상도 갱신년도 약한 라벨) − 2000, 없으면 255
  3 MGMT      춘천_임소반_산림사업실적.gpkg 의 국유림 실적(7·8·9차 중 2019~2025 사업)
              0 실적 없음, 1 수확·갱신(주벌·기타 산물·조림), 2 솎아베기·숲가꾸기, 3 기타(시설·소득·피해복구)만
              국유림 밖 255
  4 MGMT_YEAR 3번 밴드 사업의 가장 이른 연도 − 2000, 없으면 255

출력: output/Chuncheon_S2_2019_2025_labels.tif (+ Colab 전송용 base64 텍스트)
"""
import base64
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
S2 = ROOT / "output" / "Chuncheon_S2_2019_2025_42Bands.tif"
CHANGE = ROOT / "forest-change-agent" / "work" / "change_polygons.parquet"
STANDS = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
YEARS = (2019, 2025)  # Sentinel-2 스택 기간
OUT =ROOT / "output" / "Chuncheon_S2_2019_2025_labels.tif"
FC_CODES = {"no_change": 0, "harvest": 1, "conversion": 2, "gain": 3}
INNER_BUFFER_M = 10


def burn(shapes, ref, fill=255):
    return rasterize(shapes, out_shape=(ref.height, ref.width), transform=ref.transform,
                     fill=fill, dtype="uint8")


def main():
    with rasterio.open(S2) as ref:
        profile = ref.profile
        cp = gpd.read_parquet(CHANGE, columns=["change_type", "change_year", "geometry"])
        cp = cp[cp["change_type"].isin(FC_CODES)].to_crs(ref.crs)
        cp["geometry"] = cp.buffer(-INNER_BUFFER_M)
        cp = cp[~cp.is_empty]
        # 면적이 작은 변화 폴리곤이 큰 무변화 폴리곤에 덮이지 않도록 무변화를 먼저 굽는다
        cp = cp.assign(_o=cp["change_type"].ne("no_change")).sort_values("_o")
        fc_type = burn(zip(cp.geometry, cp["change_type"].map(FC_CODES)), ref)
        yr = pd.to_numeric(cp["change_year"], errors="coerce")
        has = yr.notna()
        fc_year = burn(zip(cp.geometry[has], (yr[has] - 2000).astype(int)), ref)

        # 춘천_임소반_산림사업실적.gpkg 의 실적(7·8·9차) 중 영상 기간(2019~2025) 사업
        rec = gpd.read_file(STANDS, layer="산림사업실적", ignore_geometry=True)
        rec = rec[rec["연도"].between(YEARS[0], YEARS[1])]
        rec["sid"] = rec["임반"] + "/" + rec["소반"]
        harvest = rec["사업구분"].eq("조림") | (rec["사업구분"].eq("임목생산") & rec["종별"].isin(["주벌", "기타 산물"]))
        thin = rec["사업구분"].eq("숲가꾸기") | rec["사업구분"].eq("임목생산") & ~harvest
        rec["c"] = np.select([harvest, thin], [1, 2], default=3)  # 우선순위: 수확·갱신 > 솎아베기 > 기타
        cls = rec.groupby("sid").agg(c=("c", "min"), year=("연도", "min"))
        print(f"[실적] {YEARS[0]}~{YEARS[1]} {len(rec)}건, 소반 {len(cls)}개, 차기 {rec['차기'].value_counts().to_dict()}")
        st = gpd.read_file(STANDS, layer="임소반_실적요약", columns=["임반", "소반"]).to_crs(ref.crs)
        st["sid"] = st["임반"] + "/" + st["소반"]
        st = st.join(cls, on="sid")
        st["c"] = st["c"].fillna(0).astype(int)
        st["geometry"] = st.buffer(-INNER_BUFFER_M)
        st = st[~st.is_empty].sort_values("c")
        mgmt = burn(zip(st.geometry, st["c"]), ref)
        has = st["year"].notna()
        mgmt_year = burn(zip(st.geometry[has], (st["year"][has] - 2000).astype(int)), ref)

    profile.update(count=4, dtype="uint8", nodata=255, compress="deflate", predictor=1, tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(np.stack([fc_type, fc_year, mgmt, mgmt_year]))
        dst.descriptions = ("FC_TYPE", "FC_YEAR", "MGMT", "MGMT_YEAR")
    for name, arr, codes in (("FC_TYPE", fc_type, FC_CODES), ("MGMT", mgmt, {"none": 0, "harvest": 1, "thin": 2, "other": 3})):
        print(name, {k: int((arr == v).sum()) for k, v in codes.items()}, "제외", int((arr == 255).sum()))
    b64 = OUT.with_suffix(".b64.txt")
    b64.write_text(base64.b64encode(OUT.read_bytes()).decode())
    print(f"저장: {OUT} ({OUT.stat().st_size / 1e3:.0f} KB), base64 {b64.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
