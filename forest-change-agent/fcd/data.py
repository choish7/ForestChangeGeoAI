"""보유 원자료 로더 (인코딩·좌표계 정리)."""
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import pyogrio

from . import config as C


def load_forest_map(year: int) -> gpd.GeoDataFrame:
    """임상도 → 공통 컬럼(FRTP/KOFTR/AGCLS/DNST[/UPDATE_YEAR]) + 유효 지오메트리."""
    cols = ["FRTP_CD", "KOFTR_GROU", "AGCLS_CD", "DNST_CD", "갱신년도"]
    g = gpd.read_file(C.FOREST_MAP[year], columns=cols, engine="pyogrio")
    g = g.rename(columns={"FRTP_CD": "FRTP", "KOFTR_GROU": "KOFTR", "AGCLS_CD": "AGCLS",
                          "DNST_CD": "DNST", "갱신년도": "UPDATE_YEAR"})
    g["geometry"] = g.geometry.make_valid()
    return g.to_crs(C.CRS_WORK)


def _fix_utf8(s):
    # 소반도 DBF 일부 필드가 잘린 UTF-8 이라 latin1 로 읽은 뒤 복원
    return s.encode("latin1").decode("utf-8", "replace") if isinstance(s, str) else s


@lru_cache
def load_mgmt_plan() -> gpd.GeoDataFrame:
    """국유림 경영계획부 소반도 (임반 cmblc_id / 소반 sblt_id, 면적 ha, 축적 m³)."""
    path = next(C.MGMT_PLAN_DIR.glob(C.MGMT_PLAN_GLOB))
    g = pyogrio.read_dataframe(path, encoding="latin1")
    for c in g.columns:
        if g[c].dtype == object and c != "geometry":
            g[c] = g[c].map(_fix_utf8)
    g["emd"] = g["lctn_daddr"].str.split().str[2]        # "강원특별자치도 춘천시 북산면 …"
    g = g.rename(columns={"tot_area": "area_ha", "realt_accm": "stock_m3",
                          "ha_party_r": "stock_m3_per_ha"})
    for c in ("area_ha", "stock_m3", "stock_m3_per_ha"):
        g[c] = pd.to_numeric(g[c], errors="coerce")        # DBF 문자열 필드 → 수치
    return g.set_crs(g.crs or C.CRS_WORK, allow_override=True).to_crs(C.CRS_WORK)


def build_regions() -> gpd.GeoDataFrame:
    """에이전트가 이름으로 부르는 분석단위.
    - 춘천시 전체: 2025 임상도 전체 범위
    - 춘천 국유림 전체: 경영계획부 소반 전체
    - 읍면(국유림): 경영계획부 소반을 읍면 주소로 dissolve (국유림 구역에 한정됨)
    - 임반: 경영계획부 임반 번호
    ※ 읍면동 행정경계 파일을 data/ 에 넣으면 여기서 교체하면 됨.
    """
    fm = load_forest_map(2025)
    city = gpd.GeoDataFrame({"region_id": ["51110"], "name": ["춘천시 전체"], "level": ["city"]},
                            geometry=[fm.union_all().convex_hull], crs=C.CRS_WORK)
    mp = load_mgmt_plan()
    nf = gpd.GeoDataFrame({"region_id": ["nf:51110"], "name": ["춘천 국유림 전체"], "level": ["national_forest"]},
                          geometry=[mp.union_all()], crs=C.CRS_WORK)
    mp = mp[mp["emd"].str.endswith(("읍", "면", "동"), na=False)]
    emd = mp.dissolve("emd").reset_index()
    emd = gpd.GeoDataFrame({"region_id": "emd:" + emd["emd"], "name": emd["emd"] + " 국유림",
                            "level": "emd_national_forest"}, geometry=emd.geometry, crs=C.CRS_WORK)
    cmp_ = mp.dissolve("cmblc_id").reset_index()
    cmp_ = gpd.GeoDataFrame({"region_id": "cmp:" + cmp_["cmblc_id"],
                             "name": "춘천 국유림 " + cmp_["cmblc_id"].str[:3].str.lstrip("0") + "임반",
                             "level": "compartment"}, geometry=cmp_.geometry, crs=C.CRS_WORK)
    return pd.concat([city, nf, emd, cmp_], ignore_index=True)


@lru_cache
def regions() -> gpd.GeoDataFrame:
    if C.REGIONS.exists():
        return gpd.read_parquet(C.REGIONS)
    r = build_regions()
    r.to_parquet(C.REGIONS)
    return r


@lru_cache
def change_polygons() -> gpd.GeoDataFrame:
    if not C.CHANGE_POLYGONS.exists():
        raise FileNotFoundError("scripts/01_build_change_labels.py 를 먼저 실행하세요.")
    return gpd.read_parquet(C.CHANGE_POLYGONS)
