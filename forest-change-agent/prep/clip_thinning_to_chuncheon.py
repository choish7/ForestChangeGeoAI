# [통합 사본] 원본: scripts/clip_thinning_to_chuncheon.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""산림청 숲가꾸기공간정보(2017, 전국)에서 춘천시 경계(LSMD_ADM_SECT_UMD 읍면동) 안의 폴리곤만 추출한다.

선택 기준: 폴리곤 면적의 50% 이상이 춘천시 경계 안에 있으면 포함(형상은 자르지 않고 원본 유지).
경계에 걸친 인접 시군 폴리곤의 잔편이 섞이지 않도록 단순 intersects 대신 면적 비율을 쓴다.

출력: output/춘천시_산림경영활동_숲가꾸기공간정보.gpkg (레이어: 숲가꾸기공간정보, 춘천시경계)
      output/춘천시_산림경영활동_숲가꾸기공간정보.shp (cp949)
"""
from pathlib import Path
import glob

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
OUT = ROOT / "output" / "춘천시_산림경영활동_숲가꾸기공간정보.gpkg"
CHUNCHEON_SGG = "51110"
MIN_INSIDE_RATIO = 0.5


def main():
    umd = gpd.read_file(glob.glob(str(DATA / "LSMD_ADM_SECT_UMD_강원특별자치도" / "*.shp"))[0], encoding="cp949")
    thinning = gpd.read_file(
        DATA / "산림청_산림경영활동_숲가꾸기공간정보_20171231" / "산림청_산림경영활동_숲가꾸기공간정보.shp",
        encoding="cp949")

    umd = umd[umd["COL_ADM_SE"] == CHUNCHEON_SGG].to_crs(thinning.crs)  # EPSG:5186 -> 5179
    city = umd.union_all()

    thinning["geometry"] = thinning.geometry.make_valid()
    cand = thinning[thinning.intersects(city)].copy()
    cand["춘천포함비율"] = (cand.intersection(city).area / cand.area).clip(upper=1).round(4)
    result = cand[cand["춘천포함비율"] >= MIN_INSIDE_RATIO].copy()
    result["속성시군구일치"] = (result["시도"].isin(["강원도", "강원특별자치도"]) & (result["시군구"] == "춘천시")).map({True: "Y", False: "N"})

    # 위치 기준 읍면동(최신 경계) 부여: 폴리곤 내부점이 속한 읍면동
    pts = result.copy()
    pts["geometry"] = result.representative_point()
    joined = gpd.sjoin(pts, umd[["EMD_CD", "EMD_NM", "geometry"]], how="left", predicate="within")
    result["경계_EMD_CD"] = joined["EMD_CD"].groupby(level=0).first()
    result["경계_EMD_NM"] = joined["EMD_NM"].groupby(level=0).first()

    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    result.to_file(OUT, layer="숲가꾸기공간정보", driver="GPKG")
    boundary = gpd.GeoDataFrame({"SGG_CD": [CHUNCHEON_SGG], "SGG_NM": ["춘천시"]}, geometry=[city], crs=umd.crs)
    boundary.to_file(OUT, layer="춘천시경계", driver="GPKG")
    umd_out = umd[["EMD_CD", "EMD_NM", "geometry"]]
    umd_out.to_file(OUT, layer="춘천시_읍면동경계", driver="GPKG")
    boundary.to_file(OUT.with_name("춘천시경계.shp"), encoding="cp949")
    umd_out.to_file(OUT.with_name("춘천시_읍면동경계.shp"), encoding="cp949")
    shp = result.rename(columns={"춘천포함비율": "포함비율", "속성시군구일치": "속성일치", "경계_EMD_CD": "EMD_CD", "경계_EMD_NM": "EMD_NM"})
    shp.to_file(OUT.with_suffix(".shp"), encoding="cp949")

    excluded = cand[cand["춘천포함비율"] < MIN_INSIDE_RATIO]
    print(f"전국 {len(thinning)}개 → 춘천 경계와 겹침 {len(cand)}개 → 선택 {len(result)}개 (제외 {len(excluded)}개: "
          f"{excluded['시군구'].value_counts().to_dict()})")
    print("속성 시군구가 춘천시가 아닌 선택 폴리곤:")
    print(result[result["속성시군구일치"] == "N"][["UID", "시도", "시군구", "읍면동", "사업명", "경계_EMD_NM"]].to_string())
    print(result.groupby("사업년도").size().to_string())
    print(result["작업종"].value_counts().to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
