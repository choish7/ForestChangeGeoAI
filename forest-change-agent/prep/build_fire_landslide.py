# [통합 사본] 원본: scripts/build_fire_landslide.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""전국 산불정보·산사태신고정보 CSV에서 춘천시 경계 안의 건만 점 공간정보로 만든다.

산불 좌표는 레코드마다 좌표계가 섞여 있다(검증: 같은 레코드의 경위도 좌표와 비교).
  - 경위도(EPSG:4326), Web Mercator(EPSG:3857, 2016년 이전 일부)
  - TM 좌표: EPSG:5181(가상원점 N=500,000)과 EPSG:5186(N=600,000)이 같은 연도·신고구분 안에서 혼재 → 값만으로 구분 불가
위치 결정 순서: 산불위치(경위도/메르카토르) → 신고위치(경위도) → 산불위치 TM → 신고위치 TM.
TM만 있는 건은 5181·5186 두 해석 중 춘천시 안에 들어오는 쪽을 쓰되, 법정동코드가 다른 시군이면 제외하고
법정동코드가 없으면 '좌표계추정'으로 표시한다.
개인정보(가해자·신고자·책임자 이름·주소 등)는 출력하지 않는다.

출력: output/춘천시_산불_산사태.gpkg
  - 산불, 산사태 : 춘천시 경계 안의 점(읍면동, 국유림 임소반, 2025 임상 부여)
  - 산불_위치검토 : 법정동코드는 춘천시인데 좌표가 없거나 경계 밖인 산불(geometry는 있으면 유지)
  - 춘천시경계, 춘천시_읍면동경계
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
BOUNDARY_GPKG = ROOT / "output" / "춘천시_산림경영활동_숲가꾸기공간정보.gpkg"
STAND_GPKG = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
COMBINED_GPKG = ROOT / "output" / "춘천시_숲가꾸기_임상도_사업실적_결합.gpkg"
OUT = ROOT / "output" / "춘천시_산불_산사태.gpkg"
CRS = "EPSG:5179"

FIRE_PRIVATE = ["가해자성명", "가해자주소", "가해자연령", "가해자성별구분코드", "가해자성별구분코드명", "가해자직업명",
                "가해자부상원인내용", "가해자사망원인내용", "산불신고자명", "현장지휘소책임자명", "진화후책임자명"]
LANDSLIDE_PRIVATE = ["사용자명", "최초확인자ID"]
FIRE_STATUS = {"진화완료": "산불", "산불외종료": "산불외 종료(오인 등)", "신고삭제": "신고삭제"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def points(x, y, crs, index):
    return gpd.GeoSeries(gpd.points_from_xy(x, y), crs=crs, index=index).to_crs(CRS)


def locate_fires(f, city):
    ax, ay = num(f["산불위치X좌표"]), num(f["산불위치Y좌표"])
    bx, by = num(f["산불신고위치X좌표"]), num(f["산불신고위치Y좌표"])
    is_deg = lambda x: x.between(120, 135)
    is_merc = lambda x: x > 1e7
    is_tm = lambda x: x.between(1000, 2e6)

    geom = gpd.GeoSeries([None] * len(f), crs=CRS, index=f.index)
    basis = pd.Series(pd.NA, index=f.index, dtype="object")
    for mask, x, y, crs, label in [
        (is_deg(ax), ax, ay, 4326, "산불위치(경위도)"),
        (is_merc(ax), ax, ay, 3857, "산불위치(웹메르카토르)"),
        (is_deg(bx), bx, by, 4326, "신고위치(경위도)"),
    ]:
        m = mask & geom.isna()
        geom[m] = points(x[m], y[m], crs, f.index[m])
        basis[m] = label

    code_cc = f["법정동시도코드"].isin(["42", "51"]) & (f["법정동시군구코드"] == "110")
    code_other = f["법정동시군구코드"].notna() & ~code_cc
    for x, y, label in [(ax, ay, "산불위치"), (bx, by, "신고위치")]:
        m = is_tm(x) & geom.isna()
        for epsg in (5181, 5186):
            cand = points(x[m], y[m], epsg, f.index[m])
            hit = cand.within(city) & ~code_other[m]
            idx = hit[hit].index
            geom[idx] = cand[idx]
            basis[idx] = np.where(code_cc[idx], f"{label}(TM EPSG:{epsg}, 법정동코드 일치)",
                                  f"{label}(TM EPSG:{epsg}, 좌표계추정)")
        # 나머지는 춘천 밖 건. 5181(다수)로 두되, 5181이 춘천 안에 떨어지면(법정동코드가 타 시군) 5186이 맞는 해석
        rest = m & geom.isna()
        cand = points(x[rest], y[rest], 5181, f.index[rest])
        wrong = cand.within(city)
        cand[wrong] = points(x[rest][wrong], y[rest][wrong], 5186, wrong[wrong].index)
        geom[rest] = cand
        basis[rest] = f"{label}(TM, 좌표계미확정)"
    return geom, basis, code_cc


def read_fires(city):
    f = pd.read_csv(DATA / "산불정보.csv", encoding="cp949", dtype=str)
    geom, basis, code_cc = locate_fires(f, city)
    f = f.drop(columns=FIRE_PRIVATE)
    f.insert(1, "구분", f["산불진행상태코드명"].map(FIRE_STATUS).fillna(f["산불진행상태코드명"]))
    occurred = pd.to_datetime(f["산불발화일시"], format="%d-%m-%Y", errors="coerce")
    reported = pd.to_datetime(f["산불신고일"], format="%Y%m%d", errors="coerce")
    f.insert(2, "발생일", occurred.fillna(reported).dt.date)
    f.insert(3, "연도", pd.to_datetime(f["발생일"]).dt.year.astype("Int64"))
    f.insert(4, "월", pd.to_datetime(f["발생일"]).dt.month.astype("Int64"))
    f.insert(5, "위치기준", basis)
    for c in ("총산불피해면적", "미입목지면적", "제지면적", "총산불피해금액", "응급복구금액"):
        f[c] = num(f[c])
    gdf = gpd.GeoDataFrame(f, geometry=geom, crs=CRS)
    inside = gdf.geometry.within(city)
    review = gdf[code_cc & ~inside].copy()
    review["검토사유"] = np.where(review.geometry.isna(), "좌표 없음", "법정동코드는 춘천시, 좌표는 경계 밖")
    return gdf[inside].copy(), review


def read_landslides(city):
    l = pd.read_csv(DATA / "산사태신고정보.csv", encoding="cp949", dtype=str).drop(columns=LANDSLIDE_PRIVATE)
    gdf = gpd.GeoDataFrame(l, geometry=gpd.points_from_xy(num(l["표준X좌표"]), num(l["표준Y좌표"])), crs=CRS)
    gdf = gdf[gdf.geometry.within(city)].copy()
    gdf.insert(1, "구분", np.where(gdf["산사태진행상태코드명"] == "산사태삭제", "신고삭제", "유효신고"))
    reported = pd.to_datetime(gdf["산사태신고일"], format="%Y%m%d", errors="coerce")
    gdf.insert(2, "신고일", reported.dt.date)
    gdf.insert(3, "연도", reported.dt.year.astype("Int64"))
    gdf.insert(4, "월", reported.dt.month.astype("Int64"))
    for c in ("피해금액", "복구금액", "산사태피해면적", "토석피해면적", "임도피해길이", "부상인원", "사망인원", "대피인원"):
        gdf[c] = num(gdf[c])
    return gdf


def enrich(points_gdf, umd, stands, forest):
    """점 위치의 읍면동, 국유림 임소반, 2025 임상을 붙인다."""
    out = points_gdf.copy()
    for layer, cols in ((umd, ["EMD_CD", "EMD_NM"]), (stands, ["임반", "소반"]), (forest, ["임상", "수종", "영급"])):
        j = gpd.sjoin(out[["geometry"]], layer[cols + ["geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()]
        for c in cols:
            name = {"EMD_CD": "읍면동코드", "EMD_NM": "읍면동", "임반": "국유림_임반", "소반": "국유림_소반"}.get(c, f"임상2025_{c}")
            out[name] = j[c]
    out["국유림여부"] = out["국유림_임반"].notna().map({True: "Y", False: "N"})
    return out


def main():
    city_gdf = gpd.read_file(BOUNDARY_GPKG, layer="춘천시경계")
    city = city_gdf.geometry.iloc[0]
    umd = gpd.read_file(BOUNDARY_GPKG, layer="춘천시_읍면동경계")
    stands = gpd.read_file(STAND_GPKG, layer="임소반_실적요약", columns=["임반", "소반"])
    forest = gpd.read_file(COMBINED_GPKG, layer="임상도2025", columns=["임상", "수종", "영급"])

    fires, review = read_fires(city)
    landslides = read_landslides(city)
    fires = enrich(fires, umd, stands, forest)
    landslides = enrich(landslides, umd, stands, forest)

    if OUT.exists():
        OUT.unlink()
    fires.to_file(OUT, layer="산불", driver="GPKG")
    landslides.to_file(OUT, layer="산사태", driver="GPKG")
    review.to_file(OUT, layer="산불_위치검토", driver="GPKG")
    city_gdf.to_file(OUT, layer="춘천시경계", driver="GPKG")
    umd.to_file(OUT, layer="춘천시_읍면동경계", driver="GPKG")
    for name, g in (("산불", fires), ("산사태", landslides)):
        g.drop(columns="geometry").assign(X_5179=g.geometry.x.round(2), Y_5179=g.geometry.y.round(2)).to_csv(
            OUT.with_name(f"춘천시_{name}.csv"), index=False, encoding="utf-8-sig")

    print(f"산불 {len(fires)}건 (위치검토 {len(review)}건: {review['검토사유'].value_counts().to_dict()})")
    print(" 구분:", fires["구분"].value_counts().to_dict())
    print(" 위치기준:", fires["위치기준"].value_counts().to_dict())
    print(" 연도(산불만):", fires[fires["구분"] == "산불"]["연도"].value_counts().sort_index().to_dict())
    print(" 읍면동(산불만):", fires[fires["구분"] == "산불"]["읍면동"].value_counts().head(8).to_dict())
    print(f" 국유림 내 산불 {((fires['구분'] == '산불') & (fires['국유림여부'] == 'Y')).sum()}건, "
          f"피해면적 합 {fires.loc[fires['구분'] == '산불', '총산불피해면적'].sum():.2f}")
    print(f"산사태 {len(landslides)}건, 구분: {landslides['구분'].value_counts().to_dict()}")
    print(" 연도:", landslides.groupby(["연도", "구분"]).size().to_dict())
    print(" 읍면동(유효):", landslides[landslides["구분"] == "유효신고"]["읍면동"].value_counts().to_dict())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
