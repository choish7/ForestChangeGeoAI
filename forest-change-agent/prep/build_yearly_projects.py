# [통합 사본] 원본: scripts/build_yearly_projects.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""국유림 임소반 산림사업실적(7·8·9차, 2000~2025)과 춘천시 숲가꾸기공간정보(2015)를 하나의 연도별 산림사업 공간정보로 만든다.

입력
  output/춘천_임소반_산림사업실적.gpkg               (join_management_results.py 결과)
  output/춘천시_숲가꾸기_임상도_사업실적_결합.gpkg   (combine_thinning_forest_type.py 결과)
  임상도 2019·2025 51110/51110.shp

출력: output/춘천시_연도별_산림사업.gpkg
  - 연도별_산림사업    : 사업 1건당 1행. 연도·사업구분·종별·면적·사업비, 사업시작일/종료일(QGIS 시간 슬라이더용),
                         사업지의 2019·2025 우세 임상
  - 연도별_사업지요약  : 사업지(소반 또는 숲가꾸기 폴리곤) × 연도 1행. 그 해 사업 건수·구분·종별·면적·사업비
  - 사업지_연도이력    : 사업지 1개당 1행. 사업 연도 목록, 최초·최근 사업연도, 사업 공백기간
  - 연도별_통계        : (geometry 없음) 연도 × 관리주체 × 사업구분 건수·면적·사업비
  - 춘천시경계, 춘천시_읍면동경계
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine_thinning_forest_type import YEARS, overlay_forest_type, read_forest_type

ROOT = Path(__file__).resolve().parents[2]
STAND_GPKG = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
COMBINED_GPKG = ROOT / "output" / "춘천시_숲가꾸기_임상도_사업실적_결합.gpkg"
OUT = ROOT / "output" / "춘천시_연도별_산림사업.gpkg"
FOREST_COLS = ["임상", "수종", "영급", "경급"]


def to_date(series, years):
    """지출일·수입일 문자열을 날짜로. 사업연도와 다르면(이월·정산 등) 사업연도 1월 1일로 대체."""
    d = pd.to_datetime(series, errors="coerce")
    return d.where(d.dt.year == years)


def national_records():
    r = gpd.read_file(STAND_GPKG, layer="산림사업실적")
    r = r[r["연도"].notna()].copy()
    out = gpd.GeoDataFrame({
        "관리주체": "국유림",
        "출처": r["차기"].map(lambda v: f"경영계획부 {v}차"),
        "사업지ID": r["임반"] + "/" + r["소반"],
        "임반": r["임반"],
        "소반": r["소반"],
        "연도": r["연도"].astype(int),
        "사업구분": r["사업구분"],
        "종별": r["종별"].fillna(r.get("기타복구종")),
        "면적_ha": r["면적_ha"],
        "물량": r["물량"],
        "생산량": r["생산량"],
        "수종": r["수종"],
        "총재적_m3": r["총재적_m3"],
        "생산방법": r["생산방법"],
        "사업비_천원": r["사업비_천원"],
        "수입_천원": r["수입_천원"],
        "사업일자": to_date(r["지출일"], r["연도"]).fillna(to_date(r["수입일"], r["연도"])),
        "위치기준": r["차기"].map(lambda v: "9차 임소반" if v == 9 else f"{v}차 임반·소반 번호를 9차 임소반도에 대응(번호 변경 가능)"),
        "원본": r["원본파일"] + " 행" + r["원본행"].astype(str),
    }, geometry=r.geometry, crs=r.crs)
    return out


def thinning_records():
    t = gpd.read_file(COMBINED_GPKG, layer="숲가꾸기_결합")
    start = pd.to_datetime(t["시작일자"], format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(t["종료일자"], format="%Y%m%d", errors="coerce")
    out = gpd.GeoDataFrame({
        "관리주체": "춘천시",
        "출처": "숲가꾸기공간정보",
        "사업지ID": t["UID"],
        "연도": pd.to_numeric(t["사업년도"]).astype(int),
        "사업구분": "숲가꾸기",
        "종별": t["작업종"],
        "면적_ha": (pd.to_numeric(t["작업면적"], errors="coerce") / 10000).round(4),
        "사업명": t["사업명"],
        "사업일자": start,
        "종료일자": end,
        "위치기준": "숲가꾸기 사업지 폴리곤",
        "원본": "산림청_산림경영활동_숲가꾸기공간정보 UID " + t["UID"],
    }, geometry=t.geometry, crs=t.crs)
    # 이미 계산한 우세 임상 재사용
    for y in YEARS:
        for c in FOREST_COLS:
            out[f"임상{y}_{c}"] = t[f"임상{y}_{c}"]
    return out


def stand_forest_types(stands):
    """임소반(1,136개) 폴리곤별 2019·2025 우세 임상."""
    stands = stands.rename(columns={"사업지ID": "UID"})[["UID", "geometry"]]
    stands["geometry"] = stands.geometry.make_valid()
    frames = []
    for y in YEARS:
        attrs, _ = overlay_forest_type(stands, read_forest_type(y), y)
        frames.append(attrs[[f"임상{y}_{c}" for c in FOREST_COLS]])
    return pd.concat(frames, axis=1)


def join_list(s):
    return ",".join(dict.fromkeys(x for x in s if isinstance(x, str)))


def main():
    nat = national_records()
    stands = gpd.read_file(STAND_GPKG, layer="임소반_실적요약")
    stands["사업지ID"] = stands["임반"] + "/" + stands["소반"]
    print("임소반 우세 임상 계산 중…")
    ft = stand_forest_types(stands[["사업지ID", "geometry"]])
    nat = nat.merge(ft, left_on="사업지ID", right_index=True, how="left")

    rec = pd.concat([nat, thinning_records()], ignore_index=True)
    rec = gpd.GeoDataFrame(rec, geometry="geometry", crs=nat.crs)
    year_start = pd.to_datetime(rec["연도"].astype(str) + "-01-01")
    rec["사업일자"] = rec["사업일자"].fillna(year_start)
    rec["사업시작일"] = year_start.dt.date  # 시간 슬라이더: 해당 연도 전체 구간에 표시
    rec["사업종료일"] = (year_start + pd.offsets.YearEnd(0)).dt.date
    rec["사업일자"] = rec["사업일자"].dt.date
    rec["종료일자"] = pd.to_datetime(rec["종료일자"]).dt.date
    rec = rec.sort_values(["연도", "관리주체", "사업구분", "사업지ID"]).reset_index(drop=True)
    rec.insert(0, "사업ID", range(1, len(rec) + 1))

    keys = ["관리주체", "사업지ID", "연도"]
    site_year = rec.groupby(keys, sort=True).agg(
        사업건수=("사업ID", "size"),
        사업구분=("사업구분", join_list),
        종별=("종별", join_list),
        면적합_ha=("면적_ha", "sum"),
        사업비합_천원=("사업비_천원", "sum"),
        수입합_천원=("수입_천원", "sum"),
        출처=("출처", join_list),
    ).reset_index()
    geoms = rec.drop_duplicates("사업지ID").set_index("사업지ID").geometry
    site_year = gpd.GeoDataFrame(site_year, geometry=site_year["사업지ID"].map(geoms), crs=rec.crs)
    site_year["사업시작일"] = pd.to_datetime(site_year["연도"].astype(str) + "-01-01").dt.date
    site_year["사업종료일"] = pd.to_datetime(site_year["연도"].astype(str) + "-12-31").dt.date

    def gaps(years):
        ys = sorted(set(years))
        return max((b - a for a, b in zip(ys, ys[1:])), default=0)

    site = rec.groupby(["관리주체", "사업지ID"]).agg(
        사업건수=("사업ID", "size"),
        사업연도수=("연도", "nunique"),
        최초사업연도=("연도", "min"),
        최근사업연도=("연도", "max"),
        최대공백년수=("연도", gaps),
        사업연도목록=("연도", lambda s: ",".join(map(str, sorted(set(s))))),
        사업구분=("사업구분", join_list),
        사업비합_천원=("사업비_천원", "sum"),
    ).reset_index()
    for y in YEARS:
        for c in ("임상", "수종"):
            site[f"임상{y}_{c}"] = site["사업지ID"].map(rec.drop_duplicates("사업지ID").set_index("사업지ID")[f"임상{y}_{c}"])
    site = gpd.GeoDataFrame(site, geometry=site["사업지ID"].map(geoms), crs=rec.crs)
    # 사업이 없던 국유림 소반도 함께 표시
    idle = stands[~stands["사업지ID"].isin(site["사업지ID"])][["사업지ID", "geometry"]].assign(관리주체="국유림", 사업건수=0, 사업연도수=0)
    site = pd.concat([site, idle], ignore_index=True)

    stats = rec.groupby(["연도", "관리주체", "사업구분"]).agg(
        건수=("사업ID", "size"), 사업지수=("사업지ID", "nunique"),
        면적합_ha=("면적_ha", "sum"), 사업비합_천원=("사업비_천원", "sum"), 수입합_천원=("수입_천원", "sum"),
    ).reset_index()

    if OUT.exists():
        OUT.unlink()
    rec.to_file(OUT, layer="연도별_산림사업", driver="GPKG")
    site_year.to_file(OUT, layer="연도별_사업지요약", driver="GPKG")
    gpd.GeoDataFrame(site, geometry="geometry", crs=rec.crs).to_file(OUT, layer="사업지_연도이력", driver="GPKG")
    gpd.GeoDataFrame(stats).to_file(OUT, layer="연도별_통계", driver="GPKG")
    for name in ("춘천시경계", "춘천시_읍면동경계"):
        gpd.read_file(COMBINED_GPKG, layer=name).to_file(OUT, layer=name, driver="GPKG")
    pivot = stats.pivot_table(index="연도", columns="사업구분", values="건수", aggfunc="sum", fill_value=0)
    pivot.to_csv(OUT.with_name("춘천시_연도별_산림사업_건수.csv"), encoding="utf-8-sig")

    print(f"사업 {len(rec)}건 (국유림 {len(nat)}, 춘천시 숲가꾸기 {len(rec) - len(nat)}), "
          f"연도 {rec['연도'].min()}~{rec['연도'].max()}, 사업지×연도 {len(site_year)}행, 사업지 {len(site)}개")
    print(f"지출/수입일이 있는 사업 {(rec['사업일자'] != rec['사업시작일']).sum()}건")
    print(pivot.to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
