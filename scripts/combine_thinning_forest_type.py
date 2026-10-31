"""춘천시 숲가꾸기공간정보(2015 사업지)에 2019·2025 임상도와 국유림 임소반 산림사업실적을 공간 결합한다.

입력
  output/춘천시_산림경영활동_숲가꾸기공간정보.gpkg   (clip_thinning_to_chuncheon.py 결과)
  output/춘천_임소반_산림사업실적.gpkg               (join_management_results.py 결과)
  임상도 2019 51110/51110.shp, 임상도 2025 51110/51110.shp (1:5,000 임상도, 코드는 DATA003_임상도(5000).pdf 기준)

출력: output/춘천시_숲가꾸기_임상도_사업실적_결합.gpkg
  - 숲가꾸기_결합        : 숲가꾸기 폴리곤 1개당 1행. 연도별 우세 임상(면적 최대 임분 유형)·구성, 2019→2025 변화,
                           겹치는 국유림 임소반(면적 최대)과 그 소반의 7·8·9차 산림사업실적 요약
  - 숲가꾸기x임상도2019  : 숲가꾸기 ∩ 2019 임상도 교차 조각(면적 포함)
  - 숲가꾸기x임상도2025  : 숲가꾸기 ∩ 2025 임상도 교차 조각
  - 숲가꾸기x임소반      : 숲가꾸기 ∩ 국유림 임소반 교차 조각
  - 임상도2019, 임상도2025 : 춘천시 임상도 전체(코드명 해석 컬럼 추가) — 지도 배경용
  - 춘천시경계, 춘천시_읍면동경계
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THINNING_GPKG = ROOT / "output" / "춘천시_산림경영활동_숲가꾸기공간정보.gpkg"
STAND_GPKG = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
OUT = ROOT / "output" / "춘천시_숲가꾸기_임상도_사업실적_결합.gpkg"
YEARS = (2019, 2025)

# DATA003_임상도(5000).pdf 코드 정의서
STORUNST = {"1": "입목지", "2": "무립목지", "0": "비산림"}
FROR = {"1": "인공림", "2": "천연림", "0": "무립목지/비산림"}
FRTP = {"1": "침엽수림", "2": "활엽수림", "3": "혼효림", "4": "죽림", "0": "무립목지/비산림"}
KOFTR = {
    "11": "소나무", "12": "잣나무", "13": "낙엽송", "14": "리기다소나무", "15": "곰솔", "16": "전나무",
    "17": "편백나무", "18": "삼나무", "19": "가문비나무", "20": "비자나무", "21": "은행나무", "10": "기타침엽수",
    "31": "상수리나무", "32": "신갈나무", "33": "굴참나무", "34": "기타참나무류", "35": "오리나무",
    "36": "고로쇠나무", "37": "자작나무", "38": "박달나무", "39": "밤나무", "40": "물푸레나무",
    "41": "서어나무", "42": "때죽나무", "43": "호두나무", "44": "백합나무", "45": "포플러", "46": "벚나무",
    "47": "느티나무", "48": "층층나무", "49": "아까시나무", "30": "기타활엽수", "61": "가시나무",
    "62": "구실잣밤나무", "63": "녹나무", "64": "굴거리나무", "65": "황칠나무", "66": "사스레피나무",
    "67": "후박나무", "68": "새덕이", "60": "기타상록활엽수", "77": "침활혼효림", "78": "죽림",
    "81": "미립목지", "82": "제지", "83": "관목덤불", "91": "주거지", "92": "초지", "93": "경작지",
    "94": "수체", "95": "과수원", "99": "비산림",
}
DMCLS = {"0": "치수", "1": "소경목", "2": "중경목", "3": "대경목"}
DNST = {"A": "소", "B": "중", "C": "밀"}
CLASS_KEYS = ["FROR_CD", "FRTP_CD", "KOFTR_GROU", "DMCLS_CD", "AGCLS_CD", "DNST_CD"]  # 임분고는 연속적으로 변해 유형 구분에서 제외
MIN_STAND_RATIO = 0.10  # 숲가꾸기 면적의 10% 미만 겹침은 경계 불일치에 의한 잔편으로 보고 소반 속성을 붙이지 않음
DECODED = {"임종": ("FROR_CD", FROR), "임상": ("FRTP_CD", FRTP), "수종": ("KOFTR_GROU", KOFTR),
           "경급": ("DMCLS_CD", DMCLS), "밀도": ("DNST_CD", DNST)}
STAND_SUMMARY_COLS = ["실적_총건수", "실적_사업구분", "실적_최초연도", "실적_최근연도", "실적_사업비합", "실적_수입합", "실적_이력"]


def norm_code(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def read_forest_type(year):
    gdf = gpd.read_file(ROOT / f"임상도 {year} 51110" / "51110.shp", encoding="cp949")
    gdf = gdf[[c for c in gdf.columns if c in CLASS_KEYS + ["HEIGHT", "STORUNST", "MAP_LABEL", "갱신년도", "geometry"]]]
    for c in CLASS_KEYS + ["HEIGHT", "STORUNST"]:
        gdf[c] = gdf[c].map(norm_code)
    gdf["입목존재"] = gdf["STORUNST"].map(STORUNST)
    for name, (col, table) in DECODED.items():
        gdf[name] = gdf[col].map(table)
    gdf["영급"] = gdf["AGCLS_CD"].map(lambda v: f"{v}영급" if v and v.isdigit() and v != "0" else None)
    gdf["임분고_m"] = pd.to_numeric(gdf["HEIGHT"], errors="coerce")
    unknown = {c: sorted(set(gdf[col].dropna()) - set(t)) for c, (col, t) in DECODED.items()}
    unknown = {k: v for k, v in unknown.items() if v}
    if unknown:
        print(f"  {year} 코드표에 없는 값: {unknown}")
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def class_label(r):
    parts = [r["수종"] or r["임상"], r["경급"], r["영급"], f"밀도{r['밀도']}" if r["밀도"] else None]
    return " ".join(p for p in parts if isinstance(p, str)) or "미분류"


def overlay_forest_type(thinning, ft, year):
    pieces = gpd.overlay(thinning[["UID", "geometry"]], ft, how="intersection", keep_geom_type=True)
    pieces["면적_m2"] = pieces.area.round(1)
    pieces["유형"] = pieces.apply(class_label, axis=1)
    pieces = pieces[pieces["면적_m2"] > 0]

    by_class = (pieces.groupby(["UID", "유형"] + CLASS_KEYS, dropna=False)["면적_m2"].sum().reset_index()
                .sort_values(["UID", "면적_m2"], ascending=[True, False]))
    total = thinning.set_index("UID").area
    by_class["비율"] = by_class["면적_m2"] / by_class["UID"].map(total)
    top = by_class.drop_duplicates("UID").set_index("UID")

    p = f"임상{year}_"
    out = pd.DataFrame(index=top.index)
    for name, (col, table) in DECODED.items():
        out[p + name] = top[col].map(table)
    out[p + "영급"] = top["AGCLS_CD"].map(lambda v: f"{v}영급" if isinstance(v, str) and v.isdigit() and v != "0" else None)
    # 우세 유형 조각들의 면적가중 평균 임분고(임분고코드 = 구간 하한 m)
    dom = pieces.merge(top[CLASS_KEYS].reset_index(), on=["UID"] + CLASS_KEYS)
    dom = dom[dom["임분고_m"].notna()]
    out[p + "임분고_m"] = ((dom["임분고_m"] * dom["면적_m2"]).groupby(dom["UID"]).sum()
                         / dom.groupby("UID")["면적_m2"].sum()).round(1)
    out[p + "우세비율"] = top["비율"].round(3)
    out[p + "피복비율"] = (by_class.groupby("UID")["비율"].sum().clip(upper=1)).round(3)
    out[p + "구성"] = by_class.groupby("UID").apply(
        lambda g: "; ".join(f"{r.유형} {r.비율:.0%}" for r in g.head(3).itertuples()), include_groups=False)
    out[p + "_key"] = top[CLASS_KEYS].astype(str).agg("|".join, axis=1)
    pieces.insert(0, "임상도연도", year)
    return out, pieces


def overlay_stands(thinning, stands):
    pieces = gpd.overlay(thinning[["UID", "geometry"]], stands, how="intersection", keep_geom_type=True)
    pieces["면적_m2"] = pieces.area.round(1)
    pieces = pieces[pieces["면적_m2"] > 0].sort_values(["UID", "면적_m2"], ascending=[True, False])
    total = thinning.set_index("UID").area
    pieces["숲가꾸기대비비율"] = (pieces["면적_m2"] / pieces["UID"].map(total)).round(3)

    top = pieces.drop_duplicates("UID").set_index("UID")
    out = pd.DataFrame(index=top.index)
    out["국유림_겹침비율"] = pieces.groupby("UID")["숲가꾸기대비비율"].sum().clip(upper=1).round(3)
    out["국유림_소반목록"] = pieces.groupby("UID").apply(
        lambda g: "; ".join(f"{r.임반}/{r.소반} {r.숲가꾸기대비비율:.1%}" for r in g.itertuples()), include_groups=False)
    real = out["국유림_겹침비율"] >= MIN_STAND_RATIO
    out["국유림_임반"] = top["임반"].where(real)
    out["국유림_소반"] = top["소반"].where(real)
    for c in STAND_SUMMARY_COLS + ["실적_2015숲가꾸기"]:
        out["국유림_" + c] = top[c].where(real)
    out["국유림여부"] = real.map({True: "Y", False: "경계접촉"})
    return out, pieces


def main():
    thinning = gpd.read_file(THINNING_GPKG, layer="숲가꾸기공간정보")
    assert thinning["UID"].is_unique
    stands = gpd.read_file(STAND_GPKG, layer="임소반_실적요약")
    stands = stands[["임반", "소반", "tot_area"] + STAND_SUMMARY_COLS + ["geometry"]]

    # 2015 숲가꾸기 사업지와 같은 해(2015) 국유림 숲가꾸기 실적 — 소반별 종별 목록
    records = gpd.read_file(STAND_GPKG, layer="산림사업실적", ignore_geometry=True)
    same_year = (records[(records["연도"] == 2015) & (records["사업구분"] == "숲가꾸기")]
                 .groupby(["임반", "소반"])["종별"].agg(lambda s: ",".join(dict.fromkeys(s.dropna()))))
    stands = stands.merge(same_year.rename("실적_2015숲가꾸기"), left_on=["임반", "소반"], right_index=True, how="left")

    result = thinning.set_index("UID")
    layers = {}
    for year in YEARS:
        print(f"{year} 임상도 결합 중…")
        ft = read_forest_type(year)
        attrs, pieces = overlay_forest_type(thinning, ft, year)
        result = result.join(attrs)
        layers[f"숲가꾸기x임상도{year}"] = pieces
        layers[f"임상도{year}"] = ft

    a, b = (f"임상{y}_" for y in YEARS)
    has_both = result[a + "_key"].notna() & result[b + "_key"].notna()
    result["임분유형변화"] = (result[a + "_key"] != result[b + "_key"]).where(has_both).map({True: "Y", False: "N"})
    for name in ("임상", "수종", "영급", "경급", "밀도"):
        changed = (result[a + name].fillna("") != result[b + name].fillna("")).where(has_both)
        result[f"{name}변화"] = changed.map({True: "Y", False: "N"})
    result = result.drop(columns=[a + "_key", b + "_key"])

    stand_attrs, stand_pieces = overlay_stands(thinning, stands)
    result = result.join(stand_attrs)
    result["국유림여부"] = result["국유림여부"].fillna("N")
    layers["숲가꾸기x임소반"] = stand_pieces

    result = gpd.GeoDataFrame(result.reset_index(), geometry="geometry", crs=thinning.crs)
    if OUT.exists():
        OUT.unlink()
    result.to_file(OUT, layer="숲가꾸기_결합", driver="GPKG")
    for name in ("숲가꾸기x임상도2019", "숲가꾸기x임상도2025", "숲가꾸기x임소반", "임상도2019", "임상도2025"):
        layers[name].to_file(OUT, layer=name, driver="GPKG")
    for name in ("춘천시경계", "춘천시_읍면동경계"):
        gpd.read_file(THINNING_GPKG, layer=name).to_file(OUT, layer=name, driver="GPKG")
    stands.to_file(OUT, layer="국유림_임소반", driver="GPKG")

    print(f"\n숲가꾸기 {len(result)}개")
    for y in YEARS:
        print(f"  {y} 임상도 결합 {result[f'임상{y}_임상'].notna().sum()}개, 평균 피복비율 {result[f'임상{y}_피복비율'].mean():.3f}")
    print(f"  2019→2025 우세 임분유형 변화 {(result['임분유형변화'] == 'Y').sum()}개 "
          f"(수종 {(result['수종변화'] == 'Y').sum()}, 영급 {(result['영급변화'] == 'Y').sum()}, "
          f"임상 {(result['임상변화'] == 'Y').sum()})")
    print(f"  국유림 임소반 {MIN_STAND_RATIO:.0%} 이상 겹침 {(result['국유림여부'] == 'Y').sum()}개 "
          f"(경계접촉 {(result['국유림여부'] == '경계접촉').sum()}개), "
          f"그 중 2015 숲가꾸기 실적 있는 소반 {result['국유림_실적_2015숲가꾸기'].notna().sum()}개")
    print(result[f"임상{YEARS[1]}_수종"].value_counts().head(8).to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
