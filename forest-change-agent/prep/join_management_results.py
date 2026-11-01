# [통합 사본] 원본: scripts/join_management_results.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""춘천(9) 임소반도에 경영계획부 7·8·9차 산림사업실적을 임반·소반 기준으로 결합한다.

출력: output/춘천_임소반_산림사업실적.gpkg
  - 임소반_실적요약 : 임소반 폴리곤 1개당 1행, 차기×사업구분별 건수·면적·사업비·수입 요약 + 이력 문자열
  - 산림사업실적    : 실적 1건당 1행(해당 소반 폴리곤 geometry 부여, 1:N)
  - 미결합실적      : 임소반도에 해당 임반·소반이 없는 실적 (geometry 없음)
"""
from pathlib import Path
import glob

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
OUT = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"

SHEETS = {
    "조림실적": "조림",
    "숲가꾸기실적": "숲가꾸기",
    "임목생산실적": "임목생산",
    "시설실적": "시설",
    "소득실적": "소득",
    "병충해방제및산림피해복구실적": "병해충방제및산림피해복구",
}
CATEGORY_ABBR = {"조림": "조림", "숲가꾸기": "숲가", "임목생산": "임목", "시설": "시설", "소득": "소득", "병해충방제및산림피해복구": "피해"}

# 원본 2단 헤더 -> 통일 컬럼명
RENAME = {
    "면적(㏊)": "면적_ha",
    "면적": "면적_ha",
    "본수(본)": "본수",
    "노동력(명)": "노동력_명",
    "누적면적(㏊)": "누적면적_ha",
    "목재수확률(%)": "목재수확률",
    "수종 및 목재수확재적(m³)": "수종및목재수확재적",
    "수입(천원)_수종 및 목재수확재적(m³)": "수종및목재수확재적",
    "총재적(m³)": "총재적_m3",
    "기타 복구종": "기타복구종",
    "사업비(천원)_지출액": "사업비_천원",
    "사업비(천원)_지출일": "지출일",
    "수입(천원)_수입액": "수입_천원",
    "수입(천원)_수입일": "수입일",
}
NUMERIC = ["연도", "면적_ha", "본수", "노동력_명", "횟수", "누적면적_ha", "목재수확률",
           "총재적_m3", "물량", "생산량", "사업비_천원", "수입_천원"]
COLUMN_ORDER = ["차기", "사업구분", "임반", "소반", "연도", "종별", "면적_ha", "물량", "생산량",
                "수종", "본수", "조림사유", "횟수", "누적면적_ha", "목재수확종", "목재수확률",
                "수종및목재수확재적", "총재적_m3", "생산방법", "기타소득", "기타복구종",
                "노동력_명", "사업비_천원", "지출일", "수입_천원", "수입일", "원본파일", "원본행"]


def read_sheet(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    top, sub = raw.iloc[0].ffill(), raw.iloc[1]
    cols = [t if pd.isna(s) else f"{t}_{s}" for t, s in zip(top, sub)]
    df = raw.iloc[2:].copy()
    df.columns = cols
    df["원본행"] = df.index + 1  # 엑셀 행번호(1-based)
    df = df.rename(columns=RENAME)
    # 수입 아래에 붙은 '수종 및 목재수확재적'이 수입 컬럼에만 있는 시트와 독립 컬럼인 시트가 공존 → 중복 컬럼 병합
    df = df.T.groupby(level=0).first().T
    return df[df["임반"].notna() & df["소반"].notna()]


def load_results():
    frames = []
    for plan in (7, 8, 9):
        path = glob.glob(str(DATA / f"*경영계획부_춘천_{plan}차_*.xls"))[0]
        for sheet, category in SHEETS.items():
            df = read_sheet(path, sheet)
            df.insert(0, "사업구분", category)
            df.insert(0, "차기", plan)
            df["원본파일"] = Path(path).name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce")
    for c in ("임반", "소반"):
        df[c] = df[c].str.strip()
    df["연도"] = df["연도"].astype("Int64")
    df["차기"] = df["차기"].astype(int)
    return df[[c for c in COLUMN_ORDER if c in df.columns]]


def read_stand_map():
    # dbf의 lctn_daddr 필드가 254바이트에서 잘려 UTF-8 멀티바이트 문자가 깨진 행이 있음 → latin1로 읽고 손상 바이트만 버림
    path = glob.glob(str(DATA / "춘천(9)_*" / "*.shp"))[0]
    gdf = gpd.read_file(path, encoding="latin1")
    for c in gdf.columns.drop("geometry"):
        gdf[c] = gdf[c].map(lambda v: v.encode("latin1").decode("utf-8", errors="ignore") if isinstance(v, str) else v)
    for c in ("tot_area", "realt_accm", "ha_party_r"):
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce")
    return gdf.rename(columns={"cmblc_id": "임반", "sblt_id": "소반"})


def amount_column(category):
    return {"시설": "물량", "소득": "생산량"}.get(category, "면적_ha")


def summarize(results):
    rows = []
    category_order = pd.Categorical(results["사업구분"], categories=list(SHEETS.values()), ordered=True)
    for (plan, category), g in results.groupby(["차기", category_order], observed=True):
        p = f"{plan}차_{CATEGORY_ABBR[category]}"
        amt = amount_column(category)
        s = g.groupby(["임반", "소반"]).agg(
            **{f"{p}_건수": ("사업구분", "size"),
               f"{p}_{amt.split('_')[0]}": (amt, "sum"),
               f"{p}_사업비": ("사업비_천원", "sum"),
               f"{p}_수입": ("수입_천원", "sum")})
        rows.append(s)
    summary = pd.concat(rows, axis=1)

    def history(g):
        g = g.sort_values(["연도", "사업구분"])
        return "; ".join(
            f"{r.차기}차 {'' if pd.isna(r.연도) else r.연도} {r.사업구분}"
            + (f"/{r.종별}" if isinstance(r.종별, str) else "")
            for r in g.itertuples())

    totals = results.groupby(["임반", "소반"]).agg(
        실적_총건수=("사업구분", "size"),
        실적_사업비합=("사업비_천원", "sum"),
        실적_수입합=("수입_천원", "sum"),
        실적_최초연도=("연도", "min"),
        실적_최근연도=("연도", "max"),
    )
    totals["실적_사업구분"] = results.groupby(["임반", "소반"])["사업구분"].agg(lambda s: ",".join(dict.fromkeys(s)))
    totals["실적_이력"] = results.groupby(["임반", "소반"])[["차기", "연도", "사업구분", "종별"]].apply(history)
    return totals.join(summary)


def main():
    results = load_results()
    stands = read_stand_map()
    keys = stands[["임반", "소반"]].drop_duplicates()
    assert len(keys) == len(stands), "임소반도에 중복된 임반·소반 키가 있음"

    merged = results.merge(keys.assign(_hit=True), on=["임반", "소반"], how="left")
    matched = merged[merged["_hit"].notna()].drop(columns="_hit")
    unmatched = merged[merged["_hit"].isna()].drop(columns="_hit")

    summary = summarize(matched)
    stand_summary = stands.merge(summary, left_on=["임반", "소반"], right_index=True, how="left")
    count_cols = [c for c in stand_summary.columns if c.endswith("_건수") or c == "실적_총건수"]
    stand_summary[count_cols] = stand_summary[count_cols].fillna(0).astype(int)

    records = gpd.GeoDataFrame(
        matched.merge(stands[["임반", "소반", "geometry"]], on=["임반", "소반"], how="left"),
        geometry="geometry", crs=stands.crs)

    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    stand_summary.to_file(OUT, layer="임소반_실적요약", driver="GPKG")
    records.to_file(OUT, layer="산림사업실적", driver="GPKG")
    pd.DataFrame(unmatched).assign(geometry=None).pipe(
        lambda d: gpd.GeoDataFrame(d, geometry="geometry", crs=stands.crs)
    ).to_file(OUT, layer="미결합실적", driver="GPKG")
    unmatched.to_csv(OUT.with_name("미결합실적.csv"), index=False, encoding="utf-8-sig")

    # 결합 현황 리포트
    report = (merged.assign(결합=merged["_hit"].notna())
              .groupby(["차기", "사업구분"])["결합"].agg(전체="size", 결합="sum"))
    report["미결합"] = report["전체"] - report["결합"]
    report["결합률"] = (report["결합"] / report["전체"] * 100).round(1)
    report.to_csv(OUT.with_name("결합현황.csv"), encoding="utf-8-sig")
    print(report.to_string())
    print(f"\n임소반 {len(stands)}개 중 실적 있는 소반 {int((stand_summary['실적_총건수'] > 0).sum())}개")
    print(f"실적 {len(results)}건 → 결합 {len(matched)}건, 미결합 {len(unmatched)}건")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
