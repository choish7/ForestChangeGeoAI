"""춘천시 임상도 3시기(2013·2019·2025) 중첩으로 산림변화 라벨을 만든다.

라벨 규칙은 forest-change-agent/fcd/labels.py 의 classify() 를 두 구간에 그대로 적용한다.
  P1 2013→2019, P2 2019→2025  →  change_type(no_change/harvest/conversion/gain/type_shift/ambiguous/nonforest)
  궤적(trajectory) = P1 → P2   (예: harvest→gain = 벌채 후 갱신)
발생연도(약한 라벨): 뒤 시기 임상도의 갱신년도가 구간 (y0, y1] 안이면 사용 (2013 임상도에는 갱신년도 없음)

처리
  - 2013 임상도의 중복 폴리곤(동일 형상) 제거
  - 세 시기 모두 도화된 구역은 intersection 조각으로 라벨, 일부 시기에만 도화된 구역은 'coverage_gap'(도화 범위 차이)
    폴리곤으로 따로 저장하고 학습에서 제외 (union 중첩은 1시간 넘게 걸려 같은 결과를 이 방식으로 만든다)
  - 0.1 ha 미만 조각(sliver) 제거, 대표점 기준 읍면동·국유림 임반/소반 부여
  - 검증: 9차 경영실적(2020~2024 주벌·인공갱신 등)이 있는 소반에서 P2 harvest/gain 비율을 실적 없는 소반과 비교

출력: output/춘천시_산림변화라벨_2013_2019_2025.gpkg
  - 변화라벨      : 중첩 조각별 3시기 속성 + 구간별 라벨·신뢰도·발생연도 + 궤적
  - 도화범위차이  : 일부 시기에만 도화된 구역 (도화_2013/2019/2025 = Y/N)
  - 구간별_요약 / 궤적_요약 / 실적_검증 (geometry 없음)
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "forest-change-agent"))
from fcd.labels import CHANGE_TYPES, TRAIN_CLASSES, classify  # noqa: E402

YEARS = (2013, 2019, 2025)
MAPS = {2013: "임상도 2013년 51110", 2019: "임상도 2019 51110", 2025: "임상도 2025 51110"}
ATTRS = {"FRTP_CD": "FRTP", "KOFTR_GROU": "KOFTR", "AGCLS_CD": "AGCLS", "DMCLS_CD": "DMCLS",
         "DNST_CD": "DNST", "갱신년도": "UPDATE_YEAR"}
MIN_PIECE_HA = 0.1
CRS = "EPSG:5179"
OUT = ROOT / "output" / "춘천시_산림변화라벨_2013_2019_2025.gpkg"
BOUNDARY_GPKG = ROOT / "output" / "춘천시_산림경영활동_숲가꾸기공간정보.gpkg"
STAND_GPKG = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
YEARLY_GPKG = ROOT / "output" / "춘천시_연도별_산림사업.gpkg"
CHANGE_TYPES = {**CHANGE_TYPES, "coverage_gap": "임상도 도화 범위 차이 (판정 불가)"}


def load(year):
    g = gpd.read_file(ROOT / MAPS[year] / "51110.shp", encoding="cp949",
                      columns=[c for c in ATTRS if c != "갱신년도" or year != 2013])
    g = g.rename(columns=ATTRS)
    for c in ("FRTP", "KOFTR", "AGCLS", "DMCLS"):
        g[c] = g[c].astype(str).str.replace(r"\.0$", "", regex=True)
    g["geometry"] = g.geometry.make_valid()
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
    key = g.geometry.normalize().to_wkb()
    dup = key.duplicated()
    # 같은 형상인데 속성이 다른 묶음 수 (첫 번째 속성을 유지)
    n_conflict = (pd.DataFrame(g.drop(columns="geometry")).assign(_k=key.values)
                  .groupby("_k").nunique(dropna=False).gt(1).any(axis=1).sum())
    g = g[~dup].copy()
    g.columns = [f"{c}_{year}" if c != "geometry" else c for c in g.columns]
    print(f"[{year}] {len(g):,} 폴리곤 (중복 형상 {int(dup.sum()):,}개 제거, 그 중 속성 다른 묶음 {int(n_conflict)}), "
          f"면적 {g.area.sum() / 1e6:,.1f} km²")
    return g.to_crs(CRS)


def overlay_all():
    """세 시기 모두 도화된 구역은 intersection 조각으로 라벨, 일부 시기만 도화된 구역은 coverage_gap 폴리곤으로 따로.
    (union 중첩은 춘천 전체에서 1시간 넘게 걸려 intersection + 도화범위 차집합으로 같은 결과를 만든다)"""
    t = time.time()
    maps = {y: load(y) for y in YEARS}
    a, b, c = maps.values()
    ab = gpd.overlay(a, b, how="intersection", keep_geom_type=True)
    print(f"[overlay] 2013∩2019 {len(ab):,} 조각 ({time.time() - t:.0f}s)", flush=True)
    abc = gpd.overlay(ab, c, how="intersection", keep_geom_type=True)
    abc = abc.explode(index_parts=False, ignore_index=True)
    abc["area_ha"] = abc.area / 1e4
    abc = abc[abc["area_ha"] >= MIN_PIECE_HA].reset_index(drop=True)
    print(f"[overlay] 3시기 intersection {len(abc):,} 조각 (≥{MIN_PIECE_HA} ha), {time.time() - t:.0f}s", flush=True)

    cover = {y: g.geometry.union_all() for y, g in maps.items()}
    allc = cover[2013].intersection(cover[2019]).intersection(cover[2025])
    anyc = cover[2013].union(cover[2019]).union(cover[2025])
    gap = gpd.GeoDataFrame(geometry=[anyc.difference(allc)], crs=CRS).explode(index_parts=False, ignore_index=True)
    gap["area_ha"] = gap.area / 1e4
    gap = gap[gap["area_ha"] >= MIN_PIECE_HA].reset_index(drop=True)
    pts = gap.representative_point()
    for y in YEARS:
        gap[f"도화_{y}"] = pts.within(cover[y]).map({True: "Y", False: "N"})
    print(f"[coverage] 일부 시기만 도화된 구역 {len(gap):,}개 {gap['area_ha'].sum():,.0f} ha, {time.time() - t:.0f}s", flush=True)
    return abc, gap


def label_period(df, y0, y1):
    """fcd.labels.classify 를 구간에 적용. 한쪽 임상도에 없는 면적은 coverage_gap."""
    sub = df[[f"FRTP_{y0}", f"KOFTR_{y0}", f"AGCLS_{y0}", f"FRTP_{y1}", f"KOFTR_{y1}", f"AGCLS_{y1}"]].copy()
    sub["UPDATE_YEAR"] = df[f"UPDATE_YEAR_{y1}"]
    lab = classify(sub, y0, y1)
    gap = df[f"FRTP_{y0}"].isna() | df[f"FRTP_{y1}"].isna() | df[f"FRTP_{y0}"].eq("nan") | df[f"FRTP_{y1}"].eq("nan")
    p = f"P{y0}_{y1}"
    out = pd.DataFrame(index=df.index)
    out[f"{p}_type"] = lab["change_type"].mask(gap, "coverage_gap")
    out[f"{p}_changed"] = lab["changed"].mask(gap, 0).astype("int8")
    out[f"{p}_conf"] = lab["label_conf"].mask(gap, "none")
    out[f"{p}_year"] = lab["change_year"].mask(gap)
    return out


def attach_location(df):
    pts = gpd.GeoDataFrame(geometry=df.representative_point(), crs=df.crs)
    umd = gpd.read_file(BOUNDARY_GPKG, layer="춘천시_읍면동경계")[["EMD_NM", "geometry"]].to_crs(CRS)
    stands = gpd.read_file(STAND_GPKG, layer="임소반_실적요약", columns=["임반", "소반"]).to_crs(CRS)
    for layer, cols in ((umd, ["EMD_NM"]), (stands, ["임반", "소반"])):
        j = gpd.sjoin(pts, layer, how="left", predicate="within")
        j = j[~j.index.duplicated()]
        for c in cols:
            df[{"EMD_NM": "읍면동", "임반": "국유림_임반", "소반": "국유림_소반"}[c]] = j[c]
    return df


def validate_with_records(df):
    """9차 실적(2020~2024)이 있는 국유림 소반에서 P2019_2025 라벨 면적 비율 비교."""
    rec = gpd.read_file(YEARLY_GPKG, layer="연도별_산림사업", ignore_geometry=True)
    rec = rec[(rec["관리주체"] == "국유림") & rec["연도"].between(2020, 2024)]
    groups = {
        "주벌·임목생산(모두베기 등)": rec[rec["사업구분"].eq("임목생산") & rec["종별"].isin(["주벌", "기타 산물"])],
        "조림(인공갱신·예정지정리)": rec[rec["사업구분"].eq("조림")],
        "솎아베기·숲가꾸기": rec[rec["사업구분"].eq("숲가꾸기")],
    }
    nat = df[df["국유림_임반"].notna()].copy()
    nat["sid"] = nat["국유림_임반"] + "/" + nat["국유림_소반"]
    any_rec = set(rec["사업지ID"])
    rows = []
    for name, r in list(groups.items()) + [("실적 없음(2020~2024)", None)]:
        m = nat["sid"].isin(set(r["사업지ID"])) if r is not None else ~nat["sid"].isin(any_rec)
        s = nat[m]
        tot = s["area_ha"].sum()
        share = s.groupby("P2019_2025_type")["area_ha"].sum() / tot if tot else pd.Series(dtype=float)
        rows.append({"소반구분": name, "소반수": s["sid"].nunique(), "면적_ha": round(tot, 1),
                     **{f"{k}_비율": round(share.get(k, 0), 3) for k in
                        ("harvest", "gain", "conversion", "no_change", "type_shift", "ambiguous", "coverage_gap")}})
    return pd.DataFrame(rows)


def main():
    df, gap = overlay_all()
    for y0, y1 in ((2013, 2019), (2019, 2025)):
        df = df.join(label_period(df, y0, y1))
    df["trajectory"] = df["P2013_2019_type"] + "→" + df["P2019_2025_type"]
    both_train = df["P2013_2019_type"].isin(TRAIN_CLASSES) & df["P2019_2025_type"].isin(TRAIN_CLASSES)
    df["학습가능"] = np.where(both_train, "both", np.where(
        df["P2013_2019_type"].isin(TRAIN_CLASSES), "P1_only",
        np.where(df["P2019_2025_type"].isin(TRAIN_CLASSES), "P2_only", "none")))
    df = attach_location(df)
    df.insert(0, "piece_id", [f"L{i:06d}" for i in range(len(df))])
    df = df.drop(columns=[c for c in df.columns if c.startswith("UPDATE_YEAR_")] + ["UPDATE_YEAR_2013"], errors="ignore")

    period = []
    for p in ("P2013_2019", "P2019_2025"):
        s = df.groupby(f"{p}_type")["area_ha"].agg(조각수="count", 면적_ha="sum")
        y0, y1 = int(p[1:5]), int(p[6:])
        g = gap[gap[f"도화_{y0}"].ne(gap[f"도화_{y1}"])]  # 이 구간 두 시기 중 한쪽만 도화
        s.loc["coverage_gap"] = [len(g), g["area_ha"].sum()]
        s = s.round(1)
        s["면적비율"] = (s["면적_ha"] / s["면적_ha"].sum()).round(3)
        s.insert(0, "구간", p)
        s["설명"] = s.index.map(CHANGE_TYPES)
        period.append(s.reset_index().rename(columns={f"{p}_type": "change_type"}))
    period = pd.concat(period, ignore_index=True)
    traj = (df.groupby("trajectory")["area_ha"].agg(조각수="count", 면적_ha="sum")
            .sort_values("면적_ha", ascending=False).round(1).reset_index())
    check = validate_with_records(df)

    if OUT.exists():
        OUT.unlink()
    gpd.GeoDataFrame(df, geometry="geometry", crs=CRS).to_file(OUT, layer="변화라벨", driver="GPKG")
    gap.to_file(OUT, layer="도화범위차이", driver="GPKG")
    for name, t in (("구간별_요약", period), ("궤적_요약", traj), ("실적_검증", check)):
        gpd.GeoDataFrame(t).to_file(OUT, layer=name, driver="GPKG")
        t.to_csv(OUT.with_name(f"산림변화라벨_{name}.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(period.to_string(index=False))
    print(traj.head(15).to_string(index=False))
    print(check.to_string(index=False))
    for p in ("P2013_2019", "P2019_2025"):
        y = df[df[f"{p}_changed"] == 1][f"{p}_year"].value_counts().sort_index()
        print(f"{p} 변화 발생연도(갱신년도 약한 라벨): {y.to_dict()}")
    print(f"학습가능: {df.groupby('학습가능')['area_ha'].sum().round(0).to_dict()}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
