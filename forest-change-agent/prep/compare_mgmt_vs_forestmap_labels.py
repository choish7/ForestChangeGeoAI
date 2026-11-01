# [통합 사본] 원본: scripts/compare_mgmt_vs_forestmap_labels.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""국유림 산림사업실적(춘천_임소반_산림사업실적.gpkg, 7·8·9차 전체 기간) 변화라벨 ↔ 임상도 변화라벨(2013·2019·2025) 비교.

비교 단위·기간
  P2013_2019 ↔ 실적 2014~2019, 본번 소반 단위 (예: 056-000/003-000 + 003-001)
      9차에서 벌채지를 부번 소반으로 분할했는데 8차 실적은 본번 번호로 남아 있어, 소반 단위로는 실적과 임상도 변화가
      서로 다른 소반에 찍힌다 → 본번으로 묶어 비교
  P2019_2025 ↔ 실적 2020~2025, 소반 단위 (9차 번호 그대로)
실적 라벨(단위·구간별, 우선순위 순)
  수확·갱신  : 임목생산 주벌, 조림(인공갱신·조림예정지정리 등)
  솎아베기   : 숲가꾸기, 그 밖의 임목생산(기타 산물·숲가꾸기 산물 = 솎아베기 부산물)
  피해·기타  : 병해충방제·산림피해복구, 시설, 소득
  실적없음
임상도 라벨: 단위 안 변화라벨 조각 면적 비율. 변화 = harvest+gain+conversion ≥ 10%
합의 라벨(CONS): 0 확정무변화(실적없음·변화없음), 1 확정변화(수확실적·임상도변화), 2 임상도만 변화, 3 수확실적만,
                 4 솎아베기실적·변화없음, 5 피해·기타실적·변화없음, 6 수확외 실적·임상도변화

출력: output/춘천_경영실적_임상도_변화라벨_비교.gpkg
  - 단위별_비교 : 구간별 비교 단위 폴리곤 (실적 라벨·건수·면적·최초 수확연도, 임상도 비율, 합의 라벨)
  - 교차표 / 일치율 / 불일치_목록 (geometry 없음) + 같은 이름 CSV
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STANDS = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
LABELS = ROOT / "output" / "춘천시_산림변화라벨_2013_2019_2025.gpkg"
OUT = ROOT / "output" / "춘천_경영실적_임상도_변화라벨_비교.gpkg"
CRS = "EPSG:5179"
PERIODS = {"P2013_2019": {"years": (2014, 2019), "unit": "본번"},
           "P2019_2025": {"years": (2020, 2025), "unit": "소반"}}
CHANGE_SHARE_MIN = 0.10
FC_TYPES = ["harvest", "gain", "conversion", "type_shift", "ambiguous", "no_change", "nonforest", "coverage_gap"]
MGMT_ORDER = ["수확·갱신", "솎아베기", "피해·기타", "실적없음"]
CONS = {0: "확정무변화(실적없음·변화없음)", 1: "확정변화(수확실적·임상도변화)", 2: "임상도만 변화(미기록 변화 후보)",
        3: "수확실적만(임상도 무변화)", 4: "솎아베기실적·변화없음", 5: "피해·기타실적·변화없음", 6: "수확외 실적·임상도변화"}


def mgmt_prio(rec):
    harvest = rec["사업구분"].eq("조림") | (rec["사업구분"].eq("임목생산") & rec["종별"].eq("주벌"))
    thin = rec["사업구분"].eq("숲가꾸기") | rec["사업구분"].eq("임목생산")  # 기타 산물·숲가꾸기 산물 포함
    return np.select([harvest, thin], [0, 1], default=2)


def load_units():
    st = gpd.read_file(STANDS, layer="임소반_실적요약", columns=["임반", "소반"]).to_crs(CRS)
    st["소반"] = st["임반"] + "/" + st["소반"]
    st["본번"] = st["소반"].str[:11]
    return st


def forestmap_areas(st):
    """소반별 · 구간별 임상도 라벨 면적(ha)."""
    lab = gpd.read_file(LABELS, layer="변화라벨", columns=["P2013_2019_type", "P2019_2025_type"]).to_crs(CRS)
    gap = gpd.read_file(LABELS, layer="도화범위차이").to_crs(CRS)
    pieces = gpd.overlay(st[["소반", "geometry"]], lab, how="intersection", keep_geom_type=True)
    pieces["a"] = pieces.area / 1e4
    gp = gpd.overlay(st[["소반", "geometry"]], gap, how="intersection", keep_geom_type=True)
    gp["a"] = gp.area / 1e4
    out = {}
    for p in PERIODS:
        y0, y1 = int(p[1:5]), int(p[6:])
        t = pieces.pivot_table(index="소반", columns=f"{p}_type", values="a", aggfunc="sum", fill_value=0)
        t = t.reindex(columns=FC_TYPES, fill_value=0)
        t["coverage_gap"] += gp[gp[f"도화_{y0}"].ne(gp[f"도화_{y1}"])].groupby("소반")["a"].sum().reindex(t.index).fillna(0)
        out[p] = t
    return out


def compare_period(p, cfg, st, rec, fm_area):
    unit = cfg["unit"]
    a, b = cfg["years"]
    r = rec[rec["연도"].between(a, b)].merge(st[["소반", "본번"]], on="소반")
    g = r.groupby(unit).agg(prio=("prio", "min"), 실적건수=("prio", "size"), 실적면적_ha=("면적_ha", "sum"),
                            실적연도=("연도", lambda s: ",".join(map(str, sorted(set(s))))),
                            실적종별=("종별", lambda s: ",".join(dict.fromkeys(s.dropna()))),
                            차기=("차기", lambda s: ",".join(map(str, sorted(set(s))))))
    g["수확연도_최초"] = r[r["prio"] == 0].groupby(unit)["연도"].min()
    units = st.dissolve(by=unit, as_index=False)[[unit, "geometry"]] if unit == "본번" else st[[unit, "geometry"]].copy()
    units["단위면적_ha"] = units.area / 1e4
    fa = fm_area.join(st.set_index("소반")["본번"]).groupby(unit).sum(numeric_only=True)
    d = units.merge(g, left_on=unit, right_index=True, how="left").merge(fa, left_on=unit, right_index=True, how="left")
    d[FC_TYPES] = d[FC_TYPES].fillna(0)
    tot = d[FC_TYPES].sum(axis=1).replace(0, np.nan)
    for t in FC_TYPES:
        d[f"임상도_{t}_비율"] = (d[t] / tot).round(4)
    d["임상도_변화면적_ha"] = d[["harvest", "gain", "conversion"]].sum(axis=1).round(2)
    d["임상도_변화비율"] = (d["임상도_변화면적_ha"] / tot).fillna(0).round(4)
    d = d.drop(columns=FC_TYPES)
    d["실적라벨"] = d["prio"].map(dict(enumerate(MGMT_ORDER))).fillna("실적없음")
    d["실적건수"] = d["실적건수"].fillna(0).astype(int)
    fm = d["임상도_변화비율"] >= CHANGE_SHARE_MIN
    h = d["실적라벨"].eq("수확·갱신")
    none = d["실적라벨"].eq("실적없음")
    d["CONS"] = np.select(
        [h & fm, h & ~fm, none & fm, none & ~fm, d["실적라벨"].eq("솎아베기") & ~fm, d["실적라벨"].eq("피해·기타") & ~fm],
        [1, 3, 2, 0, 4, 5], default=6).astype(int)
    d["합의라벨"] = d["CONS"].map(CONS)
    d.insert(0, "구간", p)
    d.insert(1, "단위", unit)
    return d.rename(columns={unit: "단위ID"}).drop(columns="prio")


def main():
    st = load_units()
    rec = gpd.read_file(STANDS, layer="산림사업실적", ignore_geometry=True)
    rec["소반"] = rec["임반"] + "/" + rec["소반"]
    rec["prio"] = mgmt_prio(rec)
    fm = forestmap_areas(st)
    df = pd.concat([compare_period(p, c, st, rec, fm[p]) for p, c in PERIODS.items()], ignore_index=True)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)

    rows = []
    for p in PERIODS:
        d = df[df["구간"] == p]
        hr, ch = d[d["실적라벨"] == "수확·갱신"], d[d["임상도_변화비율"] >= CHANGE_SHARE_MIN]
        rows.append({"구간": p, "단위": PERIODS[p]["unit"], "실적기간": "~".join(map(str, PERIODS[p]["years"])),
                     "단위수": len(d), "수확·갱신실적_단위": len(hr),
                     "재현율(수확실적 중 임상도변화)": round((hr["임상도_변화비율"] >= CHANGE_SHARE_MIN).mean(), 3),
                     "임상도변화_단위": len(ch), "정밀도(임상도변화 중 수확실적)": round(ch["실적라벨"].eq("수확·갱신").mean(), 3),
                     "임상도변화 중 어떤 실적이든": round(ch["실적라벨"].ne("실적없음").mean(), 3),
                     "변화비율_수확실적": round(hr["임상도_변화비율"].mean(), 4),
                     "변화비율_솎아베기": round(d[d["실적라벨"] == "솎아베기"]["임상도_변화비율"].mean(), 4),
                     "변화비율_실적없음": round(d[d["실적라벨"] == "실적없음"]["임상도_변화비율"].mean(), 4)})
    agree = pd.DataFrame(rows)
    xt = pd.crosstab([df["구간"], df["실적라벨"]], df["임상도_변화비율"] >= CHANGE_SHARE_MIN).rename(
        columns={True: "임상도변화", False: "임상도무변화"}).reset_index()
    cons = df.groupby(["구간", "합의라벨"]).agg(단위수=("단위ID", "size"), 면적_ha=("단위면적_ha", "sum")).round(1).reset_index()
    cols = ["구간", "단위ID", "단위면적_ha", "실적라벨", "실적종별", "실적연도", "차기", "실적면적_ha",
            "임상도_변화비율", "임상도_변화면적_ha", "합의라벨"]
    mismatch = df[df["CONS"].isin([2, 3])][cols].sort_values(["구간", "합의라벨", "임상도_변화비율"], ascending=[True, True, False])

    if OUT.exists():
        OUT.unlink()
    df.to_file(OUT, layer="단위별_비교", driver="GPKG")
    for old in OUT.parent.glob("경영실적_임상도_비교_*.csv"):
        old.unlink()
    for name, t in {"교차표": xt, "일치율": agree, "합의라벨_요약": cons, "불일치_목록": mismatch.round(3)}.items():
        gpd.GeoDataFrame(t).to_file(OUT, layer=name, driver="GPKG")
        t.to_csv(OUT.with_name(f"경영실적_임상도_비교_{name}.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("[일치율]")
    print(agree.T.to_string())
    print("\n[교차표: 단위 수]")
    print(xt.to_string(index=False))
    print("\n[합의 라벨]")
    print(cons.to_string(index=False))
    print("\n[수확실적만 상위]")
    print(mismatch[mismatch["합의라벨"].str.startswith("수확실적만")].head(8).to_string(index=False))
    print("\n[임상도만 변화 상위]")
    print(mismatch[mismatch["합의라벨"].str.startswith("임상도만")].head(8).to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
