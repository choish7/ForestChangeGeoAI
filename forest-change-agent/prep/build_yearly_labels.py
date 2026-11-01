# [통합 사본] 원본: scripts/build_yearly_labels.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""임상도 2019·2024·2025판과 갱신년도(조사연도)로 2017~2025 연도별 관측 라벨 구축 (AlphaEarth 비교용).

판(구축기간): 2013(2009~2013, 갱신년도 없음 → 2017년 이전 맥락만), 2019(2015~2019), 2024(2020~2024), 2025(2025)
원칙 (권장안)
  - 갱신년도 u 에 판 v 가 기록한 상태 S 는 u 년의 '관측', u 다음 해부터 판 연도 M 까지는 '유지추정'(갱신되지 않았음 ≠ 확인된 무변화)
  - 상태 S = (입목여부, 임상) ; 수종군은 보조 기록. 영급·경급·밀도는 시간에 따라 변하므로 라벨에서 제외
  - 판마다 관측을 모아 조사연도 순으로 정렬:
        같은 상태 관측 사이의 해   → 유지추정 (두 판 이상이 같은 상태를 다른 해에 확인하면 A등급)
        다른 상태 관측 사이의 해   → 변화구간 (u_a, u_b] — 시점 평가는 구간 3년 이하만
        마지막 관측 뒤 ~ 판 연도   → 유지추정 B등급 (한 판만 지지)
        첫 관측 이전               → 미상
  - 갱신년도가 없는 판은 판 연도 시점의 약한 관측으로만 쓰고 등급을 낮춤
  - 같은 해를 두 판이 다른 상태로 말하면 '판간불일치' (주 분석 제외)
  - 유지추정 구간 안에 벌채·조림 실적(국유림 소반) 또는 산지전용 고시(필지)가 있으면 C등급(유지 라벨에서 제외)
출력: output/연도별라벨/연도별관측라벨_2017_2025.gpkg (조각), 연도별관측라벨_long.parquet (조각×연도), 진단.json
"""
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
OUT = ROOT / "output" / "연도별라벨"
VERS = {2019: "임상도 2019 51110", 2024: "임상도 2024 51110", 2025: "임상도 2025 51110"}
YEARS = list(range(2017, 2026))
MIN_HA = 0.1
CRS = "EPSG:5179"
FRTP = {"1": "침엽수림", "2": "활엽수림", "3": "혼효림", "4": "죽림", "0": "비산림"}
STOR = {"1": "입목지", "2": "무립목지", "0": "비산림"}


def norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def load(v):
    g = gpd.read_file(DATA / VERS[v] / "51110.shp", encoding="cp949",
                      columns=["STORUNST", "FRTP_CD", "KOFTR_GROU", "AGCLS_CD", "갱신년도", "geometry"])
    for c in ("STORUNST", "FRTP_CD", "KOFTR_GROU", "AGCLS_CD"):
        g[c] = g[c].map(norm)
    g["u"] = pd.to_numeric(g["갱신년도"], errors="coerce")
    g["S"] = g["STORUNST"].map(STOR).fillna("?") + "|" + np.where(g["STORUNST"].eq("1"), g["FRTP_CD"].map(FRTP).fillna("?"), "-")
    g["geometry"] = g.geometry.make_valid()
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(CRS)
    g = g[["S", "KOFTR_GROU", "AGCLS_CD", "u", "geometry"]].rename(columns={"S": f"S_{v}", "KOFTR_GROU": f"수종_{v}", "AGCLS_CD": f"영급_{v}", "u": f"u_{v}"})
    print(f"[{v}] {len(g):,} 폴리곤, 갱신년도 없음 {int(g[f'u_{v}'].isna().sum()):,}", flush=True)
    return g


def overlay():
    t = time.time()
    a, b, c = (load(v) for v in VERS)
    ab = gpd.overlay(a, b, how="intersection", keep_geom_type=True)
    print(f"[overlay] 2019∩2024 {len(ab):,} ({time.time() - t:.0f}s)", flush=True)
    abc = gpd.overlay(ab, c, how="intersection", keep_geom_type=True).explode(index_parts=False, ignore_index=True)
    abc["area_ha"] = abc.area / 1e4
    abc = abc[abc["area_ha"] >= MIN_HA].reset_index(drop=True)
    abc["조각ID"] = np.arange(len(abc))
    print(f"[overlay] 3판 조각 {len(abc):,} (≥{MIN_HA} ha) {abc['area_ha'].sum():,.0f} ha ({time.time() - t:.0f}s)", flush=True)
    return abc


def diagnose(p):
    d = {"조각": len(p), "면적_ha": round(float(p["area_ha"].sum()), 1)}
    for a, b in ((2019, 2024), (2024, 2025), (2019, 2025)):
        both = p[f"u_{a}"].notna() & p[f"u_{b}"].notna()
        same_u = both & (p[f"u_{a}"] == p[f"u_{b}"])
        regress = both & (p[f"u_{b}"] < p[f"u_{a}"])
        diff_S = p[f"S_{a}"] != p[f"S_{b}"]
        w = p["area_ha"]
        d[f"{a}→{b}"] = {
            "갱신년도_둘다있음_면적비": round(float(w[both].sum() / w.sum()), 3),
            "같은갱신년도인데_상태다름(모순)": round(float(w[same_u & diff_S].sum() / max(w[same_u].sum(), 1e-9)), 3),
            "갱신년도_역행(뒤판이_더이른조사)": round(float(w[regress].sum() / max(w[both].sum(), 1e-9)), 3),
            "갱신없이_상태다름(갱신년도 같음 또는 역행)": round(float(w[(same_u | regress) & diff_S].sum() / w.sum()), 3),
            "상태변경_면적비": round(float(w[diff_S].sum() / w.sum()), 3),
            "재조사로_같은상태확인(뒤판 갱신 > 앞판 갱신)": round(float(w[both & (p[f"u_{b}"] > p[f"u_{a}"]) & ~diff_S].sum() / w.sum()), 3)}
    return d


def records_index(p):
    """조각별 벌채·조림 실적 연도(국유림 소반)와 산지전용 고시연도."""
    rp = gpd.GeoDataFrame(p[["조각ID"]], geometry=p.representative_point(), crs=CRS)
    su = gpd.read_file(ROOT / "output" / "의사결정DB" / "forest_decision.gpkg", layer="소반", columns=["단위ID", "geometry"]).to_crs(CRS)
    j = gpd.sjoin(rp, su, predicate="within", how="left")[["조각ID", "단위ID"]].drop_duplicates("조각ID")
    import sqlite3
    r = pd.read_sql('select "단위ID","연도","사업구분" from "사업실적" where "사업구분" in (\'임목생산\',\'조림\')',
                    sqlite3.connect(ROOT / "output" / "의사결정DB" / "forest_decision.sqlite"))
    ry = r.groupby("단위ID")["연도"].apply(lambda s: sorted(set(int(x) for x in s)))
    rec = j.set_index("조각ID")["단위ID"].map(ry)
    pc = gpd.read_file(ROOT / "output" / "산지전용" / "산지전용_라벨.gpkg", columns=["고시연도", "geometry"]).to_crs(CRS)
    k = gpd.sjoin(gpd.GeoDataFrame(p[["조각ID"]], geometry=p.geometry, crs=CRS), pc, predicate="intersects")
    conv = k.groupby("조각ID")["고시연도"].apply(lambda s: sorted(set(int(x) for x in s)))
    return rec, conv, j.set_index("조각ID")["단위ID"]


def build_labels(p, rec, conv):
    rows, chg = [], []
    for r in p.itertuples(index=False):
        obs = []
        for v in VERS:
            u, S = getattr(r, f"u_{v}"), getattr(r, f"S_{v}")
            obs.append({"판": v, "u": int(u) if pd.notna(u) else v, "S": S, "약함": pd.isna(u), "M": v})
        # 같은 해 다른 상태 → 판간불일치
        by_u = {}
        for o in obs:
            by_u.setdefault(o["u"], set()).add(o["S"])
        conflict_years = {u for u, s in by_u.items() if len(s) > 1}
        seq = sorted({(o["u"], o["S"]) for o in obs if o["u"] not in conflict_years})
        strong = {(o["u"], o["S"]) for o in obs if not o["약함"]}
        recs = rec.get(r.조각ID) if isinstance(rec.get(r.조각ID), list) else []
        convs = conv.get(r.조각ID) if isinstance(conv.get(r.조각ID), list) else []
        # 변화구간
        for (ua, sa), (ub, sb) in zip(seq, seq[1:]):
            if sa != sb:
                chg.append({"조각ID": r.조각ID, "이전상태": sa, "이후상태": sb, "구간시작": ua, "구간끝": ub, "구간길이": ub - ua,
                            "양끝_조사연도": (ua, sa) in strong and (ub, sb) in strong})
        maxM = {s: max(o["M"] for o in obs if o["S"] == s) for s in {o["S"] for o in obs}}
        for t in YEARS:
            if t in conflict_years:
                rows.append((r.조각ID, t, None, "판간불일치", "X"))
                continue
            at = [s for u, s in seq if u == t]
            if at:
                rows.append((r.조각ID, t, at[0], "관측", "A" if (t, at[0]) in strong else "B"))
                continue
            prev = [(u, s) for u, s in seq if u < t]
            nxt = [(u, s) for u, s in seq if u > t]
            if not prev:
                rows.append((r.조각ID, t, None, "미상", "-"))
                continue
            ua, sa = prev[-1]
            if nxt and nxt[0][1] != sa:
                rows.append((r.조각ID, t, None, "변화구간", "-"))
                continue
            if not nxt and t > maxM[sa]:
                rows.append((r.조각ID, t, None, "미상", "-"))
                continue
            grade = "A" if (nxt and (ua, sa) in strong and nxt[0] in strong) else "B"   # 두 조사가 같은 상태를 확인
            if any(ua < y <= t for y in recs) or any(ua < y <= t + 1 for y in convs):
                grade = "C"
            rows.append((r.조각ID, t, sa, "유지추정", grade))
    lab = pd.DataFrame(rows, columns=["조각ID", "연도", "상태", "라벨유형", "등급"])
    lab[["입목여부", "임상"]] = lab["상태"].str.split("|", expand=True)
    return lab, pd.DataFrame(chg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = overlay()
    d = diagnose(p)
    print(json.dumps(d, ensure_ascii=False, indent=1), flush=True)
    rec, conv, unit = records_index(p)
    lab, chg = build_labels(p, rec, conv)
    lab = lab.merge(p[["조각ID", "area_ha"]], on="조각ID")
    p["국유림소반"] = p["조각ID"].map(unit)
    # 2013판 맥락(대표점의 2013 임상)
    m13 = gpd.read_file(DATA / "임상도 2013년 51110" / "51110.shp", encoding="cp949", columns=["FRTP_CD", "geometry"]).to_crs(CRS)
    rp = gpd.GeoDataFrame(p[["조각ID"]], geometry=p.representative_point(), crs=CRS)
    p["임상_2013(맥락)"] = gpd.sjoin(rp, m13, predicate="within", how="left").drop_duplicates("조각ID")["FRTP_CD"].map(norm).map(FRTP).values

    summ = (lab.groupby(["연도", "라벨유형", "등급"])["area_ha"].agg(["size", "sum"]).round(0).reset_index()
              .rename(columns={"size": "조각", "sum": "면적_ha"}))
    d["연도별_라벨"] = summ.to_dict("records")
    d["변화구간"] = {"건수": len(chg), "구간3년이하_양끝조사": int(((chg["구간길이"] <= 3) & chg["양끝_조사연도"]).sum()) if len(chg) else 0,
                  "전이": {f"{a} → {b}": int(n) for (a, b), n in chg.groupby(["이전상태", "이후상태"]).size().sort_values(ascending=False).head(12).items()} if len(chg) else {},
                  "구간길이": chg["구간길이"].value_counts().sort_index().to_dict() if len(chg) else {}}
    d["소요_초"] = round(time.time() - t0)
    gp = OUT / "연도별관측라벨_2017_2025.gpkg"
    gp.unlink(missing_ok=True)
    p.to_file(gp, layer="조각", driver="GPKG")
    if len(chg):
        chg.merge(p[["조각ID", "area_ha", "국유림소반", "geometry"]], on="조각ID").pipe(gpd.GeoDataFrame, crs=CRS).to_file(gp, layer="변화구간", driver="GPKG")
    lab.to_parquet(OUT / "연도별관측라벨_long.parquet", index=False)
    chg.to_csv(OUT / "변화구간.csv", index=False, encoding="utf-8-sig")
    (OUT / "진단.json").write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    piv = lab.pivot_table(index="연도", columns=["라벨유형", "등급"], values="area_ha", aggfunc="sum").round(0)
    print(piv.to_string())
    print(json.dumps(d["변화구간"], ensure_ascii=False, default=str)[:1500])
    print(f"저장: {OUT} ({d['소요_초']}s)")


if __name__ == "__main__":
    main()
