"""[4.4.2] GeoAI 에이전트 도구 모음.

원칙
- 면적·통계·순위 등 모든 수치는 여기(GIS·통계엔진)에서만 계산한다. LLM 은 수치를 만들지 않는다.
- 모든 결과는 {"data": …, "meta": {source, ref_date, crs, unit, warnings}} 형태로 반환한다.
- 호출 이력은 Session.audit 에 남아 의사결정 카드의 '분석이력'이 된다.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import spatial, verify
from .data import change_polygons, load_forest_map, load_mgmt_plan, regions
from .labels import CHANGE_TYPES

CHANGE_KINDS = ["harvest", "conversion", "gain", "type_shift"]
FRTP_NM = {"0": "무립목지/비산림", "1": "침엽수림", "2": "활엽수림", "3": "혼효림"}


# ================================================================== 세션

@dataclass
class Session:
    query: str = ""
    spec: dict = field(default_factory=dict)
    audit: list = field(default_factory=list)
    evidence: list = field(default_factory=list)       # 도구 출력 수치 (근거 검증용)
    layers: dict = field(default_factory=dict)          # 지도 레이어 (EPSG:4326 GeoJSON)

    def log(self, tool: str, args: dict, result: dict):
        blob = json.dumps(result, ensure_ascii=False, default=str)
        self.audit.append({"step": len(self.audit) + 1, "tool": tool, "input": args,
                           "output_sha1": hashlib.sha1(blob.encode()).hexdigest()[:10],
                           "warnings": result.get("meta", {}).get("warnings", []),
                           "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.evidence.extend(verify.collect_numbers(result.get("data", {})))


def _meta(source: str, ref_date: str, unit: str = "ha", warnings=None) -> dict:
    return {"source": source, "ref_date": ref_date, "crs": C.CRS_WORK, "unit": unit,
            "warnings": [w for w in (warnings or []) if w]}


# ================================================================== 공통

@lru_cache
def _forest_2025() -> gpd.GeoDataFrame:
    return load_forest_map(2025)


@lru_cache
def _model():
    return joblib.load(C.MODEL) if C.MODEL.exists() else None


def _region(region_id: str) -> gpd.GeoSeries:
    r = regions()
    row = r[r["region_id"] == region_id]
    if row.empty:
        raise ValueError(f"알 수 없는 region_id '{region_id}'. list_regions 또는 set_analysis_spec 으로 확인하세요.")
    return row.iloc[0]


def _clip(gdf: gpd.GeoDataFrame, geom) -> gpd.GeoDataFrame:
    sub = gdf.iloc[gdf.sindex.query(geom, predicate="intersects")]
    return gpd.clip(sub, geom) if len(sub) else sub


def _to_layer(gdf: gpd.GeoDataFrame, cols: list[str]) -> dict:
    return json.loads(gdf[cols + ["geometry"]].to_crs(C.CRS_GEE).to_json())


def _resolve_region(text: str) -> list[dict]:
    r = regions()
    t = text.replace(" ", "")
    exact = r[r["region_id"].eq(text) | r["name"].str.replace(" ", "").eq(t)]
    if len(exact):
        return exact[["region_id", "name", "level"]].to_dict("records")
    if not any(k in t for k in ("면", "읍", "동", "임반")) and ("춘천" in t or t == "국유림"):
        lvl = "national_forest" if "국유림" in t else "city"
        return r[r["level"] == lvl][["region_id", "name", "level"]].to_dict("records")
    hit = r[r["name"].str.replace(" ", "").str.contains(t.replace("국유림", "").replace("춘천", ""), regex=False)]
    return hit[["region_id", "name", "level"]].head(10).to_dict("records")


# ================================================================== 도구 구현

def list_regions(session: Session, level: str = "all") -> dict:
    r = regions()
    if level != "all":
        r = r[r["level"] == level]
    rows = r.assign(area_ha=(r.area / 1e4).round(1))[["region_id", "name", "level", "area_ha"]]
    if len(rows) > 40:
        rows = rows.head(40)
    return {"data": {"regions": rows.to_dict("records"), "total": int(len(r))},
            "meta": _meta("임상도 2025 범위 / 북부지방산림청 춘천 경영계획부(9차)", "2025",
                          warnings=["읍면 단위는 국유림 경영계획구 범위에 한정 (행정경계 파일 미보유)"])}


def set_analysis_spec(session: Session, region: str, year_from: int, year_to: int,
                      change_types: list[str], policy_purpose: str, outputs: list[str] | None = None) -> dict:
    """질의 해석 결과를 구조화·검증 (1단계)."""
    warns, amb = [], []
    cands = _resolve_region(region)
    if not cands:
        amb.append(f"지역 '{region}' 을(를) 찾지 못함")
    elif len(cands) > 1:
        amb.append(f"지역 후보 {len(cands)}개 — 사용자 확인 필요")
    lo, hi = min(C.AE_YEARS), max(C.AE_YEARS)
    if not (lo <= year_from < year_to <= hi):
        amb.append(f"기간 {year_from}~{year_to} 은 AlphaEarth 가용기간 {lo}~{hi} 밖")
    if (year_from, year_to) != (C.YEAR_FROM, C.YEAR_TO):
        warns.append(f"임상도 비교 라벨·학습모형은 {C.YEAR_FROM}→{C.YEAR_TO} 기준. 다른 기간은 AlphaEarth 즉석 스캔만 적용")
    bad = [t for t in change_types if t not in CHANGE_KINDS]
    if bad:
        amb.append(f"알 수 없는 변화유형 {bad} (허용: {CHANGE_KINDS})")
    spec = {"region_candidates": cands, "region_id": cands[0]["region_id"] if len(cands) == 1 else None,
            "period": {"from": year_from, "to": year_to}, "change_types": change_types,
            "policy_purpose": policy_purpose, "outputs": outputs or ["stats", "map", "decision_card"],
            "ambiguities": amb}
    session.spec = spec
    return {"data": spec, "meta": _meta("질의 해석", time.strftime("%Y-%m-%d"), unit="-", warnings=warns)}


def query_forest_status(session: Session, region_id: str) -> dict:
    """현재(2025 임상도) 산림 현황: 임상·영급·주요 수종 면적."""
    reg = _region(region_id)
    fm = _clip(_forest_2025(), reg.geometry)
    fm = fm.assign(area_ha=fm.area / 1e4)
    total = float(fm["area_ha"].sum())
    by_type = fm.groupby(fm["FRTP"].map(FRTP_NM))["area_ha"].sum().round(1).to_dict()
    by_age = fm.groupby("AGCLS")["area_ha"].sum().round(1).to_dict()
    by_sp = fm.groupby("KOFTR")["area_ha"].sum().sort_values(ascending=False).head(8).round(1).to_dict()
    return {"data": {"region": reg["name"], "total_ha": round(total, 1), "by_forest_type_ha": by_type,
                     "by_age_class_ha": by_age, "top_species_group_code_ha": by_sp},
            "meta": _meta("임상도 2025 (1:5,000, 산림청)", "2025",
                          warnings=[verify.area_consistency(by_type, total)])}


def query_change_map(session: Session, region_id: str, change_types: list[str] | None = None,
                     top_n: int = 10) -> dict:
    """임상도 2019→2025 비교 변화 폴리곤 조회: 유형별 면적·패치수·갱신년도 분포·대표 패치."""
    reg = _region(region_id)
    types = change_types or ["harvest", "conversion", "gain"]
    cp = _clip(change_polygons(), reg.geometry)
    cp = cp.assign(area_ha=cp.area / 1e4)
    forest19 = float(cp.loc[cp["FRTP_2019"].isin(["1", "2", "3"]), "area_ha"].sum())
    sel = cp[cp["change_type"].isin(types)]
    by_type = {t: {"area_ha": round(float(sel.loc[sel.change_type == t, "area_ha"].sum()), 2),
                   "patches": int((sel.change_type == t).sum()), "label": CHANGE_TYPES[t]} for t in types}
    total = float(sel["area_ha"].sum())
    yr = sel["change_year"].dropna().astype(int).value_counts().sort_index()
    top = sel.sort_values("area_ha", ascending=False).head(top_n)
    cen = top.geometry.centroid.to_crs(C.CRS_GEE)
    top_rows = [{"patch_id": p, "type": t, "area_ha": round(a, 2), "lon": round(c.x, 5), "lat": round(c.y, 5)}
                for p, t, a, c in zip(top.patch_id, top.change_type, top.area_ha, cen)]
    session.layers["변화 폴리곤(임상도 비교)"] = _to_layer(sel.assign(area_ha=sel.area_ha.round(2)),
                                                    ["patch_id", "change_type", "area_ha"])
    return {"data": {"region": reg["name"], "period": f"{C.YEAR_FROM}→{C.YEAR_TO}",
                     "forest_2019_ha": round(forest19, 1), "changed_total_ha": round(total, 2),
                     "change_rate_pct": round(100 * total / forest19, 2) if forest19 else None,
                     "by_type": by_type, "map_update_year_ha": {int(k): int(v) for k, v in yr.items()},
                     "largest_patches": top_rows},
            "meta": _meta(f"임상도 {C.YEAR_FROM}·{C.YEAR_TO} 속성비교 변화라벨", f"{C.YEAR_TO}",
                          warnings=[verify.area_consistency({k: v["area_ha"] for k, v in by_type.items()}, total),
                                    "변화연도는 임상도 갱신년도(도면 갱신시점) 기준 — 실제 발생연도와 다를 수 있음"])}


def query_mgmt_plan(session: Session, region_id: str) -> dict:
    """국유림 경영계획부(9차) 소반 현황: 면적·축적·임반 수."""
    reg = _region(region_id)
    mp = load_mgmt_plan()
    sub = mp.iloc[mp.sindex.query(reg.geometry, predicate="intersects")]
    sub = sub[sub.representative_point().within(reg.geometry)]
    if sub.empty:
        return {"data": {"region": reg["name"], "n_subcompartments": 0},
                "meta": _meta("경영계획부", "9차", warnings=["해당 지역에 국유림 경영계획 소반 없음"])}
    area = pd.to_numeric(sub["area_ha"], errors="coerce")
    stock = pd.to_numeric(sub["stock_m3"], errors="coerce")
    return {"data": {"region": reg["name"], "n_compartments": int(sub["cmblc_id"].nunique()),
                     "n_subcompartments": int(len(sub)), "area_ha": round(float(area.sum()), 1),
                     "stock_m3": round(float(stock.sum()), 0),
                     "stock_m3_per_ha": round(float(stock.sum() / area.sum()), 1) if area.sum() else None,
                     "mgmt_office": sub["mnoff_nm"].iloc[0], "plan_round": int(sub["nxtrm_no"].iloc[0])},
            "meta": _meta("북부지방산림청 춘천국유림관리소 경영계획부", f"{int(sub['nxtrm_no'].iloc[0])}차 계획",
                          unit="ha, m³")}


def scan_alphaearth_change(session: Session, region_id: str, year_from: int, year_to: int) -> dict:
    """GEE 에서 AlphaEarth 연도쌍 코사인 거리로 즉석 변화 스캔 (임의 연도쌍 가능)."""
    from . import gee
    reg = _region(region_id)
    mdl = _model()
    thr = mdl["threshold"] if mdl else 0.2
    geom = gpd.GeoSeries([reg.geometry], crs=C.CRS_WORK).to_crs(C.CRS_GEE)
    res = gee.change_scan(gee.to_ee_geometry(geom), year_from, year_to, thr)
    res = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}
    if res.get("dist_pct"):
        res["dist_pct"] = {k: round(v, 4) for k, v in res["dist_pct"].items() if v is not None}
    if res.get("forest_ha"):
        res["changed_rate_pct"] = round(100 * res["changed_ha"] / res["forest_ha"], 2)
    if res["patches"]:
        pts = gpd.GeoDataFrame(res["patches"], geometry=gpd.points_from_xy(
            [p["lon"] for p in res["patches"]], [p["lat"] for p in res["patches"]]), crs=C.CRS_GEE)
        session.layers[f"AlphaEarth 변화패치 {year_from}→{year_to}"] = json.loads(pts.to_json())
    return {"data": {"region": reg["name"], "years": [year_from, year_to], "threshold_cos_dist": round(thr, 4), **res},
            "meta": _meta("AlphaEarth 연간 임베딩(GEE) × ESA WorldCover 2021 수목마스크", f"{year_to}",
                          warnings=["산림 마스크가 WorldCover 2021 (10 m) 기준 — 임상도 산림경계와 차이 있음",
                                    None if mdl else "학습모형 없음: 기본 임계값 0.2 사용"])}


def run_change_model(session: Session, region_id: str, spacing_m: int = 100, max_points: int = 1500) -> dict:
    """학습된 융합모형(AE+S2+TERR)을 격자 표본점에 적용해 변화면적을 추정하고 임상도 라벨과 교차비교."""
    from . import features, gee
    mdl = _model()
    if mdl is None:
        raise RuntimeError("학습모형 없음: scripts/04_train_evaluate.py 먼저 실행")
    reg = _region(region_id)
    cp = _clip(change_polygons(), reg.geometry)
    dom = cp[cp["FRTP_2019"].isin(["1", "2", "3"]) | cp["FRTP_2025"].isin(["1", "2", "3"])]
    dom_ha = float(dom.area.sum() / 1e4)
    xmin, ymin, xmax, ymax = dom.total_bounds
    step = max(spacing_m, int(np.sqrt(dom_ha * 1e4 / max_points)) + 1)
    gx, gy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=C.CRS_WORK)
    pts = gpd.sjoin(pts, dom[["change_type", "geometry"]], predicate="within").drop(columns="index_right")
    pts = pts[~pts.index.duplicated()].head(max_points).reset_index(drop=True)
    ll = pts.to_crs(C.CRS_GEE).geometry
    pts["pid"], pts["lon"], pts["lat"] = [f"G{i}" for i in range(len(pts))], ll.x, ll.y
    raw = gee.extract_points(pts[["pid", "lon", "lat"]], id_col="pid", verbose=False)
    df = pts.merge(raw, on="pid")
    X, _ = features.build(df)
    p = mdl["change"].predict_proba(X[mdl["features"]])[:, 1]
    df["p_change"], df["pred_type"] = p, mdl["type"].predict(X[mdl["features"]])
    chg = p >= 0.5
    rate = float(chg.mean())
    se = float(np.sqrt(rate * (1 - rate) / len(df)))
    by_type = {t: round(dom_ha * float(((df.pred_type == t) & chg).mean()), 1) for t in mdl["type_classes"]}
    ref = df["change_type"].isin(["harvest", "conversion", "gain"])
    agree = float((chg == ref).mean())
    layer = df[chg].copy()
    layer = gpd.GeoDataFrame(layer[["pred_type", "p_change"]].round(3),
                             geometry=gpd.points_from_xy(layer.lon, layer.lat), crs=C.CRS_GEE)
    session.layers["모형 변화탐지 표본점"] = json.loads(layer.to_json())
    warns = [f"모형 적용범위: {mdl['years'][0]}→{mdl['years'][1]}, 학습지역 {mdl['trained_region']}",
             f"격자표본 {len(df)}점(간격 {step} m) 기반 추정 — 소규모 패치 과소추정 가능"]
    return {"data": {"region": reg["name"], "domain_forest_ha": round(dom_ha, 1), "n_points": int(len(df)),
                     "grid_spacing_m": int(step), "est_changed_ha": round(dom_ha * rate, 1),
                     "est_changed_ha_95ci": [round(dom_ha * max(rate - 1.96 * se, 0), 1),
                                             round(dom_ha * (rate + 1.96 * se), 1)],
                     "est_by_type_ha": by_type, "agreement_with_forest_map_labels": round(agree, 3),
                     "mean_p_change": round(float(p.mean()), 3)},
            "meta": _meta("학습모형(AlphaEarth+Sentinel-2+DEM, HistGBM) × GEE", f"{mdl['years'][1]}", warnings=warns)}


def hotspot_analysis(session: Session, region_id: str, change_types: list[str] | None = None,
                     cell_m: int = 500, band_m: int = 1500) -> dict:
    """변화면적 격자집계 + Getis-Ord Gi* (FDR 5%) 핫스팟."""
    reg = _region(region_id)
    types = change_types or ["harvest", "conversion"]
    cp = _clip(change_polygons(), reg.geometry)
    grid = spatial.grid_aggregate(cp[cp["change_type"].isin(types)], reg.geometry, cell_m, "changed_ha")
    xy = np.c_[grid.centroid.x, grid.centroid.y]
    z, pval = spatial.getis_ord_gi_star(grid["changed_ha"].to_numpy(), xy, band_m)
    sig = spatial.fdr_bh(pval)
    grid["gi_z"], grid["p"] = z.round(3), pval
    grid["class"] = np.where(sig & (z > 0), "hot", np.where(sig & (z < 0), "cold", "ns"))
    hot = grid[grid["class"] == "hot"].sort_values("changed_ha", ascending=False)
    cen = hot.centroid.to_crs(C.CRS_GEE)
    session.layers["Gi* 핫스팟 격자"] = _to_layer(hot.assign(changed_ha=hot.changed_ha.round(2)),
                                               ["cell_id", "changed_ha", "gi_z"])
    return {"data": {"region": reg["name"], "change_types": types, "cell_m": cell_m, "band_m": band_m,
                     "n_cells": int(len(grid)), "n_hot": int((grid["class"] == "hot").sum()),
                     "n_cold": int((grid["class"] == "cold").sum()),
                     "hot_changed_ha": round(float(hot["changed_ha"].sum()), 1),
                     "total_changed_ha": round(float(grid["changed_ha"].sum()), 1),
                     "top_hotspots": [{"cell_id": int(c), "changed_ha": round(a, 2), "gi_z": round(zz, 2),
                                       "lon": round(p.x, 5), "lat": round(p.y, 5)}
                                      for c, a, zz, p in zip(hot.cell_id, hot.changed_ha, hot.gi_z, cen)][:10]},
            "meta": _meta("임상도 비교 변화라벨 격자집계 / Getis-Ord Gi*(거리대역 이진가중, BH-FDR 0.05)",
                          f"{C.YEAR_TO}", warnings=[None if len(grid) >= 30 else "격자 수 30 미만 — 군집 통계 신뢰도 낮음"])}


DEFAULT_WEIGHTS = {"conversion_ha": 0.4, "harvest_ha": 0.3, "change_rate": 0.2, "stock_m3_per_ha": 0.1}


def rank_priority_units(session: Session, region_id: str, weights: dict | None = None,
                        top_n: int = 10, n_sim: int = 500) -> dict:
    """국유림 임반별 현장점검 우선순위 다기준평가(가중합) + 가중치 ±20% 몬테카를로 민감도."""
    reg = _region(region_id)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    w = {k: v for k, v in w.items() if k in DEFAULT_WEIGHTS and v > 0}
    s = sum(w.values())
    w = {k: v / s for k, v in w.items()}

    mp = load_mgmt_plan()
    mp = mp.iloc[mp.sindex.query(reg.geometry, predicate="intersects")]
    mp = mp[mp.representative_point().within(reg.geometry)]
    if mp.empty:
        raise ValueError("해당 지역에 국유림 임반 없음")
    cmp_ = mp.dissolve("cmblc_id", aggfunc={"area_ha": "sum", "stock_m3": "sum"}).reset_index()
    cp = _clip(change_polygons(), reg.geometry)
    inter = gpd.overlay(cp[cp["change_type"].isin(["harvest", "conversion"])][["change_type", "geometry"]],
                        cmp_[["cmblc_id", "geometry"]], how="intersection", keep_geom_type=True)
    inter["ha"] = inter.area / 1e4
    piv = inter.pivot_table(index="cmblc_id", columns="change_type", values="ha", aggfunc="sum").fillna(0)
    t = cmp_.set_index("cmblc_id")[["area_ha", "stock_m3"]].apply(pd.to_numeric, errors="coerce")
    t["harvest_ha"] = piv.get("harvest", 0.0)
    t["conversion_ha"] = piv.get("conversion", 0.0)
    t = t.fillna(0.0)
    t["change_rate"] = (t.harvest_ha + t.conversion_ha) / t.area_ha.replace(0, np.nan)
    t["stock_m3_per_ha"] = t.stock_m3 / t.area_ha.replace(0, np.nan)
    t = t.fillna(0.0)

    crit = list(w)
    norm = (t[crit] - t[crit].min()) / (t[crit].max() - t[crit].min()).replace(0, 1)
    wv = np.array([w[c] for c in crit])
    t["score"] = norm.to_numpy() @ wv
    t["rank"] = t["score"].rank(ascending=False, method="min").astype(int)
    top = t.sort_values("rank").head(top_n)

    rng = np.random.default_rng(C.SEED)
    k = min(top_n, len(t))
    in_top = np.zeros(len(t))
    for _ in range(n_sim):
        ws = wv * rng.uniform(0.8, 1.2, len(wv))
        sc = norm.to_numpy() @ (ws / ws.sum())
        in_top[np.argsort(-sc)[:k]] += 1
    t["top_stability"] = in_top / n_sim
    top = top.assign(top_stability=t.loc[top.index, "top_stability"])

    rows = [{"compartment": cid, "rank": int(r["rank"]), "score": round(r["score"], 3),
             "harvest_ha": round(r["harvest_ha"], 2), "conversion_ha": round(r["conversion_ha"], 2),
             "change_rate": round(r["change_rate"], 4), "stock_m3_per_ha": round(r["stock_m3_per_ha"], 1),
             "top_stability": round(r["top_stability"], 2)} for cid, r in top.iterrows()]
    return {"data": {"region": reg["name"], "n_units": int(len(t)), "weights": {k: round(v, 3) for k, v in w.items()},
                     "method": "min-max 정규화 가중합(WSM)", "sensitivity": f"가중치 ±20% 균등 난수 {n_sim}회",
                     "ranking": rows},
            "meta": _meta("경영계획부 임반 × 임상도 비교 변화라벨", f"{C.YEAR_TO}", unit="ha, m³/ha",
                          warnings=["가중치는 예시 기본값 — 정책목적에 맞게 담당자가 확정해야 함" if not weights else None])}


def make_decision_card(session: Session, title: str, summary: str, findings: list[str],
                       recommendations: list[str], caveats: list[str]) -> dict:
    """의사결정 카드(Markdown) + 지도(HTML) 생성. 문장 속 수치는 도구 출력과 대조 검증."""
    text = "\n".join([summary, *findings, *recommendations])
    g = verify.grounding(text, session.evidence)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    card_dir = C.OUT / "cards"
    card_dir.mkdir(exist_ok=True)
    md_path, map_path = card_dir / f"card_{stamp}.md", card_dir / f"card_{stamp}_map.html"

    spec = session.spec
    warn_all = sorted({w for a in session.audit for w in a["warnings"]})
    md = [f"# {title}", "", f"> 상태: **초안 — 담당자 검토 필요** · 생성 {time.strftime('%Y-%m-%d %H:%M')}", "",
          "## 1. 질의 및 해석된 분석조건", f"- 질의: {session.query}",
          f"- 지역: {spec.get('region_candidates')}", f"- 기간: {spec.get('period')}",
          f"- 변화유형: {spec.get('change_types')}", f"- 정책목적: {spec.get('policy_purpose')}", "",
          "## 2. 요약", summary, "", "## 3. 핵심 결과", *[f"- {x}" for x in findings], "",
          "## 4. 권고 / 대안", *[f"- {x}" for x in recommendations], "",
          "## 5. 검토사항 (자료 한계·경고)", *[f"- {x}" for x in caveats], *[f"- [도구경고] {w}" for w in warn_all], "",
          "## 6. 근거 검증", f"- 수치 {g['numbers_checked']}개 중 근거 일치율 **{g['grounded_ratio']:.0%}**",
          f"- 도구 출력과 불일치 수치: {g['unmatched'] or '없음'}", "",
          "## 7. 분석이력", "| # | 도구 | 입력 | 출력해시 | 시각 |", "|---|---|---|---|---|",
          *[f"| {a['step']} | {a['tool']} | `{json.dumps(a['input'], ensure_ascii=False)[:120]}` | {a['output_sha1']} | {a['ts']} |"
            for a in session.audit],
          "", f"지도: [{map_path.name}]({map_path.name})"]
    md_path.write_text("\n".join(md), encoding="utf-8")

    try:
        import folium
        m = folium.Map(location=[37.88, 127.73], zoom_start=11, tiles="CartoDB positron")
        colors = {"harvest": "#d95f02", "conversion": "#7570b3", "gain": "#1b9e77", "hot": "#e7298a"}
        for name, gj in session.layers.items():
            folium.GeoJson(gj, name=name, style_function=lambda f: {
                "color": colors.get(f["properties"].get("change_type") or f["properties"].get("pred_type"), "#e7298a"),
                "weight": 1, "fillOpacity": 0.5}, marker=folium.CircleMarker(radius=3)).add_to(m)
        folium.LayerControl().add_to(m)
        if session.layers:
            m.fit_bounds(m.get_bounds())
        m.save(str(map_path))
    except Exception as e:           # 지도 실패는 카드 생성을 막지 않음
        map_path = None
        warn_all.append(f"지도 생성 실패: {e}")

    return {"data": {"card_path": str(md_path), "map_path": str(map_path) if map_path else None,
                     "grounding": g, "review_status": "초안(담당자 검토 필요)"},
            "meta": _meta("의사결정 카드", time.strftime("%Y-%m-%d"), unit="-",
                          warnings=[f"근거 불일치 수치 {g['unmatched']} — 수정 후 재생성 권장" if g["unmatched"] else None])}


# ================================================================== 도구 스키마 (Claude tool use)

def _s(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


REGION = {"type": "string", "description": "set_analysis_spec 또는 list_regions 가 반환한 region_id (예: 51110, nf:51110, emd:북산면, cmp:067-000)"}
TYPES = {"type": "array", "items": {"type": "string", "enum": CHANGE_KINDS}}

TOOLS = {
    "set_analysis_spec": (set_analysis_spec, "1단계 질의 해석. 사용자 질의에서 지역·기간·변화유형·정책목적을 추출해 등록하고, 지역명을 region_id 로 해석·검증한다. 모든 분석 전에 반드시 먼저 호출.",
        _s({"region": {"type": "string", "description": "지역명 원문 (예: '북산면', '춘천시', '67임반')"},
            "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
            "change_types": TYPES, "policy_purpose": {"type": "string"},
            "outputs": {"type": "array", "items": {"type": "string"}}},
           ["region", "year_from", "year_to", "change_types", "policy_purpose"])),
    "list_regions": (list_regions, "분석 가능한 지역 목록 (시 전체 / 국유림 전체 / 읍면 국유림 / 임반).",
        _s({"level": {"type": "string", "enum": ["all", "city", "national_forest", "emd_national_forest", "compartment"]}}, [])),
    "query_forest_status": (query_forest_status, "2단계 데이터검색. 2025 임상도 기준 산림 현황(임상·영급·수종군 면적).",
        _s({"region_id": REGION}, ["region_id"])),
    "query_change_map": (query_change_map, "2단계 데이터검색. 임상도 2019→2025 비교로 구축된 변화 폴리곤 DB 조회: 유형별 면적·패치수·갱신년도·대표 패치. 가장 먼저 조회할 변화 정보.",
        _s({"region_id": REGION, "change_types": TYPES, "top_n": {"type": "integer"}}, ["region_id"])),
    "query_mgmt_plan": (query_mgmt_plan, "2단계 데이터검색. 국유림 경영계획부(소반) 면적·축적 현황.",
        _s({"region_id": REGION}, ["region_id"])),
    "scan_alphaearth_change": (scan_alphaearth_change, "3단계 원격탐사. GEE에서 AlphaEarth 연간 임베딩 코사인거리로 임의 연도쌍(2017~2025) 변화를 즉석 스캔. 임상도에 없는 기간·최신 변화 확인용.",
        _s({"region_id": REGION, "year_from": {"type": "integer"}, "year_to": {"type": "integer"}},
           ["region_id", "year_from", "year_to"])),
    "run_change_model": (run_change_model, "3단계 원격탐사. 학습된 융합 변화탐지모형을 격자 표본점에 적용해 변화면적(95% CI)·유형을 추정하고 임상도 라벨과 일치도를 계산. 2019→2025 에만 적용 가능. 수십 초 소요.",
        _s({"region_id": REGION, "spacing_m": {"type": "integer"}, "max_points": {"type": "integer"}}, ["region_id"])),
    "hotspot_analysis": (hotspot_analysis, "4단계 GIS 분석. 변화면적 격자집계 후 Getis-Ord Gi* 핫스팟(FDR 5%).",
        _s({"region_id": REGION, "change_types": TYPES, "cell_m": {"type": "integer"}, "band_m": {"type": "integer"}},
           ["region_id"])),
    "rank_priority_units": (rank_priority_units, "5·6단계 대안 평가. 국유림 임반별 현장점검 우선순위를 명시적 가중치로 다기준평가하고 가중치 민감도(순위 안정성)를 제시.",
        _s({"region_id": REGION, "weights": {"type": "object", "description": f"기준별 가중치. 키: {list(DEFAULT_WEIGHTS)}",
                                             "properties": {k: {"type": "number"} for k in DEFAULT_WEIGHTS},
                                             "additionalProperties": False},
            "top_n": {"type": "integer"}}, ["region_id"])),
    "make_decision_card": (make_decision_card, "7단계 결과 제공. 의사결정 카드와 지도를 생성하고 문장 속 수치를 도구 출력과 대조 검증. 수치는 반드시 도구 출력값 그대로 인용.",
        _s({"title": {"type": "string"}, "summary": {"type": "string"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "caveats": {"type": "array", "items": {"type": "string"}}},
           ["title", "summary", "findings", "recommendations", "caveats"])),
}

TOOL_SCHEMAS = [{"name": n, "description": d, "input_schema": s} for n, (_, d, s) in TOOLS.items()]


def call(session: Session, name: str, args: dict) -> dict:
    fn = TOOLS[name][0]
    result = fn(session, **args)
    session.log(name, args, result)
    return result
