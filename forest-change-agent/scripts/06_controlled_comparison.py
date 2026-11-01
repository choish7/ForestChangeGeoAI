# [통합 사본] 원본: paper/revision/revision_experiments.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""리뷰 대응 보완실험 (원 논문과 같은 표본 6,027개 · 같은 5 km 공간블록 분할 · 같은 분류기 설정).

E1 비교조건 통제 : AE 두 시점 / AE 2019~2025 연간 / AE 원논문(2017~2025) / S2 두 시점 / S2 2019~2025 연간 / S2 연간+S2-TSFM
                  → 공통 평가 표본에서 T1(변화 여부) F1, T2(유형) macro-F1, 공간 CV + 독립 블록 + 블록 부트스트랩 95% CI
E2 폴리곤 중복   : 같은 원본 폴리곤(patch_id)이 학습·검증에 함께 있는지, 중복 표본 제외 후 재평가
E3 CV < 홀드아웃 : 폴드별·홀드아웃 클래스 구성과 F1
E4 미탐지 유형   : 최종 모형(AE+S2+TERR) 독립 검증 FN 의 원래 변화유형별 수·비율
E5 변화 시점     : (a) 임상도 갱신년도별 분해(2025 집단 분리)
                   (b) 독립 사건: 국유림 수확·갱신 실적 연도 vs 소반 AE 최대 변화연도·S2 NDVI 최대감소 연도
                   (c) 보전산지 해제 고시(전용확정) 고시연도 vs AE 최대 변화연도
E6 제외 지역     : 임상변화·판정보류 조각의 AE 2019→2025 거리·최종지도 변화비율을 무변화·벌채와 비교
E7 우선순위      : 변화면적 단독 순위 vs 복합(WSM) 순위 상위 10 비교
출력: forest-change-agent/outputs/results/*.json, *.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
FCA = ROOT / "forest-change-agent"
sys.path.insert(0, str(FCA))
from fcd import config as C  # noqa: E402
from fcd import features  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "results"
OUT.mkdir(exist_ok=True)
S2_42 = ROOT / "output" / "Chuncheon_S2_2019_2025_42Bands.tif"
TSFM_SCORES = FCA / "models" / "s2_tsfm_chuncheon" / "change_scores.tif"   # copy of ../s2_tsfm_chuncheon
AE_FILE = ROOT / "output" / "Chuncheon_AE_{}_64Bands_int8.tif"
FM_CLASS = ROOT / "final_change_2019_2025" / "final_change_class.tif"
TYPES = ["harvest", "conversion", "gain"]
KO = {"no_change": "무변화", "harvest": "벌채·갱신", "conversion": "산지전용", "gain": "산림증가"}


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=C.SEED)


def sample_raster(path, lon, lat, bands=None):
    from pyproj import Transformer
    with rasterio.open(path) as ds:
        x, y = Transformer.from_crs(4326, ds.crs, always_xy=True).transform(lon, lat)
        v = np.array(list(ds.sample(zip(x, y), indexes=bands)), dtype="float32")
        nod = ds.nodata
        desc = [ds.descriptions[b - 1] for b in (bands or range(1, ds.count + 1))]
    if nod is not None:
        v[v == nod] = np.nan
    return pd.DataFrame(v, columns=desc)


def unit(e):
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-6, None)


def build_sets(df):
    X0, G = features.build(df)                       # 원 논문 특징 (AE 75, S2 6, TERR 2)
    F = {}
    e = {y: unit(df[[f"ae{y}_{b}" for b in C.AE_BANDS]].to_numpy("float32")) for y in C.AE_YEARS}
    diff = pd.DataFrame(e[2025] - e[2019], columns=[f"aed_{b}" for b in C.AE_BANDS], index=df.index)
    cos = pd.Series(1 - (e[2019] * e[2025]).sum(1), index=df.index, name="ae_cos")
    cons = pd.DataFrame({f"aec{y}": 1 - (e[y - 1] * e[y]).sum(1) for y in range(2020, 2026)}, index=df.index)
    F["AE_2시점(2019·2025)"] = pd.concat([diff, cos], axis=1)
    F["AE_연간(2019~2025)"] = pd.concat([diff, cos, cons, cons.max(1).rename("aec_max")], axis=1)
    F["AE_원논문(2017~2025)"] = X0[G["AE"]]
    # Sentinel-2: 원 논문 GEE 지수(2시점) + 로컬 연간 합성(같은 4지수, 2019~2025)
    F["S2_원논문 2시점(NDVI·NBR)"] = X0[G["S2"]]
    s2 = sample_raster(S2_42, df["lon"].values, df["lat"].values)
    s2.index = df.index
    idx = ["NDVI", "NDWI", "NDBI", "NDMI"]
    two = pd.concat([s2[[f"{i}_{y}" for i in idx for y in (2019, 2025)]],
                     pd.DataFrame({f"d{i}": s2[f"{i}_2025"] - s2[f"{i}_2019"] for i in idx})], axis=1)
    F["S2_2시점(4지수)"] = two
    ann = s2[[f"{i}_{y}" for i in idx for y in range(2019, 2026)]]
    summ = {}
    for i in idx:
        a = ann[[f"{i}_{y}" for y in range(2019, 2026)]].to_numpy()
        d = np.diff(a, axis=1)
        summ[f"{i}_maxdrop"] = -np.nanmin(d, 1)
        summ[f"{i}_maxrise"] = np.nanmax(d, 1)
        summ[f"{i}_std"] = np.nanstd(a, 1)
        summ[f"d{i}"] = a[:, -1] - a[:, 0]
    F["S2_연간(4지수, 2019~2025)"] = pd.concat([ann, pd.DataFrame(summ, index=df.index)], axis=1)
    mg = sample_raster(TSFM_SCORES, df["lon"].values, df["lat"].values)
    mg.index = df.index
    F["S2_연간+S2-TSFM 임베딩거리"] = pd.concat([F["S2_연간(4지수, 2019~2025)"], mg.add_prefix("tsfm_")], axis=1)
    F["AE_연간+S2_연간"] = pd.concat([F["AE_연간(2019~2025)"], F["S2_연간(4지수, 2019~2025)"]], axis=1)
    return F, X0, G


def block_ci(y, p, blocks, avg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    vals = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        m = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        if len(np.unique(y[m])) > 1:
            vals.append(f1_score(y[m], p[m], average=avg))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def evaluate(X, y, tr, te, groups, blocks_te, avg):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    sc = []
    Xt, yt = X[tr], y[tr]
    for a, b in cv.split(Xt, yt, groups[tr]):
        m = model().fit(Xt.iloc[a], yt.iloc[a])
        sc.append(f1_score(yt.iloc[b], m.predict(Xt.iloc[b]), average=avg))
    m = model().fit(Xt, yt)
    p = m.predict(X[te])
    return {"cv_mean": round(float(np.mean(sc)), 3), "cv_sd": round(float(np.std(sc)), 3), "cv_folds": [round(float(s), 3) for s in sc],
            "holdout": round(float(f1_score(y[te], p, average=avg)), 3),
            "holdout_95ci_block_bootstrap": block_ci(y[te].to_numpy(), p, blocks_te, avg)}, m, p


def main():
    df = pd.read_parquet(FCA / "work" / "features.parquet")
    df = df[df["change_type"].isin(["no_change", *TYPES])].reset_index(drop=True)
    F, X0, G = build_sets(df)
    ok = np.ones(len(df), bool)
    for X in F.values():
        ok &= X.notna().all(1).to_numpy() | ~X.isna().all(1).to_numpy()   # 화소 결측(전 열 NaN)만 제외
        ok &= ~X.isna().all(1).to_numpy()
    df, F = df[ok].reset_index(drop=True), {k: v[ok].reset_index(drop=True) for k, v in F.items()}
    X0 = X0[ok].reset_index(drop=True)
    tr, te = df["split"].eq("train").to_numpy(), df["split"].eq("test_spatial").to_numpy()
    groups, blocks_te = df["block_id"].to_numpy(), df.loc[te, "block_id"].to_numpy()
    res = {"공통평가표본": {"전체": int(len(df)), "학습": int(tr.sum()), "검증": int(te.sum()), "제외(결측)": int((~ok).sum())}}

    # ---------- E1 비교조건 통제
    y1 = df["changed"].astype(int)
    chg = df["change_type"].isin(TYPES).to_numpy()
    y2 = df["change_type"]
    rows = []
    for name, X in F.items():
        r1, _, _ = evaluate(X, y1, tr, te, groups, blocks_te, "binary")
        r2, _, _ = evaluate(X[chg].reset_index(drop=True), y2[chg].reset_index(drop=True), tr[chg], te[chg], groups[chg],
                            df.loc[te & chg, "block_id"].to_numpy(), "macro")
        rows.append({"특징": name, "특징수": X.shape[1], "T1_CV": f"{r1['cv_mean']:.3f} ± {r1['cv_sd']:.3f}", "T1_홀드아웃": r1["holdout"],
                     "T1_95CI": r1["holdout_95ci_block_bootstrap"], "T2_CV": f"{r2['cv_mean']:.3f} ± {r2['cv_sd']:.3f}",
                     "T2_홀드아웃": r2["holdout"], "T2_95CI": r2["holdout_95ci_block_bootstrap"], "T1_folds": r1["cv_folds"]})
        print(f"[E1] {name:<28} n={X.shape[1]:>3} T1 CV {r1['cv_mean']:.3f}±{r1['cv_sd']:.3f} HO {r1['holdout']:.3f} {r1['holdout_95ci_block_bootstrap']} | "
              f"T2 CV {r2['cv_mean']:.3f} HO {r2['holdout']:.3f}", flush=True)
    res["E1_비교조건통제"] = rows
    pd.DataFrame(rows).to_csv(OUT / "E1_fair_comparison.csv", index=False, encoding="utf-8-sig")

    # ---------- E2 폴리곤 중복
    p_tr, p_te = set(df.loc[tr, "patch_id"]), set(df.loc[te, "patch_id"])
    shared = p_tr & p_te
    te_shared = te & df["patch_id"].isin(shared).to_numpy()
    allX = pd.concat([X0[G["AE"] + G["S2"] + G["TERR"]]], axis=1)
    m_all = model().fit(allX[tr], y1[tr])
    p_all = m_all.predict(allX[te])
    keep = ~df.loc[te, "patch_id"].isin(shared).to_numpy()
    res["E2_폴리곤중복"] = {"원본폴리곤_학습": len(p_tr), "원본폴리곤_검증": len(p_te), "학습·검증_공유_폴리곤": len(shared),
                         "공유폴리곤의_검증표본": int(te_shared.sum()),
                         "표본/폴리곤(학습)": round(tr.sum() / len(p_tr), 2), "표본/폴리곤(검증)": round(te.sum() / len(p_te), 2),
                         "AE+S2+TERR_홀드아웃F1_전체": round(float(f1_score(y1[te], p_all)), 3),
                         "AE+S2+TERR_홀드아웃F1_공유제외": round(float(f1_score(y1[te][keep], p_all[keep])), 3) if keep.any() else None}
    print("[E2]", res["E2_폴리곤중복"], flush=True)

    # ---------- E3 폴드·홀드아웃 구성
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    comp = []
    Xa = F["AE_원논문(2017~2025)"]
    for k, (a, b) in enumerate(cv.split(Xa[tr], y1[tr], groups[tr])):
        sub = df[tr].iloc[b]
        m = model().fit(Xa[tr].iloc[a], y1[tr].iloc[a])
        comp.append({"집합": f"CV폴드{k + 1}", "블록": int(sub["block_id"].nunique()), "표본": len(sub),
                     **{KO[c]: round(float((sub["change_type"] == c).mean()), 3) for c in KO},
                     "AE_F1": round(float(f1_score(y1[tr].iloc[b], m.predict(Xa[tr].iloc[b]))), 3)})
    sub = df[te]
    mA = model().fit(Xa[tr], y1[tr])
    comp.append({"집합": "독립검증", "블록": int(sub["block_id"].nunique()), "표본": len(sub),
                 **{KO[c]: round(float((sub["change_type"] == c).mean()), 3) for c in KO},
                 "AE_F1": round(float(f1_score(y1[te], mA.predict(Xa[te]))), 3)})
    res["E3_폴드구성"] = comp
    pd.DataFrame(comp).to_csv(OUT / "E3_fold_composition.csv", index=False, encoding="utf-8-sig")
    print("[E3]", pd.DataFrame(comp).to_string(index=False), flush=True)

    # ---------- E4 미탐지 유형별
    fn = []
    t = df.loc[te, "change_type"].to_numpy()
    for c in TYPES:
        s = t == c
        fn.append({"변화유형": KO[c], "검증표본": int(s.sum()), "미탐지(FN)": int((p_all[s] == 0).sum()),
                   "미탐지율": round(float((p_all[s] == 0).mean()), 3)})
    fn.append({"변화유형": "합계", "검증표본": int(np.isin(t, TYPES).sum()), "미탐지(FN)": int(((p_all == 0) & np.isin(t, TYPES)).sum()),
               "미탐지율": round(float((p_all[np.isin(t, TYPES)] == 0).mean()), 3)})
    fp = int(((p_all == 1) & (t == "no_change")).sum())
    res["E4_미탐지유형"] = {"FN": fn, "오탐(FP)": fp}
    print("[E4]", fn, "FP", fp, flush=True)

    # ---------- E5a 갱신년도별 분해
    peak = X0["ae_peak_year"]
    yl = pd.to_numeric(df["change_year"], errors="coerce")
    has = df["change_type"].isin(TYPES) & yl.notna()
    t5 = []
    for name, sel in [("전체", has), ("갱신 2025", has & yl.eq(2025)), ("갱신 2020~2024", has & yl.between(2020, 2024))]:
        lag = (yl[sel] - peak[sel])
        t5.append({"집단": name, "표본": int(sel.sum()), "정확일치": round(float((lag == 0).mean()), 3),
                   "±1년": round(float((lag.abs() <= 1).mean()), 3), "정점≤갱신": round(float((lag >= 0).mean()), 3),
                   "정점>갱신(갱신 뒤 변화)": round(float((lag < 0).mean()), 3)})
    res["E5a_갱신년도별"] = t5
    print("[E5a]", t5, flush=True)

    # ---------- E6 제외 지역
    import geopandas as gpd
    cp = gpd.read_parquet(FCA / "work" / "change_polygons.parquet", columns=["change_type", "area_ha", "geometry"])
    rng = np.random.default_rng(C.SEED)
    pts = []
    for c in ["type_shift", "ambiguous", "no_change", "harvest", "conversion", "gain"]:
        g = cp[cp["change_type"].eq(c)]
        g = g.sample(min(800, len(g)), weights=g["area_ha"], random_state=C.SEED)
        rp = g.geometry.representative_point().to_crs(4326)
        pts.append(pd.DataFrame({"change_type": c, "lon": rp.x.values, "lat": rp.y.values}))
    pts = pd.concat(pts, ignore_index=True)
    e19 = sample_raster(Path(str(AE_FILE).format(2019)), pts.lon.values, pts.lat.values).to_numpy()
    e25 = sample_raster(Path(str(AE_FILE).format(2025)), pts.lon.values, pts.lat.values).to_numpy()
    okp = ~((e19 == -128).all(1) | (e19 == 0).all(1) | np.isnan(e19).any(1))
    d = 1 - (unit(e19 / 127) * unit(e25 / 127)).sum(1)
    fm = sample_raster(FM_CLASS, pts.lon.values, pts.lat.values).iloc[:, 0].to_numpy()
    pts = pts.assign(ae_dist=np.where(okp, d, np.nan), fm=fm)
    thr = 0.0665
    e6 = []
    for c, g in pts.groupby("change_type", sort=False):
        e6.append({"클래스": {"type_shift": "임상변화(제외)", "ambiguous": "판정보류(제외)", **KO}.get(c, c), "표본점": int(g["ae_dist"].notna().sum()),
                   "AE거리_중앙값": round(float(g["ae_dist"].median()), 3), f"AE거리>{thr}": round(float((g["ae_dist"] > thr).mean()), 3),
                   "최종지도_변화(1~3)": round(float(g["fm"].isin([1, 2, 3]).mean()), 3), "최종지도_산림밖": round(float((g["fm"] == 255).mean()), 3)})
    res["E6_제외지역"] = e6
    print("[E6]", pd.DataFrame(e6).to_string(index=False), flush=True)

    # ---------- E7 변화면적 순위 vs 복합 순위 (원 논문 국유림 112 임반)
    from fcd import tools as FT
    from fcd.data import change_polygons, load_mgmt_plan
    reg = FT._region("nf:51110")
    mp = load_mgmt_plan()
    mp = mp.iloc[mp.sindex.query(reg.geometry, predicate="intersects")]
    mp = mp[mp.representative_point().within(reg.geometry)]
    cmp_ = mp.dissolve("cmblc_id", aggfunc={"area_ha": "sum", "stock_m3": "sum"}).reset_index()
    cpp = FT._clip(change_polygons(), reg.geometry)
    inter = gpd.overlay(cpp[cpp["change_type"].isin(["harvest", "conversion"])][["change_type", "geometry"]],
                        cmp_[["cmblc_id", "geometry"]], how="intersection", keep_geom_type=True)
    inter["ha"] = inter.area / 1e4
    piv = inter.pivot_table(index="cmblc_id", columns="change_type", values="ha", aggfunc="sum").fillna(0)
    tt = cmp_.set_index("cmblc_id")[["area_ha", "stock_m3"]].apply(pd.to_numeric, errors="coerce")
    tt["harvest_ha"], tt["conversion_ha"] = piv.get("harvest", 0.0), piv.get("conversion", 0.0)
    tt = tt.fillna(0.0)
    tt["change_rate"] = (tt.harvest_ha + tt.conversion_ha) / tt.area_ha.replace(0, np.nan)
    tt["stock_m3_per_ha"] = tt.stock_m3 / tt.area_ha.replace(0, np.nan)
    tt = tt.fillna(0.0)
    w = FT.DEFAULT_WEIGHTS
    norm = (tt[list(w)] - tt[list(w)].min()) / (tt[list(w)].max() - tt[list(w)].min()).replace(0, 1)
    tt["복합점수"] = norm.to_numpy() @ np.array(list(w.values()))
    tt["변화면적_ha"] = tt.harvest_ha + tt.conversion_ha
    tt["복합순위"] = tt["복합점수"].rank(ascending=False, method="min").astype(int)
    tt["면적순위"] = tt["변화면적_ha"].rank(ascending=False, method="min").astype(int)
    top_c, top_a = set(tt.nsmallest(10, "복합순위").index), set(tt.nsmallest(10, "면적순위").index)
    only_c = tt.loc[sorted(top_c - top_a)][["복합순위", "면적순위", "harvest_ha", "conversion_ha", "change_rate", "stock_m3_per_ha"]]
    only_a = tt.loc[sorted(top_a - top_c)][["복합순위", "면적순위", "harvest_ha", "conversion_ha", "change_rate", "stock_m3_per_ha"]]
    from scipy.stats import spearmanr
    res["E7_순위비교"] = {"임반": int(len(tt)), "상위10_공통": len(top_c & top_a),
                       "spearman_rho": round(float(spearmanr(tt["복합점수"], tt["변화면적_ha"]).statistic), 3),
                       "복합순위에만_있는_임반": json.loads(only_c.round(3).reset_index().to_json(orient="records", force_ascii=False)),
                       "면적순위에만_있는_임반": json.loads(only_a.round(3).reset_index().to_json(orient="records", force_ascii=False))}
    print("[E7]", json.dumps(res["E7_순위비교"], ensure_ascii=False)[:800], flush=True)

    (OUT / "revision_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
