"""E10 인제군 전이 시험 — Colab 노트북(셀 1~6)을 한 파일로 옮긴 것.

순서: inje_prepare.py(로컬: 라벨·표본) → 표본을 GEE 자산으로 올림(projects/<project>/assets/inje_transfer)
     → 이 파일(Colab 또는 로컬, GEE 인증 필요) → results/E10_inje_transfer.json, inje/visual_check_points_raw.csv
  python paper/revision/inje_transfer_colab.py [--project ee-choish7838]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

AE_YEARS = list(range(2017, 2026))
AE_BANDS = [f"A{i:02d}" for i in range(64)]
Y0, Y1, NODATA, SEED = 2019, 2025, -999.0, 42
CLASSES = ["no_change", "harvest", "conversion", "gain"]
TYPES = CLASSES[1:]


# ---------------------------------------------------------------- 특징 (forest-change-agent/fcd/gee.py·features.py 와 동일)
def ae_year(y):
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filter(ee.Filter.calendarRange(y, y, "year")).mosaic().select(AE_BANDS))


def s2_summer(y):
    def mask(img):
        scl = img.select("SCL")
        return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    med = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(f"{y}-06-01", f"{y}-09-30")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)).map(mask).median())
    return (med.normalizedDifference(["B8", "B4"]).rename(f"ndvi{y}")
            .addBands(med.normalizedDifference(["B8", "B12"]).rename(f"nbr{y}")))


def feature_image():
    img = ee.Image.cat([ae_year(y).rename([f"ae{y}_{b}" for b in AE_BANDS]) for y in AE_YEARS])
    for y in (Y0, Y1):
        img = img.addBands(s2_summer(y))
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    return img.addBands(dem.rename("elev")).addBands(ee.Terrain.slope(dem).rename("slope")).unmask(NODATA)


def extract(img, points, chunk=150, workers=6):
    def one(c):
        feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"sample_id": r.sample_id}) for r in c.itertuples()]
        for k in range(4):
            try:
                fc = img.sampleRegions(collection=ee.FeatureCollection(feats), properties=["sample_id"],
                                       scale=10, tileScale=4, geometries=False)
                return [f["properties"] for f in fc.getInfo()["features"]]
            except Exception:
                if k == 3:
                    raise
                time.sleep(5 * (k + 1))
    parts = [points.iloc[i:i + chunk] for i in range(0, len(points), chunk)]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for res in ex.map(one, parts) for r in res]
    df = pd.DataFrame(rows)
    num = df.columns.drop("sample_id")
    df[num] = df[num].astype("float32").replace(NODATA, np.nan)
    return df


def _ae(df, y):
    e = df[[f"ae{y}_{b}" for b in AE_BANDS]].to_numpy("float32")
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-6, None)


def build(df):
    F = {}
    e0, e1 = _ae(df, Y0), _ae(df, Y1)
    for i, b in enumerate(AE_BANDS):
        F[f"ae_diff_{b}"] = e1[:, i] - e0[:, i]
    F["ae_cos_dist"] = 1 - (e0 * e1).sum(1)
    embs = {y: _ae(df, y) for y in AE_YEARS}
    consec = {y: 1 - (embs[y - 1] * embs[y]).sum(1) for y in AE_YEARS[1:]}
    for y, d in consec.items():
        F[f"ae_d{y}"] = d
    win = [y for y in consec if Y0 < y <= Y1]
    pre = [y for y in consec if y <= Y0]
    W = np.stack([consec[y] for y in win], 1)
    F["ae_consec_max"], F["ae_peak_year"] = W.max(1), np.array(win)[W.argmax(1)]
    F["ae_pre_noise"] = np.stack([consec[y] for y in pre], 1).max(1)
    for idx in ("ndvi", "nbr"):
        F[f"{idx}_{Y0}"], F[f"{idx}_{Y1}"] = df[f"{idx}{Y0}"].to_numpy(), df[f"{idx}{Y1}"].to_numpy()
        F[f"d{idx}"] = F[f"{idx}_{Y1}"] - F[f"{idx}_{Y0}"]
    F["elev"], F["slope"] = df["elev"].to_numpy(), df["slope"].to_numpy()
    X = pd.DataFrame(F, index=df.index)
    G = {"AE": [c for c in X if c.startswith("ae_") and c != "ae_peak_year"],
         "S2": [c for c in X if c.startswith(("ndvi", "nbr", "dndvi", "dnbr"))], "TERR": ["elev", "slope"]}
    return X, G


def read_asset(root, name):
    fc = ee.FeatureCollection(f"{root}/{name}")
    n = fc.size().getInfo()
    rows = []
    for off in range(0, n, 5000):
        for f in fc.toList(5000, off).getInfo():
            p = f["properties"]
            p["lon"], p["lat"] = f["geometry"]["coordinates"]
            rows.append(p)
    return pd.DataFrame(rows)


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=SEED)


def cv_f1(X, y, g, average="binary"):
    s = [f1_score(y.iloc[b], model().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]), average=average)
         for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g)]
    return float(np.mean(s)), float(np.std(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ee-choish7838")
    ap.add_argument("--out", default="paper/revision/results/E10_inje_transfer_rerun.json")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    root = f"projects/{a.project}/assets/inje_transfer"
    img = feature_image()
    feat = {k: (p := read_asset(root, k)).merge(extract(img, p), on="sample_id")
            for k in ("chuncheon_samples", "inje_samples", "inje_events")}
    ch, ij = feat["chuncheon_samples"], feat["inje_samples"]
    Xc, G = build(ch)
    Xi, _ = build(ij)
    yc, yi = (ch["cls"] > 0).astype(int), (ij["cls"] > 0).astype(int)
    tr, te = ch["split"].eq("train"), ch["split"].eq("test_spatial")
    combos = {"AE": G["AE"], "S2": G["S2"], "TERR": G["TERR"], "AE+S2": G["AE"] + G["S2"],
              "AE+S2+TERR": G["AE"] + G["S2"] + G["TERR"]}

    # T1 전이 (재학습 없음) + 인제 자체 공간블록 CV
    t1 = []
    for name, cols in combos.items():
        m = model().fit(Xc.loc[tr, cols], yc[tr])
        p = m.predict(Xi[cols])
        cvm, cvs = cv_f1(Xi[cols], yi, ij["block_id"])
        t1.append({"features": name, "n_feat": len(cols), "chuncheon_holdout_F1": round(f1_score(yc[te], m.predict(Xc.loc[te, cols])), 3),
                   "inje_transfer_F1": round(f1_score(yi, p), 3), "P": round(precision_score(yi, p), 3),
                   "R": round(recall_score(yi, p), 3), "inje_local_CV_F1": f"{cvm:.3f} ± {cvs:.3f}"})
    print(pd.DataFrame(t1).to_string(index=False))

    # T2 변화유형
    cols = combos["AE+S2+TERR"]
    cc, ci = ch["cls"] > 0, ij["cls"] > 0
    lab = dict(enumerate(CLASSES))
    yt_c, yt_i = ch["cls"].map(lab), ij["cls"].map(lab)
    mt = model().fit(Xc.loc[tr & cc, cols], yt_c[tr & cc])
    pt_i = mt.predict(Xi.loc[ci, cols])
    t2 = {"inje_transfer_macroF1": round(f1_score(yt_i[ci], pt_i, average="macro"), 3),
          "inje_transfer_classF1": dict(zip(TYPES, f1_score(yt_i[ci], pt_i, labels=TYPES, average=None).round(3).tolist())),
          "inje_local_CV_macroF1": "%.3f ± %.3f" % cv_f1(Xi.loc[ci, cols], yt_i[ci], ij.loc[ci, "block_id"], "macro")}

    # 인제 사업실적 시점: 소반 평균 연속연도 AE 거리(2018~2025) 정점 − 사업 연도
    ev = feat["inje_events"]
    emb = {y: _ae(ev, y) for y in AE_YEARS}
    D = pd.DataFrame({y: 1 - (emb[y - 1] * emb[y]).sum(1) for y in range(2018, 2026)})
    D["unit"] = ev["unit"].values
    info = ev.groupby("unit")[["year", "grade"]].first().join(D.groupby("unit").mean().idxmax(axis=1).astype(int).rename("ae_peak"))
    info["lag"] = info["ae_peak"] - info["year"]
    events = {g: {"n": len(d), "exact": round(float((d.lag == 0).mean()), 3), "within1": round(float((d.lag.abs() <= 1).mean()), 3)}
              for g, d in (("all", info), ("A", info[info.grade == "A"]), ("B", info[info.grade == "B"]))}

    # 육안 판독 표본: 춘천 AE+S2 모델 예측 유형별 25점
    cols2 = combos["AE+S2"]
    prob = model().fit(Xc.loc[tr, cols2], yc[tr]).predict_proba(Xi[cols2])[:, 1]
    typ = model().fit(Xc.loc[tr & cc, cols2], yt_c[tr & cc]).predict(Xi[cols2])
    pred = np.where(prob >= 0.5, typ, "no_change")
    rng = np.random.default_rng(SEED)
    pick = [i for c in CLASSES for i in rng.choice(np.flatnonzero(pred == c), size=min(25, int((pred == c).sum())), replace=False)]
    vc = ij.iloc[pick][["sample_id", "lon", "lat", "block_id"]].assign(
        map_label=ij["cls"].iloc[pick].map(lab).values, pred_type=pred[pick], p_change=prob[pick].round(3))
    vc.to_csv(a.out.replace(".json", "_visual_points.csv"), index=False)

    json.dump({"T1": t1, "T2": t2, "events": events}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
