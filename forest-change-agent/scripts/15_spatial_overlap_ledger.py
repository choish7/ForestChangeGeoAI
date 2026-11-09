"""R2 (editorial query Q3): spatial-partition overlap ledger for the hold-out and each CV fold.

Identity types
  overlay fragment (patch_id): piece of the 2019 × 2025 forest-map intersection; labels and sampling are defined on it
  map polygon: original polygon of the 2019 or 2025 forest type map containing the sample point (ID = feature index)
CV folds are those of the T1 binary task (StratifiedGroupKFold, 5 folds, groups = 5 km blocks, seed 42).
Needs the raw forest type maps (see data/README.md).
  python forest-change-agent/scripts/15_spatial_overlap_ledger.py
Outputs: outputs/results/R2_spatial_overlap.csv, outputs/results/R2_shared_ids.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from fcd import config as C
from fcd import features

SEED = 42
OUT = ROOT / "outputs" / "results"


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=SEED)


def main():
    smp = pd.read_parquet(C.SAMPLES)
    pts = gpd.GeoDataFrame(smp, geometry=gpd.GeoSeries.from_wkb(smp["geometry"]), crs=C.CRS_WORK)
    for y in (2019, 2025):
        m = gpd.read_file(C.FOREST_MAP[y], columns=["FRTP_CD"], engine="pyogrio").to_crs(C.CRS_WORK)
        m[f"poly{y}"] = [f"M{y}_{i:06d}" for i in m.index]
        j = gpd.sjoin(pts[["sample_id", "geometry"]], m[[f"poly{y}", "geometry"]], how="left", predicate="within")
        pts = pts.merge(j.drop_duplicates("sample_id")[["sample_id", f"poly{y}"]], on="sample_id", how="left")
    df = pd.read_parquet(C.FEATURES).reset_index(drop=True).merge(pts[["sample_id", "poly2019", "poly2025"]],
                                                                  on="sample_id", how="left")
    X, G = features.build(df)
    cols = G["AE"] + G["S2"] + G["TERR"]
    y = df["changed"].astype(int)
    tr, te = df["split"].eq("train").to_numpy(), df["split"].eq("test_spatial").to_numpy()
    keys = (("patch_id", "overlay fragments"), ("poly2019", "2019 map polygons"), ("poly2025", "2025 map polygons"))
    rows, shared = [], []

    def ledger(name, a, b, extra=None):
        r = {"Comparison": name, "Evaluation samples": int(b.sum())}
        for key, lab in keys:
            A, B = set(df.loc[a, key].dropna()), set(df.loc[b, key].dropna())
            sh = A & B
            r.update({f"Training {lab}": len(A), f"Evaluation {lab}": len(B), f"Shared {lab}": len(sh),
                      f"Evaluation samples in shared {lab}": int((b & df[key].isin(sh)).sum())})
            shared.extend({"Comparison": name, "ID type": lab, "ID": s, "Training samples": int((a & df[key].eq(s)).sum()),
                           "Evaluation samples": int((b & df[key].eq(s)).sum())} for s in sorted(sh))
        r.update(extra or {})
        rows.append(r)

    pred = model().fit(X.loc[tr, cols], y[tr]).predict(X.loc[te, cols])
    frag = set(df.loc[tr, "patch_id"]) & set(df.loc[te, "patch_id"])
    keep_f = ~df.loc[te, "patch_id"].isin(frag).to_numpy()
    keep_any = np.ones(te.sum(), bool)
    for key, _ in keys:
        sh = set(df.loc[tr, key].dropna()) & set(df.loc[te, key].dropna())
        keep_any &= ~df.loc[te, key].isin(sh).to_numpy()
    ledger("Training blocks vs spatial-block hold-out", tr, te,
           {"Hold-out F1 (AE+S2+TERR), all": round(f1_score(y[te], pred), 3),
            "Hold-out F1 excluding shared fragments": round(f1_score(y[te][keep_f], pred[keep_f]), 3),
            "Hold-out F1 excluding samples in any shared fragment or map polygon": round(f1_score(y[te][keep_any], pred[keep_any]), 3),
            "Hold-out samples kept (any-ID rule)": int(keep_any.sum())})
    idx = np.flatnonzero(tr)
    for k, (a_i, b_i) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                                   .split(X.loc[tr, cols], y[tr], df.loc[tr, "block_id"]), 1):
        a, b = np.zeros(len(df), bool), np.zeros(len(df), bool)
        a[idx[a_i]], b[idx[b_i]] = True, True
        ledger(f"CV fold {k}: other training folds vs fold {k}", a, b)
    pd.DataFrame(rows).to_csv(OUT / "R2_spatial_overlap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shared).to_csv(OUT / "R2_shared_ids.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).iloc[:, :8].to_string())


if __name__ == "__main__":
    main()
