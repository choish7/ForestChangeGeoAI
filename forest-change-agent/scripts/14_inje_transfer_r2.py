"""E10 (R2) Inje transfer test rerun on the archived Chuncheon feature matrix.

Why: in R1 the Chuncheon column of Table 11 came from a separate Colab run (features re-extracted from the GEE
asset copy of the samples), so it differed from Table 5 (S2 0.772→0.764, TERR 0.513→0.487, AE+S2+TERR 0.936→0.934,
type macro-F1 0.777→0.785). A local check on 2026-11-09 showed that re-extracting the Chuncheon features with the
same code reproduces work/features.parquet exactly (all 582 columns, max abs diff 0) and that the Table 5 scores
are reproduced exactly, independent of row order. The R1 discrepancy therefore came from the Colab run environment,
not from the data. This script removes it: Chuncheon features = work/features.parquet (the Table 5 matrix),
Inje features = fcd.gee.extract_points (identical extraction code), feature construction = fcd.features.build,
classifier = 04_train_evaluate.model().

  python forest-change-agent/scripts/14_inje_transfer_r2.py
Outputs: outputs/results/E10_inje_transfer_r2.json, work/inje/inje_features_r2.parquet (cached Inje features)
"""
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from fcd import config as C
from fcd import features, gee

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

model = import_module("04_train_evaluate").model

INJE = ROOT / "work" / "inje"
OUT = ROOT / "outputs" / "results" / "E10_inje_transfer_r2.json"
TYPES = ["harvest", "conversion", "gain"]
SEED = C.SEED


def cv_f1(X, y, g, average="binary"):
    s = [f1_score(y.iloc[b], model().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]), average=average)
         for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g)]
    return float(np.mean(s)), float(np.std(s))


def inje_features():
    cache = INJE / "inje_features_r2.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    s = pd.read_parquet(INJE / "inje_samples.parquet")
    f = gee.extract_points(s[["sample_id", "lon", "lat"]], chunk=150, workers=6, verbose=False)
    df = s.merge(f, on="sample_id", how="inner")
    df.to_parquet(cache)
    return df


def main():
    ch = pd.read_parquet(C.FEATURES).reset_index(drop=True)
    ij = inje_features().reset_index(drop=True)
    Xc, G = features.build(ch)
    Xi, _ = features.build(ij)
    yc, yi = ch["changed"].astype(int), ij["changed"].astype(int)
    tr, te = ch["split"].eq("train"), ch["split"].eq("test_spatial")
    combos = {"AE": G["AE"], "S2": G["S2"], "TERR": G["TERR"], "AE+S2": G["AE"] + G["S2"],
              "AE+S2+TERR": G["AE"] + G["S2"] + G["TERR"]}

    t1, thr_diag, extra = [], {}, {}
    for name, cols in combos.items():
        m = model().fit(Xc.loc[tr, cols], yc[tr])
        prob = m.predict_proba(Xi[cols])[:, 1]
        p = (prob >= 0.5).astype(int)
        cvm, cvs = cv_f1(Xi[cols], yi, ij["block_id"])
        row = {"features": name, "n_feat": len(cols),
               "chuncheon_holdout_F1": round(f1_score(yc[te], m.predict(Xc.loc[te, cols])), 3),
               "inje_transfer_F1": round(f1_score(yi, p), 3), "P": round(precision_score(yi, p), 3),
               "R": round(recall_score(yi, p), 3), "inje_local_CV_F1": f"{cvm:.3f} ± {cvs:.3f}"}
        if name in ("AE", "AE+S2", "AE+S2+TERR"):
            d = {}
            for t in (0.5, 0.2, 0.1):
                pt = (prob >= t).astype(int)
                rec = {c: round(float(pt[ij["change_type"].eq(c)].mean()), 3) for c in TYPES}
                d[str(t)] = [round(f1_score(yi, pt), 3), rec]
            thr_diag[name] = d
            row["inje_transfer_F1_thr0.2"] = d["0.2"][0]
        if name == "AE+S2+TERR":
            extra["confusion[[TN,FP],[FN,TP]]"] = confusion_matrix(yi, p).tolist()
            extra["FN_by_type"] = {c: [int(((p == 0) & ij["change_type"].eq(c)).sum()), int(ij["change_type"].eq(c).sum()),
                                       round(float(((p == 0) & ij["change_type"].eq(c)).sum() / ij["change_type"].eq(c).sum()), 3)]
                                   for c in TYPES}
        t1.append(row)
        print(row, flush=True)

    med = {k: {c: round(float(X.loc[d["change_type"].eq(c), "ae_cos_dist"].median()), 3)
               for c in ["no_change"] + TYPES} for k, X, d in (("chuncheon", Xc, ch), ("inje", Xi, ij))}

    cols = combos["AE+S2+TERR"]
    cc, ci = ch["change_type"].isin(TYPES), ij["change_type"].isin(TYPES)
    mt = model().fit(Xc.loc[tr & cc, cols], ch.loc[tr & cc, "change_type"])
    pc = mt.predict(Xc.loc[te & cc, cols])
    pi = mt.predict(Xi.loc[ci, cols])
    t2 = {"chuncheon_holdout_macroF1": round(f1_score(ch.loc[te & cc, "change_type"], pc, average="macro"), 3),
          "inje_transfer_macroF1": round(f1_score(ij.loc[ci, "change_type"], pi, average="macro"), 3),
          "inje_transfer_classF1": dict(zip(TYPES, f1_score(ij.loc[ci, "change_type"], pi, labels=TYPES, average=None).round(3).tolist())),
          "inje_local_CV_macroF1": "%.3f ± %.3f" % cv_f1(Xi.loc[ci, cols], ij.loc[ci, "change_type"], ij.loc[ci, "block_id"], "macro"),
          "inje_transfer_confusion(rows=true: harvest, conversion, gain)": confusion_matrix(ij.loc[ci, "change_type"], pi, labels=TYPES).tolist()}
    print(t2)

    res = {"description": "R2 rerun of E10. Chuncheon = work/features.parquet (Table 5 matrix); Inje = same fcd extraction code.",
           "env": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                   "pandas": pd.__version__},
           "n_inje": int(len(ij)), "T1_binary": t1, "threshold_diagnostic": thr_diag,
           "T1_transfer_AE+S2+TERR": extra, "AE_cos_dist_median_by_class": med, "T2_type": t2}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
