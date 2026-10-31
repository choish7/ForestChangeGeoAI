"""[4.2-④] 변화탐지 모형 학습·검증 (정확도 / 공간 일반화 / 자료별 결합효과).

과제
  T1 변화 여부   : no_change vs {harvest, conversion, gain}         특징 AE/S2/TERR 조합 (ablation)
  T1b 산림속성 결합: 2019 산림 표본 한정, ALL vs ALL+ATTR            (gain 은 사전 속성이 없어 제외)
  T2 변화 유형   : harvest / conversion / gain (변화 표본)
  T3 발생연도    : AlphaEarth 연속연도 거리 최대 연도 vs 임상도 갱신년도(약한 라벨)

검증: 학습 블록 내 StratifiedGroupKFold(5, 공간블록) + 독립 공간 홀드아웃(test_spatial)
산출: outputs/change_model.joblib, outputs/metrics.json, outputs/ablation.csv, outputs/test_predictions.csv
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from fcd import config as C
from fcd import features
from fcd.labels import TRAIN_CLASSES

TYPE_CLASSES = ["harvest", "conversion", "gain"]


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=C.SEED)


def spatial_cv(X, y, groups, n=5, average="binary"):
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=C.SEED)
    scores = []
    for tr, va in cv.split(X, y, groups):
        m = model().fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[va], m.predict(X.iloc[va]), average=average))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    df = pd.read_parquet(C.FEATURES)
    df = df[df["change_type"].isin(TRAIN_CLASSES)].reset_index(drop=True)
    X, G = features.build(df)
    tr, te = df["split"].eq("train"), df["split"].eq("test_spatial")
    y_chg = df["changed"].astype(int)
    metrics = {"n_train": int(tr.sum()), "n_test_spatial": int(te.sum()),
               "class_counts": df["change_type"].value_counts().to_dict()}

    # ---------------- T1 변화 여부 + ablation (7 조합)
    rows = []
    base = ["AE", "S2", "TERR"]
    for k in (1, 2, 3):
        for combo in itertools.combinations(base, k):
            cols = sum((G[g] for g in combo), [])
            cv_m, cv_s = spatial_cv(X.loc[tr, cols], y_chg[tr], df.loc[tr, "block_id"])
            m = model().fit(X.loc[tr, cols], y_chg[tr])
            f1_te = f1_score(y_chg[te], m.predict(X.loc[te, cols]))
            rows.append({"task": "T1_change", "features": "+".join(combo), "n_feat": len(cols),
                         "cv_f1_mean": cv_m, "cv_f1_sd": cv_s, "test_spatial_f1": f1_te,
                         "generalization_gap": cv_m - f1_te})
            print(f"[T1] {'+'.join(combo):<12} CV F1 {cv_m:.3f}±{cv_s:.3f} | 공간홀드아웃 F1 {f1_te:.3f}")

    # ---------------- T1b 산림속성 결합효과 (2019 산림 표본)
    forest = df["change_type"].isin(["no_change", "harvest", "conversion"])
    all_cols = G["AE"] + G["S2"] + G["TERR"]
    for name, cols in (("AE+S2+TERR", all_cols), ("AE+S2+TERR+ATTR", all_cols + G["ATTR"])):
        cv_m, cv_s = spatial_cv(X.loc[tr & forest, cols], y_chg[tr & forest], df.loc[tr & forest, "block_id"])
        m = model().fit(X.loc[tr & forest, cols], y_chg[tr & forest])
        f1_te = f1_score(y_chg[te & forest], m.predict(X.loc[te & forest, cols]))
        rows.append({"task": "T1b_loss_forest2019", "features": name, "n_feat": len(cols),
                     "cv_f1_mean": cv_m, "cv_f1_sd": cv_s, "test_spatial_f1": f1_te,
                     "generalization_gap": cv_m - f1_te})
        print(f"[T1b] {name:<16} CV F1 {cv_m:.3f}±{cv_s:.3f} | 공간홀드아웃 F1 {f1_te:.3f}")
    pd.DataFrame(rows).round(4).to_csv(C.OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    # ---------------- 최종 T1 모형 (AE+S2+TERR)
    m_chg = model().fit(X.loc[tr, all_cols], y_chg[tr])
    p_te = m_chg.predict_proba(X.loc[te, all_cols])[:, 1]
    yhat = (p_te >= 0.5).astype(int)
    metrics["T1_change"] = {
        "features": "AE+S2+TERR", "f1": f1_score(y_chg[te], yhat),
        "precision": precision_score(y_chg[te], yhat), "recall": recall_score(y_chg[te], yhat),
        "balanced_accuracy": balanced_accuracy_score(y_chg[te], yhat),
        "confusion[[TN,FP],[FN,TP]]": confusion_matrix(y_chg[te], yhat).tolist()}

    # ---------------- T2 변화 유형
    chg = df["change_type"].isin(TYPE_CLASSES)
    y_type = df["change_type"]
    m_type = model().fit(X.loc[tr & chg, all_cols], y_type[tr & chg])
    t_pred = m_type.predict(X.loc[te & chg, all_cols])
    cv_m, cv_s = spatial_cv(X.loc[tr & chg, all_cols], y_type[tr & chg], df.loc[tr & chg, "block_id"], average="macro")
    metrics["T2_type"] = {
        "cv_macro_f1": [cv_m, cv_s],
        "test_macro_f1": f1_score(y_type[te & chg], t_pred, average="macro"),
        "per_class_f1": dict(zip(TYPE_CLASSES, f1_score(y_type[te & chg], t_pred, labels=TYPE_CLASSES, average=None).round(3).tolist())),
        "confusion(rows=true)": {"labels": TYPE_CLASSES,
                                 "matrix": confusion_matrix(y_type[te & chg], t_pred, labels=TYPE_CLASSES).tolist()}}
    print(f"[T2] 유형 macro-F1 CV {cv_m:.3f}±{cv_s:.3f} | 공간홀드아웃 {metrics['T2_type']['test_macro_f1']:.3f}")

    # ---------------- T3 발생연도 (규칙 기반, 약한 라벨)
    yl = df["change_year"].astype("float")
    has = chg & yl.notna()
    lag = yl[has] - X.loc[has, "ae_peak_year"]          # 도면 갱신은 실제 변화 이후 → lag ≥ 0 기대
    err = lag.abs()
    metrics["T3_year"] = {"n_labeled": int(has.sum()),
                          "label_year_distribution": {int(k): int(v) for k, v in yl[has].value_counts().sort_index().items()},
                          "exact": float((err == 0).mean()) if len(err) else None,
                          "within_1yr": float((err <= 1).mean()) if len(err) else None,
                          "peak_before_update_0to2yr": float(lag.between(0, 2).mean()) if len(err) else None,
                          "peak_before_update_any": float((lag >= 0).mean()) if len(err) else None,
                          "note": "라벨=2025 임상도 갱신년도(도면 갱신시점) → 실제 발생연도보다 늦을 수 있음"}
    print(f"[T3] 연도라벨 {has.sum()}건, 일치 {metrics['T3_year']['exact']}, ±1년 {metrics['T3_year']['within_1yr']}")

    # ---------------- 즉석 스캔용 코사인거리 임계값 (학습셋 산림손실 F1 최대)
    loss = df["change_type"].isin(["harvest", "conversion"])
    sel = tr & (loss | df["change_type"].eq("no_change"))
    pr, rc, th = precision_recall_curve(loss[sel].astype(int), X.loc[sel, "ae_cos_dist"])
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    thr = float(th[np.argmax(f1s[:-1])])
    metrics["scan_threshold"] = {"cos_dist": thr, "train_f1_at_thr": float(f1s[:-1].max())}
    print(f"[임계값] AlphaEarth 코사인거리 {thr:.4f} (학습 F1 {f1s[:-1].max():.3f})")

    joblib.dump({"change": m_chg, "type": m_type, "features": all_cols, "threshold": thr,
                 "years": (C.YEAR_FROM, C.YEAR_TO), "trained_region": "춘천시(51110)",
                 "type_classes": TYPE_CLASSES}, C.MODEL)
    out = df.loc[te, ["sample_id", "change_type", "block_id", "lon", "lat"]].assign(
        p_change=p_te.round(4), pred_type=m_type.predict(X.loc[te, all_cols]),
        ae_cos_dist=X.loc[te, "ae_cos_dist"].round(4), ae_peak_year=X.loc[te, "ae_peak_year"])
    out.to_csv(C.OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (C.OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(metrics["T1_change"], ensure_ascii=False, default=float))
    print(f"[04] ✓ {C.MODEL}, outputs/metrics.json, outputs/ablation.csv")


if __name__ == "__main__":
    main()
