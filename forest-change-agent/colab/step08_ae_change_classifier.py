#!/usr/bin/env python3
# coding: utf-8
# [colab 정리본] 원본: s2_tsfm/ae_change_classifier.py — 파일명·모듈명만 실험 단계 번호에 맞춤 (계산 로직 동일)
"""AlphaEarth Foundations 연간 임베딩(2017~2025, 64차원, 10 m) 산림변화 분류기 — S2-TSFM 과 같은 조건으로 비교.

비교 조건 (step05_s2_tsfm.py 와 동일)
  - 라벨: build_training_label_rasters.py 의 TRAIN_P1/P2 (경영실적·임상도 합의), AlphaEarth 격자(10 m) 판
  - 공간 홀드아웃: Sentinel 20 m 격자의 512 px 블록(seed 42)을 그대로 10 m 화소에 대응 → 지리적으로 같은 검증 구역
  - 분류기·지표: step05_s2_tsfm.train_classifier / auc_ap (HistGradientBoosting, 클래스당 최대 6만 화소)
특징 (구간 y0→y1: P2 2019→2025, P1 2017→2019 — AlphaEarth 는 2017 부터)
  AE: e_y0, e_y1, e_y1−e_y0 (각 64), 연속 연도 코사인 거리 8, y0→y1 거리, 구간 내 최대 연속 거리
  AE+S2: 위 + Sentinel-2 지수 특징(step05_s2_tsfm.features 의 '지수만' 세트, 20 m 화소 값을 10 m 에 대응)
메모리: 연도별 1.35 GB(int8)를 한 해씩 읽어 라벨 화소만 뽑고, 지도는 행 띠 단위로 예측

Colab:
  %run step08_ae_change_classifier.py --ae-dir /content/data/ae --labels /content/data/Chuncheon_labels_AE_10m.tif \
       --s2 /content/data/Chuncheon_S2_2017_2025_54Bands.tif --period P2 --outdir /content/runs/ae_p2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from step05_s2_tsfm import (FC_NAMES, auc_ap, block_split, load_stack, normalize, read_labels,
                         train_classifier, write_tif)

AE_YEARS = list(range(2017, 2026))
AE_NODATA = -128
PERIOD_YEARS = {"P2": (2019, 2025), "P1": (2017, 2019)}


def ae_path(ae_dir, year):
    hits = sorted(Path(ae_dir).glob(f"Chuncheon_AE_{year}_64Bands_int8*.tif"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{ae_dir} 에서 {year} AlphaEarth 파일을 하나로 찾지 못함: {hits}")
    return hits[0]


def s2_index(ae_tf, s2_tf, rows, cols):
    """AlphaEarth 화소 중심 → Sentinel 20 m 화소 (행, 열)."""
    x = ae_tf.c + (cols + 0.5) * ae_tf.a
    y = ae_tf.f + (rows + 0.5) * ae_tf.e
    return ((y - s2_tf.f) / s2_tf.e).astype(int), ((x - s2_tf.c) / s2_tf.a).astype(int)


def unit(e):
    e = e.astype(np.float32) / 127.0
    return e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-6)


def ae_features(embs, y0, y1):
    """embs: {year: [N, 64] int8} → 특징 [N, F], 이름, 비지도 점수 dict."""
    u = {y: unit(e) for y, e in embs.items()}
    cos = lambda a, b: 1.0 - (u[a] * u[b]).sum(-1)
    consec = np.stack([cos(a, b) for a, b in zip(AE_YEARS, AE_YEARS[1:])], 1)
    in_p = [i for i, (a, b) in enumerate(zip(AE_YEARS, AE_YEARS[1:])) if y0 <= a and b <= y1]
    d01 = cos(y0, y1)
    mx = consec[:, in_p].max(1)
    feats = np.concatenate([u[y0], u[y1], u[y1] - u[y0], consec, d01[:, None], mx[:, None]], 1)
    names = ([f"ae{y0}_{i:02d}" for i in range(64)] + [f"ae{y1}_{i:02d}" for i in range(64)]
             + [f"aedelta_{i:02d}" for i in range(64)] + [f"aecos_{a}_{b}" for a, b in zip(AE_YEARS, AE_YEARS[1:])]
             + [f"aecos_{y0}_{y1}", f"aecos_max_{y0}_{y1}"])
    year_max = np.asarray([AE_YEARS[i + 1] for i in in_p])[consec[:, in_p].argmax(1)]
    valid = np.all([((e != AE_NODATA) & (e != 0)).any(-1) for e in embs.values()], 0)  # 경계 밖은 0벡터로 저장됨
    return feats.astype(np.float32), names, {"ae_first_last": d01, "ae_max_consec": mx, "ae_max_year": year_max}, valid


def s2_features(s2_path, r2, c2):
    """step05_s2_tsfm.features 의 '지수만' 세트를 지정 화소에서 계산."""
    arr, valid, n, meta = load_stack(s2_path)
    stats = {"slope_mean": float(np.nanmean(arr[3 * n][valid])), "slope_std": float(max(1, np.nanstd(arr[3 * n][valid]))),
             "elev_mean": float(np.nanmean(arr[3 * n + 1][valid])), "elev_std": float(max(1, np.nanstd(arr[3 * n + 1][valid])))}
    x = normalize(arr, valid, n, stats)[:, r2, c2]
    del arr
    ndvi, ndwi, ndmi = x[0:3 * n:3] / 2, x[1:3 * n:3] / 2, x[2:3 * n:3] / 2
    f = np.concatenate([ndvi, ndwi, ndmi, np.diff(ndvi, axis=0), np.diff(ndmi, axis=0), x[3 * n:3 * n + 2],
                        np.stack([ndvi[-1] - ndvi[0], ndmi[-1] - ndmi[0], -np.diff(ndvi, axis=0).min(0)])]).T
    yrs = meta["years"]
    names = ([f"ndvi{y}" for y in yrs] + [f"ndwi{y}" for y in yrs] + [f"ndmi{y}" for y in yrs]
             + [f"dndvi{y}" for y in yrs[1:]] + [f"dndmi{y}" for y in yrs[1:]] + ["slope", "elev", "dNDVI", "dNDMI", "ndvi_max_drop"])
    return f.astype(np.float32), names, valid[r2, c2]


def evaluate_ae_scores(scores, labels, hold):
    fc, mg, nat = labels["FC_TYPE"], labels["MGMT"], labels["SOURCE"] == 1
    tasks = {
        "임상도 변화여부(벌채·전용·증가 vs 무변화)": (np.isin(fc, [1, 2, 3]), fc == 0),
        "국유림 검증라벨 변화여부(실적·임상도 합의)": (np.isin(fc, [1, 2, 3]) & nat, (fc == 0) & nat),
        "임상도 벌채·갱신 vs 무변화": (fc == 1, fc == 0),
        "국유림 실적 수확·갱신 vs 실적없음": (mg == 1, mg == 0),
        "국유림 실적 솎아베기 vs 실적없음": (mg == 2, mg == 0),
    }
    rows = []
    for task, (pos, neg) in tasks.items():
        m = hold & (pos | neg)
        y = pos[m].astype(int)
        for name, key in (("AE 구간 첫해→끝해 코사인거리", "ae_first_last"), ("AE 구간 최대 연속 코사인거리", "ae_max_consec")):
            auc, ap = auc_ap(y, scores[key][m])
            rows.append({"task": task, "score": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                         "auroc": round(auc, 4), "ap": round(ap, 4)})
    return rows


def predict_map(clf, ae_dir, y0, y1, profile, outdir, strip=128):  # 128행 띠 ≈ 9년×64밴드 int8 340 MB
    files = {y: rasterio.open(ae_path(ae_dir, y)) for y in AE_YEARS}
    h, w = profile["height"], profile["width"]
    cls_map = np.full((h, w), 255, np.uint8)
    prob = np.full((len(clf.classes_), h, w), np.nan, np.float32)
    try:
        for r in range(0, h, strip):
            win = Window(0, r, w, min(strip, h - r))
            embs = {y: f.read(window=win).reshape(64, -1).T for y, f in files.items()}
            F, _, _, valid = ae_features(embs, y0, y1)
            if valid.any():
                p = clf.predict_proba(F[valid])
                rows = slice(r, r + win.height)
                flat_c = cls_map[rows].reshape(-1)
                flat_c[valid] = clf.classes_[p.argmax(1)]
                cls_map[rows] = flat_c.reshape(win.height, w)
                for k in range(len(clf.classes_)):
                    flat_p = prob[k, rows].reshape(-1)
                    flat_p[valid] = p[:, k]
                    prob[k, rows] = flat_p.reshape(win.height, w)
            if (r // strip) % 8 == 7:
                print(f"  [map] 행 {r + win.height}/{h}", flush=True)
    finally:
        for f in files.values():
            f.close()
    write_tif(outdir / "ae_change_class.tif", profile, {"change_class": cls_map}, dtype="uint8", nodata=255)
    write_tif(outdir / "ae_change_prob.tif", profile, {f"p_{FC_NAMES[k]}": prob[i] for i, k in enumerate(clf.classes_)})
    return cls_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-dir", required=True)
    p.add_argument("--labels", required=True, help="AlphaEarth 격자(10 m) 학습 라벨 래스터")
    p.add_argument("--s2", default=None, help="Sentinel 지수 스택 (AE+S2 결합 특징·홀드아웃 블록 격자)")
    p.add_argument("--s2-labels", default=None, help="Sentinel 20 m 라벨(--s2 없을 때 블록 격자 기준)")
    p.add_argument("--period", choices=["P1", "P2"], default="P2")
    p.add_argument("--outdir", default="runs/ae")
    p.add_argument("--block", type=int, default=512, help="Sentinel 20 m 기준 홀드아웃 블록(px) — step05_s2_tsfm 와 같아야 함")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-map", action="store_true")
    args, _ = p.parse_known_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y0, y1 = PERIOD_YEARS[args.period]

    labels_full, (w, h), fmt = read_labels(args.labels, args.period)
    with rasterio.open(args.labels) as lb:
        ae_tf, profile = lb.transform, lb.profile.copy()
    with rasterio.open(ae_path(args.ae_dir, AE_YEARS[0])) as ref:
        if (ref.width, ref.height, ref.transform) != (w, h, ae_tf):
            raise ValueError("라벨 격자가 AlphaEarth 격자와 다름 — build_training_label_rasters.py --ae-ref 로 다시 만드세요")
        profile.update(crs=ref.crs)
    grid_src = args.s2 or args.s2_labels
    with rasterio.open(grid_src) as s:
        s2_tf, s2_h, s2_w = s.transform, s.height, s.width
    hold_s2 = block_split(s2_h, s2_w, args.block, args.seed)

    # 평가·학습에 쓸 화소: 홀드아웃 블록의 모든 라벨 화소 + 학습 블록의 클래스당 최대 6만 화소
    fc, mg = labels_full["FC_TYPE"].ravel(), labels_full["MGMT"].ravel()
    rows_all, cols_all = np.divmod(np.arange(h * w), w)
    r2, c2 = s2_index(ae_tf, s2_tf, rows_all, cols_all)
    inside = (r2 >= 0) & (r2 < s2_h) & (c2 >= 0) & (c2 < s2_w)
    hold = np.zeros(h * w, bool)
    hold[inside] = hold_s2[r2[inside], c2[inside]]
    rng = np.random.default_rng(args.seed)
    test_idx = np.where(hold & inside & ((fc != 255) | (mg != 255)))[0]
    train_idx = np.concatenate([rng.permutation(np.where(~hold & inside & (fc == k))[0])[:60000] for k in FC_NAMES])
    idx = np.sort(np.concatenate([train_idx, test_idx]))
    del rows_all, cols_all
    print(f"[라벨] {fmt}, 격자 {w}×{h} 10 m, 구간 {y0}→{y1}")
    print(f"[표본] 학습 {len(train_idx):,} 화소 (클래스당 ≤6만) / 홀드아웃 평가 {len(test_idx):,} 화소")

    embs = {}
    for y in AE_YEARS:
        t = time.time()
        with rasterio.open(ae_path(args.ae_dir, y)) as ds:
            embs[y] = ds.read().reshape(64, -1)[:, idx].T.copy()
        print(f"[AE] {y} 읽기 {time.time() - t:.0f}s", flush=True)
    F_ae, n_ae, scores, valid = ae_features(embs, y0, y1)
    del embs
    lab = {k: v.ravel()[idx] for k, v in labels_full.items()}
    hold_s = hold[idx]
    lab["FC_TYPE"] = np.where(valid, lab["FC_TYPE"], 255)

    score_rows = evaluate_ae_scores(scores, lab, hold_s & valid)
    results, clf_ae = {}, None
    sets = [("AE", F_ae, n_ae)]
    if args.s2:
        F_s2, n_s2, s2_valid = s2_features(args.s2, r2[idx], c2[idx])
        lab["FC_TYPE"] = np.where(s2_valid, lab["FC_TYPE"], 255)
        sets.append(("AE+S2지수", np.concatenate([F_ae, F_s2], 1), n_ae + n_s2))
        sets.append(("S2지수(10m 대응)", F_s2, n_s2))
    for tag, F, names in sets:
        clf, met = train_classifier(F, names, lab, hold_s, args.seed)
        met["n_features"] = len(names)
        results[tag] = met
        print(f"[classifier] {tag}: 특징 {len(names)}개, balanced acc {met['balanced_accuracy']}, macro F1 {met['macro_f1']}, "
              f"변화 AUROC {met['changed_auroc']}, AP {met['changed_ap']} | 국유림 검증 AUROC {met['verified_changed_auroc']}, "
              f"AP {met['verified_changed_ap']}", flush=True)
        if tag == "AE":
            clf_ae = clf

    fy = lab["FC_YEAR"].astype(float)
    fy[fy == 255] = np.nan
    m = hold_s & np.isin(lab["FC_TYPE"], [1, 2, 3]) & np.isfinite(fy) & (fy + 2000 >= y0)
    diff = np.abs(scores["ae_max_year"][m] - (fy[m] + 2000))
    year_acc = {"n": int(m.sum()), "exact": round(float((diff == 0).mean()), 4) if m.any() else None,
                "within_1yr": round(float((diff <= 1).mean()), 4) if m.any() else None}

    class_area = None
    if not args.no_map:
        print("[map] 전체 영상 AE 분류 지도 예측 …", flush=True)
        cls_map = predict_map(clf_ae, args.ae_dir, y0, y1, profile, outdir)
        class_area = {FC_NAMES[k]: round(float((cls_map == k).sum()) * 0.01, 1) for k in FC_NAMES}
        import joblib
        joblib.dump({"clf": clf_ae, "features": n_ae, "classes": FC_NAMES, "period": args.period}, outdir / "step08_ae_change_classifier.joblib")

    report = {"model": "AlphaEarth V1 annual embedding", "period": args.period, "years": [y0, y1],
              "unsupervised_scores": score_rows, "classifier": results, "change_year": year_acc,
              "class_area_ha": class_area, "elapsed_min": round((time.time() - t0) / 60, 1)}
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[AE 비지도 점수 AUROC / AP — 홀드아웃 블록]")
    for r in score_rows:
        print(f"  {r['task']:<32} {r['score']:<24} AUROC {r['auroc']:.3f}  AP {r['ap']:.3f}  (양성 {r['n_pos']:,} / 음성 {r['n_neg']:,})")
    print("[변화 연도 ±1년]", year_acc)
    print("[분류 면적 ha]", class_area)
    print(f"[완료] {report['elapsed_min']}분, 결과: {outdir}")


if __name__ == "__main__":
    main()
