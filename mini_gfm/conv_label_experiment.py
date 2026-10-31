#!/usr/bin/env python3
# coding: utf-8
"""산지전용 고시 라벨(Chuncheon_convlabels_AE_10m.tif)로 ① 최종 변화지도 검증 ② 라벨 보강 재학습 비교.

① 검증: final_change_class.tif 에서 고시 라벨 화소·필지가 산지전용(2)/변화(1~3)로 분류된 비율, 무작위 산림 화소와 비교
② 재학습: 같은 표본·홀드아웃(final_change_map.samples)으로 AE+S2지수 분류기를 두 번 학습
   기본  : 합의 라벨 TRAIN_P2
   보강  : + 전용확정(1) 화소 → 산지전용(2), 전용이전(3) 중 라벨 없는 화소 → 무변화(0) (2019 이전 전용, 이후 안정)
           학습 블록의 고시 라벨 화소는 표본에 모두 포함
   비교  : 홀드아웃 블록 산지전용 정밀도·재현율·F1, macro F1, 홀드아웃 전용확정 화소 재현율

Colab:
  %run conv_label_experiment.py --final /content/runs/final_p2/final_change_class.tif --conv /content/data/Chuncheon_convlabels_AE_10m.tif \
       --ae-dir /content/data/ae --labels /content/data/Chuncheon_labels_AE_10m.tif --s2 {s2_54} --outdir /content/runs/conv_exp
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import precision_recall_fscore_support

from ae_change_classifier import AE_YEARS, PERIOD_YEARS, ae_features, ae_path, s2_features
from final_change_map import samples
from mini_gfm_ts import FC_NAMES, block_split, read_labels, train_classifier

CONV = {1: "전용확정", 2: "전용후보", 3: "전용이전"}


def validate_final(final_path, conv, forest_hold=None):
    with rasterio.open(final_path) as ds:
        fin = ds.read(1)
    out = {}
    rng = np.random.default_rng(0)
    forest = np.flatnonzero(fin.ravel() != 255)
    base = fin.ravel()[rng.choice(forest, 200000, replace=False)]
    out["무작위_산림화소"] = {"산지전용": round(float((base == 2).mean()), 4), "변화(1~3)": round(float(np.isin(base, [1, 2, 3]).mean()), 4)}
    for k, name in CONV.items():
        m = conv == k
        v = fin[m]
        inside = v != 255
        out[name] = {"화소": int(m.sum()), "산림마스크_안": round(float(inside.mean()), 3),
                     "산지전용_비율": round(float((v[inside] == 2).mean()), 3) if inside.any() else None,
                     "변화(1~3)_비율": round(float(np.isin(v[inside], [1, 2, 3]).mean()), 3) if inside.any() else None,
                     "벌채·갱신_비율": round(float((v[inside] == 1).mean()), 3) if inside.any() else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--final", required=True)
    p.add_argument("--conv", required=True)
    p.add_argument("--ae-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--s2", required=True)
    p.add_argument("--outdir", default="runs/conv_exp")
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    labels_full, (w, h), _ = read_labels(args.labels, "P2")
    with rasterio.open(args.labels) as lb:
        ae_tf = lb.transform
    with rasterio.open(args.conv) as cv:
        assert (cv.width, cv.height, cv.transform) == (w, h, ae_tf), "고시 라벨 격자가 AlphaEarth 격자와 다름"
        conv = cv.read(1)

    # ① 최종 지도 검증
    val = validate_final(args.final, conv)
    print("[① 최종 지도 검증]")
    for k, v in val.items():
        print(f"  {k}: {v}")

    # ② 라벨 보강 재학습
    with rasterio.open(args.s2) as s:
        s2_tf, s2_h, s2_w = s.transform, s.height, s.width
    hold_s2 = block_split(s2_h, s2_w, 512, args.seed)
    base_fc = labels_full["FC_TYPE"]
    aug_fc = base_fc.copy()
    aug_fc[conv == 1] = 2
    aug_fc[(conv == 3) & (base_fc == 255)] = 0
    variants = {"기본(합의 라벨)": base_fc, "보강(+고시 산지전용)": aug_fc}

    idx_sets, hold = {}, None
    for name, fc in variants.items():
        lab = dict(labels_full, FC_TYPE=fc)
        idx, hold, r2, c2 = samples(lab, w, h, ae_tf, s2_tf, s2_h, s2_w, hold_s2, args.seed)
        if name.startswith("보강"):  # 학습 블록의 고시 라벨 화소는 모두 포함
            extra = np.flatnonzero(~hold & np.isin(conv.ravel(), [1, 3]) & (fc.ravel() != 255))
            idx = np.union1d(idx, extra)
        idx_sets[name] = idx
    all_idx = np.union1d(*idx_sets.values())
    print(f"[② 표본] 기본 {len(idx_sets['기본(합의 라벨)']):,} / 보강 {len(idx_sets['보강(+고시 산지전용)']):,} 화소, "
          f"보강으로 추가된 산지전용 학습화소 {int((~hold & (conv.ravel() == 1)).sum())}, 홀드아웃 전용확정 화소 {int((hold & (conv.ravel() == 1)).sum())}")

    embs = {}
    for y in AE_YEARS:
        with rasterio.open(ae_path(args.ae_dir, y)) as ds:
            embs[y] = ds.read().reshape(64, -1)[:, all_idx].T.copy()
    F_ae, n_ae, _, valid_ae = ae_features(embs, *PERIOD_YEARS["P2"])
    del embs
    s2_flat = r2 * s2_w + c2
    F_s2, n_s2, valid_s2 = s2_features(args.s2, r2[all_idx], c2[all_idx])
    F_all = np.concatenate([F_ae, F_s2], 1)
    ok_all = valid_ae & valid_s2

    results = {}
    for name, fc in variants.items():
        sel = np.searchsorted(all_idx, idx_sets[name])
        F = F_all[sel]
        lab = {k: v.ravel()[idx_sets[name]] for k, v in labels_full.items()}
        lab["FC_TYPE"] = np.where(ok_all[sel], fc.ravel()[idx_sets[name]], 255)
        clf, met = train_classifier(F, n_ae + n_s2, lab, hold[idx_sets[name]], args.seed, max_per_class=60000)
        # 같은 평가 기준: 홀드아웃 블록의 '기본' 라벨 + 홀드아웃 고시 전용확정 화소
        te = np.searchsorted(all_idx, np.intersect1d(all_idx, np.flatnonzero(hold)))
        te = te[ok_all[te]]
        pred = clf.predict(F_all[te])
        y_base = base_fc.ravel()[all_idx[te]]
        m = y_base != 255
        pr, rc, f1, _ = precision_recall_fscore_support(y_base[m], pred[m], labels=[0, 1, 2, 3], zero_division=0)
        cm = conv.ravel()[all_idx[te]] == 1
        res = {"macro_F1(기본라벨)": round(float(f1.mean()), 4),
               "산지전용_정밀도": round(float(pr[2]), 3), "산지전용_재현율": round(float(rc[2]), 3), "산지전용_F1": round(float(f1[2]), 3),
               "벌채갱신_F1": round(float(f1[1]), 3), "산림증가_F1": round(float(f1[3]), 3),
               "홀드아웃_전용확정화소": int(cm.sum()),
               "전용확정_산지전용예측비율": round(float((pred[cm] == 2).mean()), 3) if cm.any() else None,
               "전용확정_변화예측비율": round(float(np.isin(pred[cm], [1, 2, 3]).mean()), 3) if cm.any() else None,
               "분류기_자체지표": {k: met[k] for k in ("macro_f1", "changed_auroc", "changed_ap", "n_train")}}
        results[name] = res
        print(f"[② {name}] {json.dumps(res, ensure_ascii=False)}", flush=True)

    report = {"검증_최종지도": val, "재학습_비교": results, "elapsed_min": round((time.time() - t0) / 60, 1)}
    (out / "conv_experiment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] {report['elapsed_min']}분 → {out / 'conv_experiment.json'}")


if __name__ == "__main__":
    main()
