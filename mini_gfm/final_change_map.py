#!/usr/bin/env python3
# coding: utf-8
"""춘천 최종 산림변화 지도 (2019→2025): AlphaEarth + Sentinel-2 지수 분류기, 산림 마스크, 클래스별 최소 면적.

1) 학습: ae_change_classifier 와 같은 표본(합의 라벨 TRAIN_P2, 같은 공간 홀드아웃 블록)으로 AE+S2지수 분류기
2) 예측: 산림 마스크(임상도 2019·2025 임지) 안 10 m 화소만, 128행 띠 단위
3) 후처리(최소 면적, 8방향 연결 체 거르기): 벌채·갱신·산림증가 0.5 ha(임상도 최소 구획), 산지전용 0.2 ha
4) 산출 (outdir)
   final_change_class.tif  0 무변화, 1 벌채·갱신, 2 산지전용, 3 산림증가, 255 비산림·자료없음
   final_change_prob.tif   클래스별 확률 4밴드
   final_change_year.tif   변화 화소의 추정 발생연도(AlphaEarth 연속연도 코사인 거리 최대 연도)
   final_change_patches.gpkg  변화 패치 폴리곤 (클래스, 면적 ha, 평균 확률, 최빈 발생연도)
   final_report.json, final_preview.png

Colab:
  %run final_change_map.py --ae-dir /content/data/ae --labels /content/data/Chuncheon_labels_AE_10m.tif \
       --s2 /content/data/Chuncheon_S2_2017_2025_54Bands.tif --forest-mask /content/data/Chuncheon_forestmask_AE_10m.tif \
       --outdir /content/runs/final_p2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from rasterio.windows import Window
from scipy import ndimage

from ae_change_classifier import AE_YEARS, PERIOD_YEARS, ae_features, ae_path, s2_features, s2_index
from mini_gfm_ts import FC_NAMES, block_split, read_labels, train_classifier, write_tif

CLASS_KO = {0: "무변화", 1: "벌채·갱신", 2: "산지전용", 3: "산림증가"}


def samples(labels_full, w, h, ae_tf, s2_tf, s2_h, s2_w, hold_s2, seed):
    """ae_change_classifier.main 과 같은 표본: 홀드아웃 블록 라벨 화소 전부 + 학습 블록 클래스당 ≤6만."""
    fc, mg = labels_full["FC_TYPE"].ravel(), labels_full["MGMT"].ravel()
    rows, cols = np.divmod(np.arange(h * w), w)
    r2, c2 = s2_index(ae_tf, s2_tf, rows, cols)
    inside = (r2 >= 0) & (r2 < s2_h) & (c2 >= 0) & (c2 < s2_w)
    hold = np.zeros(h * w, bool)
    hold[inside] = hold_s2[r2[inside], c2[inside]]
    rng = np.random.default_rng(seed)
    test_idx = np.where(hold & inside & ((fc != 255) | (mg != 255)))[0]
    train_idx = np.concatenate([rng.permutation(np.where(~hold & inside & (fc == k))[0])[:60000] for k in FC_NAMES])
    idx = np.sort(np.concatenate([train_idx, test_idx]))
    return idx, hold, r2, c2


def class_mmu(cls, valid, px_ha, mmu):
    """클래스별 최소 면적: 큰 기준(0.5 ha)으로 거른 지도에, 작은 기준 클래스(산지전용 0.2 ha)의 결과를 덮어씀."""
    out = None
    for thr in sorted(set(mmu.values()), reverse=True):
        sv = sieve(cls, size=math.ceil(thr / px_ha), mask=valid, connectivity=8).astype(np.uint8)
        keep = [k for k, v in mmu.items() if v == thr]
        out = sv if out is None else np.where(np.isin(sv, keep), sv, out)
    out[~valid] = 255
    return out.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--s2", required=True)
    p.add_argument("--forest-mask", required=True)
    p.add_argument("--period", choices=["P1", "P2"], default="P2")
    p.add_argument("--outdir", default="runs/final")
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mmu-harvest", type=float, default=0.5)
    p.add_argument("--mmu-conversion", type=float, default=0.2)
    p.add_argument("--mmu-gain", type=float, default=0.5)
    p.add_argument("--strip", type=int, default=128)
    args, _ = p.parse_known_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y0, y1 = PERIOD_YEARS[args.period]
    mmu = {1: args.mmu_harvest, 2: args.mmu_conversion, 3: args.mmu_gain}

    labels_full, (w, h), fmt = read_labels(args.labels, args.period)
    with rasterio.open(args.labels) as lb:
        ae_tf, profile = lb.transform, lb.profile.copy()
    with rasterio.open(args.forest_mask) as fm:
        assert (fm.width, fm.height, fm.transform) == (w, h, ae_tf), "산림 마스크 격자가 AlphaEarth 격자와 다름"
        forest = fm.read(1) == 1
    with rasterio.open(args.s2) as s:
        s2_tf, s2_h, s2_w = s.transform, s.height, s.width
    hold_s2 = block_split(s2_h, s2_w, args.block, args.seed)
    px_ha = abs(ae_tf.a * ae_tf.e) / 1e4

    # 1) 학습 (AE + S2 지수)
    idx, hold, r2, c2 = samples(labels_full, w, h, ae_tf, s2_tf, s2_h, s2_w, hold_s2, args.seed)
    embs = {}
    for y in AE_YEARS:
        with rasterio.open(ae_path(args.ae_dir, y)) as ds:
            embs[y] = ds.read().reshape(64, -1)[:, idx].T.copy()
    F_ae, n_ae, _, valid_s = ae_features(embs, y0, y1)
    del embs
    print("[S2] 전체 화소 지수 특징 계산 …", flush=True)
    s2_rows, s2_cols = np.divmod(np.arange(s2_h * s2_w), s2_w)
    F_s2_all, n_s2, s2_valid_all = s2_features(args.s2, s2_rows, s2_cols)  # [S2 화소, 48]
    del s2_rows, s2_cols
    s2_flat = r2 * s2_w + c2
    inside = (r2 >= 0) & (r2 < s2_h) & (c2 >= 0) & (c2 < s2_w)
    lab = {k: v.ravel()[idx] for k, v in labels_full.items()}
    lab["FC_TYPE"] = np.where(valid_s & s2_valid_all[s2_flat[idx]], lab["FC_TYPE"], 255)
    F = np.concatenate([F_ae, F_s2_all[s2_flat[idx]]], 1)
    clf, met = train_classifier(F, n_ae + n_s2, lab, hold[idx], args.seed)
    del F, F_ae
    print(f"[학습] AE+S2지수 특징 {len(n_ae) + len(n_s2)}개: macro F1 {met['macro_f1']}, 변화 AUROC {met['changed_auroc']}, "
          f"AP {met['changed_ap']} | 국유림 검증 AP {met['verified_changed_ap']}", flush=True)

    # 2) 산림 마스크 안 예측 (띠 단위)
    files = {y: rasterio.open(ae_path(args.ae_dir, y)) for y in AE_YEARS}
    raw = np.full((h, w), 255, np.uint8)
    prob = np.full((len(clf.classes_), h, w), np.nan, np.float32)
    year = np.zeros((h, w), np.uint16)
    in_p = [i for i, (a, b) in enumerate(zip(AE_YEARS, AE_YEARS[1:])) if y0 <= a and b <= y1]
    try:
        for r in range(0, h, args.strip):
            win = Window(0, r, w, min(args.strip, h - r))
            rows = slice(r, r + win.height)
            sel = forest[rows].ravel() & inside.reshape(h, w)[rows].ravel()
            if not sel.any():
                continue
            embs = {y: f.read(window=win).reshape(64, -1).T[sel] for y, f in files.items()}
            Fa, _, sc, ok = ae_features(embs, y0, y1)
            s2i = s2_flat.reshape(h, w)[rows].ravel()[sel]
            ok &= s2_valid_all[s2i]
            pr = clf.predict_proba(np.concatenate([Fa[ok], F_s2_all[s2i[ok]]], 1))
            pos = np.where(sel)[0][ok]
            for name, arr, vals in (("raw", raw, clf.classes_[pr.argmax(1)]), ("year", year, sc["ae_max_year"][ok])):
                flat = arr[rows].reshape(-1)
                flat[pos] = vals
                arr[rows] = flat.reshape(win.height, w)
            for k in range(len(clf.classes_)):
                flat = prob[k, rows].reshape(-1)
                flat[pos] = pr[:, k]
                prob[k, rows] = flat.reshape(win.height, w)
            if (r // args.strip) % 8 == 7:
                print(f"  [map] 행 {r + win.height}/{h} ({(time.time() - t0) / 60:.0f}분)", flush=True)
    finally:
        for f in files.values():
            f.close()
    del F_s2_all

    # 3) 클래스별 최소 면적
    valid = raw != 255
    final = class_mmu(raw, valid, px_ha, mmu)
    year = np.where(np.isin(final, [1, 2, 3]), year, 0).astype(np.uint16)

    # 4) 저장
    profile.update(count=1)
    write_tif(outdir / "final_change_class.tif", profile, {"change_class": final}, dtype="uint8", nodata=255)
    write_tif(outdir / "final_change_prob.tif", profile, {f"p_{FC_NAMES[k]}": prob[i] for i, k in enumerate(clf.classes_)})
    write_tif(outdir / "final_change_year.tif", profile, {"change_year": year}, dtype="uint16", nodata=0)
    import joblib
    joblib.dump({"clf": clf, "features": n_ae + n_s2, "classes": FC_NAMES, "period": args.period, "mmu_ha": mmu},
                outdir / "final_classifier.joblib")

    # 변화 패치 폴리곤
    import geopandas as gpd
    from shapely.geometry import shape
    recs = []
    for k in (1, 2, 3):
        comp, n = ndimage.label(final == k, structure=np.ones((3, 3)))
        if n == 0:
            continue
        ids = comp.ravel()
        area = np.bincount(ids, minlength=n + 1) * px_ha
        pk = np.nan_to_num(prob[list(clf.classes_).index(k)]).ravel()
        mean_p = np.bincount(ids, weights=pk, minlength=n + 1) / np.maximum(np.bincount(ids, minlength=n + 1), 1)
        yrs = np.zeros((n + 1, y1 - y0 + 1), np.int64)
        yv = year.ravel()
        m = (ids > 0) & (yv >= y0)
        np.add.at(yrs, (ids[m], yv[m] - y0), 1)
        for geom, cid in shapes(comp.astype(np.int32), mask=comp > 0, transform=ae_tf, connectivity=8):
            cid = int(cid)
            recs.append({"cid": cid, "class": k, "class_ko": CLASS_KO[k], "area_ha": round(area[cid], 3),
                         "mean_prob": round(float(mean_p[cid]), 3),
                         "change_year": int(y0 + yrs[cid].argmax()) if yrs[cid].sum() else None, "geometry": shape(geom)})
    patches = gpd.GeoDataFrame(recs, crs=profile["crs"])
    # 같은 연결요소가 여러 폴리곤으로 나올 수 있어 (클래스, 연결요소 번호)로 묶음
    patches = patches.dissolve(by=["class", "cid"], aggfunc="first", as_index=False).drop(columns="cid")
    patches.insert(0, "patch_id", [f"C{i:05d}" for i in range(len(patches))])
    patches.to_file(outdir / "final_change_patches.gpkg", layer="변화패치", driver="GPKG")

    # 홀드아웃 평가 (최종 지도)
    from sklearn.metrics import precision_recall_fscore_support
    labf = labels_full["FC_TYPE"]
    mm = hold.reshape(h, w) & (labf != 255) & (final != 255)
    pr_, rc_, f1_, _ = precision_recall_fscore_support(labf[mm], final[mm], labels=[0, 1, 2, 3], zero_division=0)
    area_ha = {CLASS_KO[k]: round(float((final == k).sum()) * px_ha, 1) for k in CLASS_KO}
    report = {"period": f"{y0}→{y1}", "model": "AlphaEarth V1 + Sentinel-2 지수 (HistGradientBoosting)",
              "forest_mask": Path(args.forest_mask).name, "mmu_ha": {CLASS_KO[k]: v for k, v in mmu.items()},
              "classifier_holdout": met, "final_map_holdout": {
                  "macro_f1": round(float(f1_.mean()), 4),
                  **{CLASS_KO[k]: {"precision": round(float(pr_[k]), 3), "recall": round(float(rc_[k]), 3),
                                   "f1": round(float(f1_[k]), 3)} for k in CLASS_KO}},
              "area_ha": area_ha, "patches": patches.groupby("class_ko").agg(
                  개수=("patch_id", "size"), 면적_ha=("area_ha", "sum"), 중앙값_ha=("area_ha", "median")).round(2).to_dict("index"),
              "change_year_area_ha": {CLASS_KO[k]: {int(yy): round(float(((final == k) & (year == yy)).sum()) * px_ha, 1)
                                                    for yy in range(y0 + 1, y1 + 1)} for k in (1, 2, 3)},
              "elapsed_min": round((time.time() - t0) / 60, 1)}
    (outdir / "final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 미리보기
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    step = 4
    view = final[::step, ::step].astype(float)
    view[view == 255] = np.nan
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(view, cmap=ListedColormap(["#d9ead3", "#d7301f", "#7b3294", "#1a9850"]), vmin=-0.5, vmax=3.5,
              interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"Chuncheon forest change {y0}-{y1} (AlphaEarth+S2, forest mask, MMU)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#d9ead3", "#d7301f", "#7b3294", "#1a9850"]]
    ax.legend(handles, [f"no change", f"harvest {area_ha['벌채·갱신']:,.0f} ha", f"conversion {area_ha['산지전용']:,.0f} ha",
                        f"gain {area_ha['산림증가']:,.0f} ha"], loc="lower right", fontsize=9)
    fig.savefig(outdir / "final_preview.png", bbox_inches="tight")

    print(json.dumps({k: report[k] for k in ("final_map_holdout", "area_ha", "patches", "change_year_area_ha")},
                     indent=1, ensure_ascii=False))
    print(f"[완료] {report['elapsed_min']}분, 결과: {outdir}")


if __name__ == "__main__":
    main()
