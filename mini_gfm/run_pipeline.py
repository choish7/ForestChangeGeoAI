#!/usr/bin/env python3
# coding: utf-8
"""CarbonAce Mini-GFM end-to-end pipeline.

  1) 입력·정답지 짝 맞추기 + 격자 일치 검사
  2) 자기지도 사전학습 (mini_gfm.train)        --checkpoint 주면 건너뜀
  3) 추론 → change_scores.tif (mini_gfm.infer)
  4) GEE 정답지(labels.tif)와 비교 평가
       - AUROC, AP(AUPRC)
       - 공간 블록 체커보드 분할: 짝수 블록에서 임계값(F1 최대) 결정 → 홀수 블록에서 F1/IoU/Kappa 보고
       - 원시 지수 변화(|ΔNDVI|, |ΔNDMI|) 기준선과 비교 → GFM 임베딩의 부가가치 확인
       - 오류 지도(TN/FP/FN/TP) GeoTIFF 저장 → QGIS 에서 확인

사용 예:
  python run_pipeline.py \
      --inputs "data/gfm_2023_2024_2025_input15*.tif" \
      --labels "data/gfm_2023_2024_2025_labels*.tif" \
      --outdir runs/exp01 --epochs 30

  # 이미 학습된 모델로 추론+평가만
  python run_pipeline.py --inputs ... --labels ... --outdir runs/exp01 \
      --checkpoint runs/exp01/model/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mini_gfm as gfm  # noqa: E402

SCORE_NODATA = -9999.0
LABEL_NODATA = 255
LABEL_BANDS = 13
NBINS = 2048

# change_scores.tif 밴드(1-based)와 점수 변환
SCORES = {
    'emb_Y1Y2': (1, 'raw'),
    'emb_Y2Y3': (2, 'raw'),
    'emb_Y1Y3': (3, 'raw'),
    'absdNDVI': (4, 'abs'),   # 기준선: 지수 변화량 크기
    'absdNDMI': (5, 'abs'),
    'negdNDVI': (4, 'neg'),   # 기준선: NDVI 감소(식생/산림 손실용)
}
# labels.tif 밴드(1-based), 양성 값, 음성 값 (그 외 값은 평가 제외)
TASKS = {
    'CHG_Y1Y2': (7, [1], [0]),
    'CHG_Y2Y3': (8, [1], [0]),
    'CHG_Y1Y3': (9, [1], [0]),
    'VEG_LOSS': (11, [1], [0]),      # 식생손실 vs 안정 (식생증가·기타변화 제외)
    'HANSEN_LOSS': (12, [1], [0]),
}
# (정답 과제, 점수) 평가 조합
PAIRS = [
    ('CHG_Y1Y2', 'emb_Y1Y2'),
    ('CHG_Y2Y3', 'emb_Y2Y3'),
    ('CHG_Y1Y3', 'emb_Y1Y3'),
    ('CHG_Y1Y3', 'absdNDVI'),
    ('CHG_Y1Y3', 'absdNDMI'),
    ('VEG_LOSS', 'emb_Y1Y3'),
    ('VEG_LOSS', 'negdNDVI'),
    ('HANSEN_LOSS', 'emb_Y1Y3'),
    ('HANSEN_LOSS', 'negdNDVI'),
]
PRIMARY = ('CHG_Y1Y3', 'emb_Y1Y3')

trapz = getattr(np, 'trapezoid', None) or np.trapz


# ------------------------------------------------------------------
# 1. 파일 짝 맞추기 / 격자 확인
# ------------------------------------------------------------------
def expand(patterns):
    out = []
    for p in patterns:
        m = sorted(glob.glob(p))
        out.extend(m if m else [p])
    out = [str(Path(p).resolve()) for p in dict.fromkeys(out)]
    missing = [p for p in out if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'파일 없음: {missing}')
    return out


def pair_files(inputs, labels):
    """GEE 분할 파일(…_input15-0000-0000.tif ↔ …_labels-0000-0000.tif)을 이름으로 짝지음."""
    if len(inputs) == 1 and len(labels) == 1:
        return [(inputs[0], labels[0])]
    by_key = {Path(p).name.replace('_labels', ''): p for p in labels}
    pairs = []
    for p in inputs:
        key = Path(p).name.replace('_input15', '')
        if key not in by_key:
            raise ValueError(f'짝이 되는 정답지를 찾지 못함: {Path(p).name} '
                             f'(기대 이름: {key.replace(".tif", "")}의 _labels 버전)')
        pairs.append((p, by_key[key]))
    return pairs


def same_grid(a, b):
    return (a.crs == b.crs and a.width == b.width and a.height == b.height
            and a.transform.almost_equals(b.transform, precision=1e-6))


def open_labels_on(ref, label_path):
    """정답지를 ref 격자로 엶. 격자가 다르면 최근린 리샘플 VRT 로 맞춤."""
    src = rasterio.open(label_path)
    if src.count < LABEL_BANDS:
        src.close()
        raise ValueError(f'{label_path}: 정답지 밴드 수 {src.count} (13 필요)')
    if same_grid(src, ref):
        return src, [src]
    print(f'  [경고] 격자 불일치 → 최근린으로 재투영: {Path(label_path).name}')
    vrt = WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                    width=ref.width, height=ref.height,
                    resampling=Resampling.nearest, nodata=LABEL_NODATA)
    return vrt, [vrt, src]


# ------------------------------------------------------------------
# 2-3. 학습 / 추론
# ------------------------------------------------------------------
def run_train(args, inputs):
    outdir = Path(args.outdir) / 'model'
    targs = Namespace(
        inputs=inputs, outdir=str(outdir), years=args.years,
        epochs=args.epochs, steps=args.steps, batch=args.batch,
        patch=args.patch, dim=args.dim, lr=args.lr,
        mask_ratio=args.mask_ratio, mask_block=args.mask_block,
        workers=args.workers, device=args.device, seed=args.seed)
    gfm.train(targs)
    return outdir / 'best_model.pt'


def run_infer(args, ckpt, input_path):
    out = Path(args.outdir) / 'infer' / Path(input_path).stem
    prefix = gfm.output_prefix(input_path, getattr(args, 'output_prefix', None))
    score = out / f'{prefix}change_scores.tif'
    if score.exists() and not args.overwrite:
        print(f'  추론 결과 재사용: {score}')
        return score
    gfm.infer(Namespace(input=input_path, checkpoint=str(ckpt), outdir=str(out),
                        tile=args.tile, halo=args.halo, device=args.device,
                        save_embedding=args.save_embedding,
                        output_prefix=getattr(args, 'output_prefix', None)))
    return score


# ------------------------------------------------------------------
# 4. 평가
# ------------------------------------------------------------------
def iter_windows(jobs, size=1024):
    """jobs: [(score_path, label_path)] → (score[5,h,w], label[13,h,w], rows, cols)"""
    for score_path, label_path in jobs:
        with rasterio.open(score_path) as sc_ds:
            lab_ds, closers = open_labels_on(sc_ds, label_path)
            try:
                for win in gfm.tiles_for_inference(sc_ds.width, sc_ds.height, size):
                    sc = sc_ds.read(window=win).astype('float32')
                    lab = lab_ds.read(window=win)
                    rows = np.arange(win.row_off, win.row_off + win.height)[:, None]
                    cols = np.arange(win.col_off, win.col_off + win.width)[None, :]
                    yield score_path, win, sc, lab, rows, cols
            finally:
                for c in closers:
                    c.close()


def score_of(sc, name):
    band, mode = SCORES[name]
    raw = sc[band - 1]
    ok = np.isfinite(raw) & (raw != SCORE_NODATA)
    v = raw.copy()
    if mode == 'abs':
        v = np.abs(v)
    elif mode == 'neg':
        v = -v
    return v, ok


def task_of(lab, task):
    band, pos, neg = TASKS[task]
    l = lab[band - 1]
    return np.isin(l, pos), np.isin(l, neg)


def estimate_ranges(jobs, rng, per_win=4000):
    samples = {k: [] for k in SCORES}
    for _, _, sc, _, _, _ in iter_windows(jobs):
        for name in SCORES:
            v, ok = score_of(sc, name)
            v = v[ok]
            if v.size:
                take = rng.choice(v.size, min(per_win, v.size), replace=False)
                samples[name].append(v[take])
    ranges = {}
    for name, arrs in samples.items():
        if not arrs:
            ranges[name] = (0.0, 1.0)
            continue
        v = np.concatenate(arrs)
        lo, hi = np.percentile(v, [0.1, 99.9])
        ranges[name] = (float(lo), float(hi if hi > lo else lo + 1e-6))
    return ranges


def build_histograms(jobs, ranges, block):
    H = {p: {s: {'pos': np.zeros(NBINS, np.int64), 'neg': np.zeros(NBINS, np.int64)}
             for s in ('cal', 'test')} for p in PAIRS}
    for _, _, sc, lab, rows, cols in iter_windows(jobs):
        test = ((rows // block + cols // block) % 2).astype(bool)
        test = np.broadcast_to(test, sc.shape[1:])
        for task, sname in PAIRS:
            v, ok = score_of(sc, sname)
            pos, neg = task_of(lab, task)
            lo, hi = ranges[sname]
            idx = np.clip(((v - lo) / (hi - lo) * NBINS).astype(np.int64), 0, NBINS - 1)
            for split, smask in (('cal', ~test), ('test', test)):
                for cls, cmask in (('pos', pos), ('neg', neg)):
                    m = ok & cmask & smask
                    if m.any():
                        H[(task, sname)][split][cls] += np.bincount(idx[m], minlength=NBINS)
    return H


def cum_counts(pos, neg):
    """bin k 이상을 양성으로 예측할 때 TP/FP (k = 0..NBINS-1)."""
    tp = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    return tp, fp, float(pos.sum()), float(neg.sum())


def auroc_ap(pos, neg):
    tp, fp, P, N = cum_counts(pos, neg)
    if P == 0 or N == 0:
        return float('nan'), float('nan')
    tpr = np.concatenate([[0.0], (tp / P)[::-1]])
    fpr = np.concatenate([[0.0], (fp / N)[::-1]])
    auc = float(trapz(tpr, fpr))
    rec = tpr
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.concatenate([[1.0], (tp / np.maximum(tp + fp, 1))[::-1]])
    ap = float(np.sum(np.diff(rec) * prec[1:]))
    return auc, ap


def binary_metrics(tp, fp, P, N):
    fn, tn = P - tp, N - fp
    n = P + N
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / P if P else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    oa = (tp + tn) / n if n else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0.0
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    return dict(precision=prec, recall=rec, f1=f1, iou=iou, oa=oa, kappa=kappa,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def evaluate_pair(h, rng_range):
    cal, test = h['cal'], h['test']
    allpos, allneg = cal['pos'] + test['pos'], cal['neg'] + test['neg']
    auc, ap = auroc_ap(allpos, allneg)
    res = dict(n_pos=int(allpos.sum()), n_neg=int(allneg.sum()),
               prevalence=float(allpos.sum() / max(1, allpos.sum() + allneg.sum())),
               auroc=auc, ap=ap)
    tp_c, fp_c, P_c, N_c = cum_counts(cal['pos'], cal['neg'])
    tp_t, fp_t, P_t, N_t = cum_counts(test['pos'], test['neg'])
    if P_c == 0 or N_c == 0 or P_t == 0 or N_t == 0:
        res.update(threshold=float('nan'), note='양성 또는 음성 화소 부족')
        return res
    f1_c = 2 * tp_c / np.maximum(2 * tp_c + fp_c + (P_c - tp_c), 1)
    k = int(np.argmax(f1_c))
    lo, hi = rng_range
    res['threshold'] = float(lo + k * (hi - lo) / NBINS)
    res['cal_f1'] = float(f1_c[k])
    res.update({f'test_{key}': val for key, val in
                binary_metrics(tp_t[k], fp_t[k], P_t, N_t).items()})
    return res


def write_eval_maps(jobs, outroot, threshold):
    """PRIMARY 조합의 예측/오류 지도. band1 예측(0/1), band2 오류(0 TN,1 FP,2 FN,3 TP), 255=제외"""
    task, sname = PRIMARY
    paths = []
    for score_path, label_path in jobs:
        prefix = Path(score_path).name[:-len('change_scores.tif')]
        out = Path(score_path).parent / f'{prefix}eval_map_{task}_{sname}.tif'
        with rasterio.open(score_path) as sc_ds:
            prof = sc_ds.profile.copy()
            prof.update(count=2, dtype='uint8', nodata=255, compress='deflate', predictor=1)
            with rasterio.open(out, 'w', **prof) as dst:
                dst.descriptions = ('Pred_change', 'Error_TN0_FP1_FN2_TP3')
                for _, win, sc, lab, _, _ in iter_windows([(score_path, label_path)]):
                    v, ok = score_of(sc, sname)
                    pos, neg = task_of(lab, task)
                    pred = (v >= threshold)
                    b1 = np.full(v.shape, 255, np.uint8)
                    b1[ok] = pred[ok]
                    b2 = np.full(v.shape, 255, np.uint8)
                    m = ok & (pos | neg)
                    b2[m & neg & ~pred] = 0
                    b2[m & neg & pred] = 1
                    b2[m & pos & ~pred] = 2
                    b2[m & pos & pred] = 3
                    dst.write(np.stack([b1, b2]), window=win)
        paths.append(out)
    return paths


def evaluate(args, jobs):
    evaldir = Path(args.outdir) / 'eval'
    evaldir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    print('\n[평가] 점수 범위 추정 …')
    ranges = estimate_ranges(jobs, rng)
    print('[평가] 히스토그램 누적 …')
    H = build_histograms(jobs, ranges, args.eval_block)
    rows = []
    for task, sname in PAIRS:
        r = evaluate_pair(H[(task, sname)], ranges[sname])
        rows.append(dict(task=task, score=sname, **r))

    keys = ['task', 'score', 'n_pos', 'n_neg', 'prevalence', 'auroc', 'ap', 'threshold',
            'cal_f1', 'test_f1', 'test_precision', 'test_recall', 'test_iou',
            'test_kappa', 'test_oa', 'test_tp', 'test_fp', 'test_fn', 'test_tn', 'note']
    with open(evaldir / 'metrics.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    (evaldir / 'metrics.json').write_text(
        json.dumps({'ranges': ranges, 'eval_block': args.eval_block, 'results': rows},
                   indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n' + '=' * 92)
    print(f'{"task":<12}{"score":<10}{"n_pos":>9}{"n_neg":>11}{"AUROC":>8}{"AP":>8}'
          f'{"thr":>9}{"F1":>7}{"IoU":>7}{"Kappa":>8}')
    print('-' * 92)
    for r in rows:
        def g(k, fmt='.3f'):
            v = r.get(k, float('nan'))
            return format(v, fmt) if isinstance(v, (int, float)) and np.isfinite(v) else '-'
        print(f'{r["task"]:<12}{r["score"]:<10}{r["n_pos"]:>9}{r["n_neg"]:>11}'
              f'{g("auroc"):>8}{g("ap"):>8}{g("threshold", ".4f"):>9}'
              f'{g("test_f1"):>7}{g("test_iou"):>7}{g("test_kappa"):>8}')
    print('=' * 92)
    print('F1/IoU/Kappa: 짝수 공간블록에서 정한 임계값을 홀수 블록에 적용한 값')

    prim = next(r for r in rows if (r['task'], r['score']) == PRIMARY)
    if np.isfinite(prim.get('threshold', float('nan'))):
        maps = write_eval_maps(jobs, evaldir, prim['threshold'])
        for p in maps:
            print('오류 지도:', p)
    print('지표 파일:', evaldir / 'metrics.csv')
    return rows


# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Mini-GFM 입력→학습→추론→정답지 평가 파이프라인')
    p.add_argument('--inputs', nargs='+', required=True, help='15밴드 입력 GeoTIFF (glob 가능)')
    p.add_argument('--labels', nargs='+', required=True, help='GEE 정답지 GeoTIFF (glob 가능)')
    p.add_argument('--outdir', default='runs/pipeline')
    p.add_argument('--checkpoint', default=None, help='있으면 학습 생략')
    p.add_argument('--overwrite', action='store_true', help='기존 추론 결과 덮어쓰기')
    p.add_argument('--skip-eval', action='store_true')
    # 학습
    p.add_argument('--years', nargs=3, type=int, default=[2023, 2024, 2025])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--patch', type=int, default=128)
    p.add_argument('--dim', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mask-ratio', type=float, default=0.35)
    p.add_argument('--mask-block', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    # 추론
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--halo', type=int, default=16)
    p.add_argument('--save-embedding', action='store_true')
    p.add_argument('--output-prefix', default=None, help='Filename prefix (default: input stem)')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    # 평가
    p.add_argument('--eval-block', type=int, default=256,
                   help='임계값 결정/검증 분할용 공간 블록 크기(화소)')
    args = p.parse_args()

    if not 0.05 <= args.mask_ratio <= 0.9:
        p.error('--mask-ratio 는 0.05~0.9')
    if args.tile < 32 or args.halo < 8:
        p.error('--tile >= 32, --halo >= 8')

    inputs = expand(args.inputs)
    labels = expand(args.labels)
    pairs = pair_files(inputs, labels)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print('[1] 입력/정답지 확인')
    for ip, lp in pairs:
        gfm.check_raster(ip)
        with rasterio.open(ip) as a, rasterio.open(lp) as b:
            status = '격자 일치' if same_grid(a, b) else '격자 불일치(평가 시 재투영)'
            print(f'  {Path(ip).name}  ↔  {Path(lp).name}  [{a.count}밴드 / {b.count}밴드, {status}]')

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        print(f'[2] 학습 생략, 체크포인트 사용: {ckpt}')
    else:
        print('[2] 자기지도 사전학습 (정답지는 사용하지 않음)')
        ckpt = run_train(args, inputs)

    print('[3] 추론')
    jobs = []
    for ip, lp in pairs:
        jobs.append((str(run_infer(args, ckpt, ip)), lp))

    if not args.skip_eval:
        print('[4] 정답지 비교')
        evaluate(args, jobs)


if __name__ == '__main__':
    main()
