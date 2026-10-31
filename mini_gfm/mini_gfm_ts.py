#!/usr/bin/env python3
# coding: utf-8
"""Mini-GFM 다연도판: 춘천 Sentinel-2 42밴드(2019~2025, 7년 × 6밴드) 산림변화 탐지.

mini_gfm.py(3년 18밴드)의 구조를 N년으로 일반화하고, 자기지도 사전학습 뒤 라벨로 변화 분류기를 학습한다.
  1) 입력: 연도별 [NDVI, NDWI, NDBI, NDMI, Slope, Elevation] × N  →  논리 밴드 [NDVI, NDWI, NDMI] × N + Slope + Elevation
     (NDBI = −NDMI 이므로 제외, 지형은 첫 해 것만 사용 — mini_gfm 과 동일)
  2) 사전학습: 연도 공유 CNN 인코더 + 시간 결합 디코더, 공간 블록 마스크 복원(L1). 연도 전체를 가리는 마스크도 섞어
     시간 문맥으로 복원하도록 함. 학습/검증은 512 px 공간 블록 단위로 분리
  3) 추론(change_scores.tif): 연속 연도 임베딩 거리 N−1개, 첫해→끝해 거리, 최대 연속 거리, 최대 변화 연도,
     ΔNDVI·ΔNDMI(끝−첫), 최대 NDVI 감소량과 그 연도
  4) 변화 분류기: 라벨 래스터의 학습 라벨로 화소 분류기 학습
     - 새 형식(build_training_label_rasters.py): 경영실적·임상도 합의로 거른 TRAIN_P1/P2 (--period 로 구간 선택)
     - 이전 형식(make_s2_label_raster.py): 임상도 2019→2025 변화 FC_TYPE
     (HistGradientBoosting, 공간 블록 홀드아웃). 특징 세트 비교: 지수만 / 지수+GFM 임베딩
  5) 평가: AUROC·AP(변화 여부, 벌채), 국유림 산림사업실적(MGMT, 춘천_임소반_산림사업실적.gpkg 2019~2025)과의 일치,
     변화 연도 ±1년 정확도
     → change_class.tif (0 무변화, 1 벌채·갱신, 2 산지전용, 3 산림 증가) + change_prob.tif

Colab:
  %run mini_gfm_ts.py all --input /content/data/Chuncheon_S2_2019_2025_42Bands.tif \
       --labels /content/data/Chuncheon_labels_S2_20m.tif --period P2 --outdir /content/runs/chuncheon --epochs 30
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn

NODATA = -9999.0
FIRST_YEAR = 2019
FC_NAMES = {0: "no_change", 1: "harvest", 2: "conversion", 3: "gain"}
# 합의 라벨(CONS) → 실적 기준 평가 라벨(MGMT): 0 실적없음, 1 수확·갱신 실적, 2 솎아베기 실적, 3 피해·기타 실적
CONS_TO_MGMT = {0: 0, 2: 0, 1: 1, 3: 1, 4: 2, 5: 3}


def read_labels(path, period):
    """라벨 래스터 → {FC_TYPE, FC_YEAR, MGMT, SOURCE}.
    새 형식(build_training_label_rasters.py: TRAIN_P1/P2, YEAR_P*, CONS_P*, SOURCE)과
    이전 형식(make_s2_label_raster.py: FC_TYPE, FC_YEAR, MGMT, MGMT_YEAR)을 모두 읽는다."""
    with rasterio.open(path) as lb:
        bands = {d: lb.read(i + 1) for i, d in enumerate(lb.descriptions)}
        size = (lb.width, lb.height)
    if f"TRAIN_{period}" in bands:
        cons = bands[f"CONS_{period}"]
        mgmt = np.full(cons.shape, 255, np.uint8)
        for c, m in CONS_TO_MGMT.items():
            mgmt[cons == c] = m
        labels = {"FC_TYPE": bands[f"TRAIN_{period}"], "FC_YEAR": bands[f"YEAR_{period}"],
                  "MGMT": mgmt, "SOURCE": bands["SOURCE"]}
        fmt = f"새 형식 {period} (TRAIN·YEAR·CONS·SOURCE)"
    else:
        labels = {k: bands[k] for k in ("FC_TYPE", "FC_YEAR", "MGMT")}
        labels["SOURCE"] = np.where(bands["MGMT"] != 255, 1, 2).astype(np.uint8)
        fmt = "이전 형식 (FC_TYPE·FC_YEAR·MGMT)"
    return labels, size, fmt


# ------------------------------------------------------------------ 입력

def load_stack(path):
    """42밴드 → 논리 배열 [3N+2, H, W] (float32, 무효 NaN) + 유효 마스크."""
    with rasterio.open(path) as ds:
        if ds.count % 6:
            raise ValueError(f"6의 배수 밴드여야 함: {ds.count}")
        n = ds.count // 6
        desc = ds.descriptions
        expected = ["NDVI_", "NDWI_", "NDBI_", "NDMI_", "Slope_", "Elevation_"] * n
        if all(desc):
            bad = [(i + 1, d) for i, (d, e) in enumerate(zip(desc, expected)) if not str(d).startswith(e)]
            if bad:
                raise ValueError(f"밴드 순서가 GEE 내보내기와 다름: {bad[:5]}")
            years = [int(desc[6 * y].split("_")[1]) for y in range(n)]
        else:
            years = list(range(FIRST_YEAR, FIRST_YEAR + n))
        idx = [6 * y + b + 1 for y in range(n) for b in (0, 1, 3)] + [5, 6]
        arr = ds.read(idx, masked=True).astype("float32").filled(np.nan)
        meta = {"profile": ds.profile.copy(), "years": years}
    valid = np.isfinite(arr).all(axis=0)
    valid &= (np.abs(arr[:3 * n]) <= 1.05).all(axis=0)
    valid &= (arr[3 * n] >= 0) & (arr[3 * n] <= 90) & (arr[3 * n + 1] > -500)
    return arr, valid, n, meta


def normalize(arr, valid, n, stats):
    x = np.zeros_like(arr, dtype=np.float32)
    x[:3 * n] = np.clip(arr[:3 * n], -1, 1) * 2.0
    x[3 * n] = np.clip((arr[3 * n] - stats["slope_mean"]) / stats["slope_std"], -5, 5)
    x[3 * n + 1] = np.clip((arr[3 * n + 1] - stats["elev_mean"]) / stats["elev_std"], -5, 5)
    x[:, ~valid] = 0.0
    x[~np.isfinite(x)] = 0.0
    return x


def block_split(h, w, block, seed, val_frac=0.2):
    """512 px 공간 블록 단위 학습/검증(=홀드아웃) 분할 마스크. True = 홀드아웃 블록."""
    by, bx = math.ceil(h / block), math.ceil(w / block)
    rng = np.random.default_rng(seed)
    hold = rng.random((by, bx)) < val_frac
    return np.kron(hold, np.ones((block, block), bool))[:h, :w]


# ------------------------------------------------------------------ 모델 (mini_gfm.TemporalMaskedAE 의 N년 일반화)

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(4, c), nn.GELU(),
                                 nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(4, c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalMaskedAE(nn.Module):
    def __init__(self, n_years, dim=24):
        super().__init__()
        self.n = n_years
        self.encoder = nn.Sequential(nn.Conv2d(6, dim, 3, padding=1), nn.GELU(),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.decoder = nn.Sequential(nn.Conv2d(n_years * dim, dim * 2, 3, padding=1), nn.GELU(),
                                     ResidualBlock(dim * 2), nn.Conv2d(dim * 2, 3 * n_years, 1))

    def encode_years(self, x, mask=None):
        n = self.n
        topo = x[:, 3 * n:3 * n + 2]
        emb = []
        for y in range(n):
            idx = x[:, 3 * y:3 * y + 3]
            m = torch.zeros_like(idx[:, :1]) if mask is None else mask[:, y:y + 1]
            emb.append(self.encoder(torch.cat((idx * (1 - m), topo, m), dim=1)))
        return emb

    def forward(self, x, mask):
        return self.decoder(torch.cat(self.encode_years(x, mask), dim=1))


def make_mask(b, n, hw, ratio, block, year_drop, device):
    h, w = hw
    small = torch.rand(b, n, math.ceil(h / block), math.ceil(w / block), device=device) < ratio
    m = nn.functional.interpolate(small.float(), size=(h, w), mode="nearest")
    drop = (torch.rand(b, n, 1, 1, device=device) < year_drop).float()  # 연도 전체 가리기
    return torch.maximum(m, drop)


def masked_l1(pred, x, valid, mask, n):
    use = mask.repeat_interleave(3, dim=1) * valid
    return (torch.abs(pred - x[:, :3 * n]) * use).sum() / use.sum().clamp_min(1.0)


# ------------------------------------------------------------------ 1. 사전학습

def sample_patches(x, valid, region_ok, patch, k, rng):
    h, w = valid.shape
    out_x, out_v = [], []
    tries = 0
    while len(out_x) < k and tries < k * 50:
        tries += 1
        r, c = rng.integers(0, h - patch), rng.integers(0, w - patch)
        if region_ok[r:r + patch, c:c + patch].mean() < 0.9 or valid[r:r + patch, c:c + patch].mean() < 0.5:
            continue
        xx, vv = x[:, r:r + patch, c:c + patch], valid[r:r + patch, c:c + patch]
        rot = rng.integers(4)
        xx, vv = np.rot90(xx, rot, axes=(1, 2)), np.rot90(vv, rot)
        if rng.random() < 0.5:
            xx, vv = xx[:, :, ::-1], vv[:, ::-1]
        out_x.append(np.ascontiguousarray(xx))
        out_v.append(np.ascontiguousarray(vv[None]))
    return torch.from_numpy(np.stack(out_x)), torch.from_numpy(np.stack(out_v).astype("float32"))


def pretrain(args, x, valid, n, hold, device, outdir):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TemporalMaskedAE(n, args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    val_x, val_v = sample_patches(x, valid, hold, args.patch, 32, np.random.default_rng(0))
    best, history = float("inf"), []
    for ep in range(1, args.epochs + 1):
        model.train()
        t, losses = time.time(), []
        for _ in range(args.steps):
            xb, vb = sample_patches(x, valid, ~hold, args.patch, args.batch, rng)
            xb, vb = xb.to(device), vb.to(device)
            m = make_mask(xb.shape[0], n, xb.shape[-2:], args.mask_ratio, args.mask_block, args.year_drop, device)
            loss = masked_l1(model(xb, m), xb, vb, m, n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(1)
            vl = []
            for i in range(0, len(val_x), args.batch):
                xb, vb = val_x[i:i + args.batch].to(device), val_v[i:i + args.batch].to(device)
                small = torch.rand(xb.shape[0], n, math.ceil(args.patch / args.mask_block),
                                   math.ceil(args.patch / args.mask_block), device=device, generator=g)
                m = nn.functional.interpolate((small < args.mask_ratio).float(), size=xb.shape[-2:], mode="nearest")
                vl.append(masked_l1(model(xb, m), xb, vb, m, n).item())
        tr, va = float(np.mean(losses)), float(np.mean(vl))
        history.append({"epoch": ep, "train_mae": tr, "val_mae": va})
        flag = ""
        if va < best:
            best = va
            torch.save({"model_state": model.state_dict(), "n_years": n, "dim": args.dim,
                        "stats": args.stats, "best_val_masked_mae": best}, outdir / "best_model.pt")
            flag = " *saved"
        print(f"[pretrain] epoch {ep:03d}/{args.epochs} masked MAE train {tr:.4f} val {va:.4f} "
              f"({time.time() - t:.0f}s){flag}", flush=True)
    (outdir / "history.json").write_text(json.dumps(history, indent=2))
    return best


# ------------------------------------------------------------------ 2. 추론 (전체 영상 타일 처리)

def infer(model, x, valid, n, years, tile, halo, device):
    """반환: scores[dict 이름→H×W], emb_delta[dim,H,W](끝해−첫해 임베딩)."""
    _, h, w = x.shape
    names = [f"emb_d{years[i]}_{years[i + 1]}" for i in range(n - 1)] + [
        "emb_first_last", "emb_max_consec", "emb_max_year", "dNDVI", "dNDMI", "ndvi_max_drop", "ndvi_drop_year"]
    scores = {k: np.full((h, w), np.nan, np.float32) for k in names}
    dim = model.encoder[0].out_channels
    emb_delta = np.zeros((dim, h, w), np.float16)
    xp = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))
    model.eval()
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            th, tw = min(tile, h - r), min(tile, w - c)
            xt = torch.from_numpy(xp[None, :, r:r + th + 2 * halo, c:c + tw + 2 * halo]).to(device)
            with torch.inference_mode():
                emb = [e[0, :, halo:halo + th, halo:halo + tw] for e in model.encode_years(xt)]
                d = torch.stack([(emb[i + 1] - emb[i]).square().mean(0).sqrt() for i in range(n - 1)])
                fl = (emb[-1] - emb[0]).square().mean(0).sqrt()
            d, fl = d.cpu().numpy(), fl.cpu().numpy()
            win = (slice(r, r + th), slice(c, c + tw))
            for i in range(n - 1):
                scores[names[i]][win] = d[i]
            scores["emb_first_last"][win] = fl
            scores["emb_max_consec"][win] = d.max(0)
            scores["emb_max_year"][win] = np.asarray(years[1:])[d.argmax(0)]
            emb_delta[(slice(None), *win)] = (emb[-1] - emb[0]).cpu().numpy()
    ndvi = x[0:3 * n:3] / 2.0
    ndmi = x[2:3 * n:3] / 2.0
    scores["dNDVI"] = ndvi[-1] - ndvi[0]
    scores["dNDMI"] = ndmi[-1] - ndmi[0]
    steps = np.diff(ndvi, axis=0)
    scores["ndvi_max_drop"] = -steps.min(0)
    scores["ndvi_drop_year"] = np.asarray(years[1:], np.float32)[steps.argmin(0)]
    for k in scores:
        scores[k][~valid] = np.nan
    return scores, emb_delta


def write_tif(path, profile, bands: dict, dtype="float32", nodata=NODATA):
    p = profile.copy()
    p.update(driver="GTiff", count=len(bands), dtype=dtype, nodata=nodata, compress="deflate",
             tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **p) as dst:
        for i, (k, v) in enumerate(bands.items(), 1):
            dst.write(np.where(np.isfinite(v), v, nodata).astype(dtype), i)
            dst.set_band_description(i, k)


# ------------------------------------------------------------------ 3. 분류기·평가

def features(x, n, scores, emb_delta, with_gfm):
    """화소 특징 행렬 [H*W, F]. 지수 기반(원시 지수·연속 차분·지형) + 선택적으로 GFM 임베딩 특징."""
    ndvi, ndwi, ndmi = x[0:3 * n:3] / 2, x[1:3 * n:3] / 2, x[2:3 * n:3] / 2
    feats = [ndvi, ndwi, ndmi, np.diff(ndvi, axis=0), np.diff(ndmi, axis=0), x[3 * n:3 * n + 2],
             np.stack([scores["dNDVI"], scores["dNDMI"], scores["ndvi_max_drop"]])]
    names = ([f"ndvi{i}" for i in range(n)] + [f"ndwi{i}" for i in range(n)] + [f"ndmi{i}" for i in range(n)]
             + [f"dndvi{i}" for i in range(n - 1)] + [f"dndmi{i}" for i in range(n - 1)]
             + ["slope", "elev", "dNDVI", "dNDMI", "ndvi_max_drop"])
    if with_gfm:
        dist = np.stack([v for k, v in scores.items() if k.startswith("emb_") and k != "emb_max_year"])
        feats += [dist, emb_delta.astype(np.float32)]
        names += [k for k in scores if k.startswith("emb_") and k != "emb_max_year"]
        names += [f"embdelta{i}" for i in range(emb_delta.shape[0])]
    f = np.concatenate([a.reshape(a.shape[0], -1) for a in feats]).T
    return f.astype(np.float32), names


def auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(s)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y[ok], s[ok])), float(average_precision_score(y[ok], s[ok]))


def evaluate_scores(scores, labels, hold):
    """비지도 점수의 변화 탐지력 (홀드아웃 블록 화소)."""
    fc, mg, nat = labels["FC_TYPE"], labels["MGMT"], labels["SOURCE"] == 1
    tasks = {
        "임상도 변화여부(벌채·전용·증가 vs 무변화)": (np.isin(fc, [1, 2, 3]), fc == 0),
        "국유림 검증라벨 변화여부(실적·임상도 합의)": (np.isin(fc, [1, 2, 3]) & nat, (fc == 0) & nat),
        "임상도 벌채·갱신 vs 무변화": (fc == 1, fc == 0),
        "국유림 실적 수확·갱신 vs 실적없음": (mg == 1, mg == 0),
        "국유림 실적 솎아베기 vs 실적없음": (mg == 2, mg == 0),
    }
    cands = {"GFM emb_first_last": scores["emb_first_last"], "GFM emb_max_consec": scores["emb_max_consec"],
             "지수 |ΔNDVI|": np.abs(scores["dNDVI"]), "지수 −ΔNDVI": -scores["dNDVI"],
             "지수 최대 NDVI 감소": scores["ndvi_max_drop"]}
    rows = []
    for task, (pos, neg) in tasks.items():
        m = hold & (pos | neg)
        y = pos[m].astype(int)
        for name, s in cands.items():
            auc, ap = auc_ap(y, s[m])
            rows.append({"task": task, "score": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                         "prevalence": round(float(y.mean()), 4) if len(y) else None,
                         "auroc": round(auc, 4), "ap": round(ap, 4)})
    return rows


def train_classifier(F, names, labels, hold, seed, max_per_class=60000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
    fc = labels["FC_TYPE"].ravel()
    tr = (~hold.ravel()) & (fc != 255) & np.isfinite(F).all(1)
    te = hold.ravel() & (fc != 255) & np.isfinite(F).all(1)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.where(tr & (fc == k))[0])[:max_per_class] for k in FC_NAMES])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                         l2_regularization=1.0, class_weight="balanced", random_state=seed)
    clf.fit(F[idx], fc[idx])
    pred = clf.predict(F[te])
    proba = clf.predict_proba(F[te])
    changed_true = np.isin(fc[te], [1, 2, 3]).astype(int)
    p_changed = proba[:, [list(clf.classes_).index(k) for k in (1, 2, 3)]].sum(1)
    auc, ap = auc_ap(changed_true, p_changed)
    nat = labels["SOURCE"].ravel()[te] == 1
    v_auc, v_ap = auc_ap(changed_true[nat], p_changed[nat])
    rep = classification_report(fc[te], pred, labels=list(FC_NAMES), target_names=list(FC_NAMES.values()),
                                output_dict=True, zero_division=0)
    metrics = {"n_train": int(len(idx)), "n_test": int(te.sum()),
               "balanced_accuracy": round(float(balanced_accuracy_score(fc[te], pred)), 4),
               "macro_f1": round(float(f1_score(fc[te], pred, average="macro")), 4),
               "changed_auroc": round(auc, 4), "changed_ap": round(ap, 4),
               "verified_changed_auroc": round(v_auc, 4), "verified_changed_ap": round(v_ap, 4),
               "n_test_verified": int(nat.sum()),
               "per_class": {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score", "support")}
                             for k, v in rep.items() if k in FC_NAMES.values()}}
    return clf, metrics


def year_accuracy(scores, labels, cls_map, hold):
    """임상도 갱신년도(약한 라벨)와 GFM 최대 변화 연도 비교 (변화로 분류·라벨된 홀드아웃 화소)."""
    fy = labels["FC_YEAR"].astype(float)
    fy[fy == 255] = np.nan
    fy += 2000
    m = hold & np.isfinite(fy) & np.isin(labels["FC_TYPE"], [1, 2, 3]) & np.isfinite(scores["emb_max_year"])
    out = {"n": int(m.sum())}
    for k in ("emb_max_year", "ndvi_drop_year"):
        diff = np.abs(scores[k][m] - fy[m])
        out[k] = {"exact": round(float((diff == 0).mean()), 4), "within_1yr": round(float((diff <= 1).mean()), 4)}
    return out


def mgmt_agreement(cls_map, labels):
    """국유림 산림사업실적(2019~2025) 소반 화소에서 모델이 변화(1~3)로 본 비율."""
    mg = labels["MGMT"]
    return {name: round(float(np.isin(cls_map[mg == v], [1, 2, 3]).mean()), 4) if (mg == v).any() else None
            for v, name in ((1, "수확·갱신 실적 소반"), (2, "솎아베기 실적 소반"),
                            (3, "기타(시설·소득·피해복구) 실적 소반"), (0, "실적 없는 소반"))}


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["all", "pretrain", "evaluate"])
    p.add_argument("--input", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--period", choices=["P1", "P2"], default="P2",
                   help="새 형식 라벨의 비교 구간: P1 2013→2019, P2 2019→2025 (이전 형식이면 무시)")
    p.add_argument("--outdir", default="runs/chuncheon")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--dim", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--mask-block", type=int, default=16)
    p.add_argument("--year-drop", type=float, default=0.15, help="연도 전체를 가릴 확률")
    p.add_argument("--block", type=int, default=512, help="공간 홀드아웃 블록 크기(px)")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args, _ = p.parse_known_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    arr, valid, n, meta = load_stack(args.input)
    years = meta["years"]
    h, w = valid.shape
    print(f"[입력] {Path(args.input).name}: {n}년 {years}, {w}×{h} px, 유효 {valid.mean():.1%}, device {device}")
    vs = valid
    args.stats = {"slope_mean": float(np.nanmean(arr[3 * n][vs])), "slope_std": float(max(1, np.nanstd(arr[3 * n][vs]))),
                  "elev_mean": float(np.nanmean(arr[3 * n + 1][vs])), "elev_std": float(max(1, np.nanstd(arr[3 * n + 1][vs])))}
    x = normalize(arr, valid, n, args.stats)
    del arr
    hold = block_split(h, w, args.block, args.seed)
    labels, lsize, fmt = read_labels(args.labels, args.period)
    if lsize != (w, h):
        raise ValueError(f"라벨 래스터 격자 {lsize} 가 입력 {(w, h)} 과 다름 — build_training_label_rasters.py 로 다시 만드세요")
    fc = labels["FC_TYPE"]
    print(f"[라벨] {fmt}: " + ", ".join(f"{v} {int((fc == k).sum()):,}" for k, v in FC_NAMES.items())
          + f" (국유림 검증 {int(((fc != 255) & (labels['SOURCE'] == 1)).sum()):,} / 국유림 밖 "
          f"{int(((fc != 255) & (labels['SOURCE'] == 2)).sum()):,})")
    print(f"[분할] 홀드아웃 블록 화소 {hold[valid].mean():.1%} (블록 {args.block}px)")

    ckpt = outdir / "best_model.pt"
    if args.mode in ("all", "pretrain"):
        best = pretrain(args, x, valid, n, hold, device, outdir)
        print(f"[pretrain] 완료 best val masked MAE {best:.4f}")
        if args.mode == "pretrain":
            return
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = TemporalMaskedAE(state["n_years"], state["dim"])
    model.load_state_dict(state["model_state"])
    model.to(device)

    print("[infer] 전체 영상 변화점수 계산 …", flush=True)
    scores, emb_delta = infer(model, x, valid, n, years, args.tile, args.halo, device)
    write_tif(outdir / "change_scores.tif", meta["profile"], scores)

    print("[eval] 비지도 변화점수 평가 …", flush=True)
    score_rows = evaluate_scores(scores, labels, hold)

    results = {}
    for tag, with_gfm in (("지수만", False), ("지수+GFM", True)):
        F, names = features(x, n, scores, emb_delta, with_gfm)
        clf, met = train_classifier(F, names, labels, hold, args.seed)
        met["n_features"] = len(names)
        results[tag] = met
        print(f"[classifier] {tag}: 특징 {len(names)}개, balanced acc {met['balanced_accuracy']}, "
              f"macro F1 {met['macro_f1']}, 변화 AUROC {met['changed_auroc']}, AP {met['changed_ap']} | "
              f"국유림 검증 AUROC {met['verified_changed_auroc']}, AP {met['verified_changed_ap']}", flush=True)
        if with_gfm:
            ok = valid.ravel() & np.isfinite(F).all(1)
            prob = np.full((len(FC_NAMES), h * w), np.nan, np.float32)
            prob[:, ok] = clf.predict_proba(F[ok]).T
            cls_map = np.full(h * w, 255, np.uint8)
            cls_map[ok] = clf.classes_[prob[:, ok].argmax(0)]
            cls_map, prob = cls_map.reshape(h, w), prob.reshape(len(FC_NAMES), h, w)
            write_tif(outdir / "change_class.tif", meta["profile"], {"change_class": cls_map}, dtype="uint8", nodata=255)
            write_tif(outdir / "change_prob.tif", meta["profile"],
                      {f"p_{FC_NAMES[k]}": prob[i] for i, k in enumerate(clf.classes_)})
            import joblib
            joblib.dump({"clf": clf, "features": names, "classes": FC_NAMES}, outdir / "change_classifier.joblib")
        del F

    report = {"input": Path(args.input).name, "years": years, "pretrain": json.loads((outdir / "history.json").read_text())[-1]
              if (outdir / "history.json").exists() else None,
              "unsupervised_scores": score_rows, "classifier": results,
              "change_year": year_accuracy(scores, labels, cls_map, hold),
              "mgmt_agreement_pred_changed_share": mgmt_agreement(cls_map, labels),
              "class_area_ha": {FC_NAMES[k]: round(float((cls_map == k).sum()) * 0.04, 1) for k in FC_NAMES},
              "elapsed_min": round((time.time() - t0) / 60, 1)}
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[비지도 점수 AUROC / AP — 홀드아웃 블록]")
    for r in score_rows:
        print(f"  {r['task']:<32} {r['score']:<22} AUROC {r['auroc']:.3f}  AP {r['ap']:.3f}  (양성 {r['n_pos']:,} / 음성 {r['n_neg']:,})")
    print("[변화 연도 ±1년 정확도]", report["change_year"])
    print("[국유림 실적 소반에서 변화로 분류된 비율]", report["mgmt_agreement_pred_changed_share"])
    print("[분류 면적 ha]", report["class_area_ha"])
    print(f"[완료] {report['elapsed_min']}분, 결과: {outdir}")


if __name__ == "__main__":
    main()
