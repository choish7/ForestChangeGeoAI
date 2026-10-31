"""임상도 연도별 관측 라벨(검증 전용)로 AlphaEarth 와 Sentinel-2 를 같은 조건에서 비교.

검증 세트는 학습·임계값 조정에 쓰지 않는다.
  V1 연도별 임상 분류 : 분류모델 표본(ae_class_samples, 임상도 2019·2025) 중 학습 블록(test=0)으로 AE·S2 모델을 각각 학습 →
                       검증 세트의 검증블록·A등급 표본점에서 2017~2025 연도별 macro-F1 (B등급 별도)
  V2 안정 조각 오탐률  : 5개 연도 이상 A등급으로 같은 상태인 조각에서 연도별 변화 신호 비율
                       AE: 연속연도 임베딩 거리 > 0.0532, S2: 연간 NDVI 감소 > 0.1526 (둘 다 기존 합의라벨로 정한 오탐률 5% 임계값)
  V3 변화 판별력       : 변화구간 조각(입목↔무립목, 임상 간) vs 안정 조각 — 2018~2025 최대 신호의 AUROC
  V4 변화 시점         : 시점평가용 변화구간(≤3년, 양끝 조사)에서 최대 신호 연도가 (구간시작, 구간끝]에 드는 비율 + 무작위 기대값
  V5 변화유형 2019→2025: 분류모델 CH 표본 학습 블록으로 AE·S2 모델 학습 → 구간이 2019~2025 안인 입목↔무립목 변화와 안정 조각(검증블록)
출력: output/연도별라벨/검증결과/검증결과.json, *.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
V = ROOT / "output" / "연도별라벨" / "검증용"
OUT = ROOT / "output" / "연도별라벨" / "검증결과"
AE = {y: ROOT / "output" / f"Chuncheon_AE_{y}_64Bands_int8.tif" for y in range(2017, 2026)}
S2 = ROOT / "output" / "Chuncheon_S2_2017_2025_54Bands.tif"
YRS = list(range(2017, 2026))
IDX = ["NDVI", "NDWI", "NDBI", "NDMI"]
T_AE, T_S2 = 0.0532, 0.1526
FT3 = ["침엽수림", "활엽수림", "혼효림"]


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)


def read_ae(rows, cols):
    """화소 목록의 연도별 단위 임베딩 (n, 64) — 띠 단위로 읽어 메모리 절약."""
    E = {y: np.full((len(rows), 64), np.nan, "float32") for y in YRS}
    order = np.argsort(rows)
    srcs = {y: rasterio.open(p) for y, p in AE.items()}
    W = srcs[2025].width
    for r0 in range(0, srcs[2025].height, 64):
        m = (rows >= r0) & (rows < r0 + 64)
        if not m.any():
            continue
        for y, ds in srcs.items():
            a = ds.read(window=Window(0, r0, W, min(64, ds.height - r0)))[:, rows[m] - r0, cols[m]].T.astype("float32")
            bad = (a == -128).any(1) | (a == 0).all(1)
            a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-6, None)
            a[bad] = np.nan
            E[y][m] = a
    for ds in srcs.values():
        ds.close()
    return E


def read_s2(xs, ys):
    with rasterio.open(S2) as ds:
        desc = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        r, c = rasterio.transform.rowcol(ds.transform, xs, ys)
        r, c = np.clip(np.asarray(r), 0, ds.height - 1), np.clip(np.asarray(c), 0, ds.width - 1)
        S = {y: np.full((len(xs), 4), np.nan, "float32") for y in YRS}
        for r0 in range(0, ds.height, 128):
            m = (r >= r0) & (r < r0 + 128)
            if not m.any():
                continue
            h = min(128, ds.height - r0)
            for y in YRS:
                a = ds.read([desc[f"{k}_{y}"] for k in IDX], window=Window(0, r0, ds.width, h)).astype("float32")
                a[a == -9999] = np.nan
                S[y][m] = a[:, r[m] - r0, c[m]].T
    return S


def ae_state(E, idx, y):
    mean = np.nanmean(np.stack([E[k][idx] for k in YRS]), 0)
    return np.concatenate([E[y][idx], E[max(y - 1, 2017)][idx], E[min(y + 1, 2025)][idx], mean], 1)


def s2_state(S, idx, y):
    mean = np.nanmean(np.stack([S[k][idx] for k in YRS]), 0)
    return np.concatenate([S[y][idx], S[max(y - 1, 2017)][idx], S[min(y + 1, 2025)][idx], mean], 1)


def ae_change(E, idx):
    e19, e25 = E[2019][idx], E[2025][idx]
    cons = np.stack([1 - (E[y - 1][idx] * E[y][idx]).sum(1) for y in range(2018, 2026)], 1)
    return np.concatenate([e19, e25, e25 - e19, (1 - (e19 * e25).sum(1))[:, None], cons, cons[:, 2:].max(1, keepdims=True)], 1)


def s2_change(S, idx):
    a = np.stack([S[y][idx] for y in YRS], 1)          # (n, 9, 4)
    d = np.diff(a, axis=1)
    return np.concatenate([a.reshape(len(idx), -1), d.reshape(len(idx), -1), -np.nanmin(d, 1), np.nanmax(d, 1), a[:, -1] - a[:, 2]], 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v = pd.read_parquet(V / "검증_표본점_연도별.parquet")
    pts = v.drop_duplicates("조각ID")[["조각ID", "행", "열", "x", "y", "검증블록"]].reset_index(drop=True)
    S = dict(np.load(ROOT / "output" / "ae_class_samples.npz"))
    from rasterio.transform import Affine, xy
    TF = Affine(10, 0, 368350, 0, -10, 4216210)
    sx, sy = xy(TF, S["row"], S["col"])
    # 검증 표본점 + 학습 표본을 한 번에 읽기
    rows = np.r_[pts["행"].to_numpy(), S["row"].astype(int)]
    cols = np.r_[pts["열"].to_numpy(), S["col"].astype(int)]
    xs, ys = np.r_[pts["x"].to_numpy(), np.asarray(sx)], np.r_[pts["y"].to_numpy(), np.asarray(sy)]
    E = read_ae(rows, cols)
    print(f"AE 추출 {time.time() - t0:.0f}s", flush=True)
    Sx = read_s2(xs, ys)
    print(f"S2 추출 {time.time() - t0:.0f}s", flush=True)
    nv = len(pts)
    vi = np.arange(nv)                                   # 검증 표본점 인덱스
    si = nv + np.arange(len(S["row"]))                   # 학습 표본 인덱스
    res = {"검증표본점": nv, "임계값": {"AE": T_AE, "S2_NDVI감소": T_S2}}

    # ---- V1 연도별 임상 분류
    tr = (S["task"] == "FT") & (S["test"] == 0)
    Xa = np.concatenate([ae_state(E, si[tr & (S["year"] == y)], y) for y in (2019, 2025)])
    Xs = np.concatenate([s2_state(Sx, si[tr & (S["year"] == y)], y) for y in (2019, 2025)])
    yl = np.concatenate([S["label"][tr & (S["year"] == y)] for y in (2019, 2025)])
    okA, okS = ~np.isnan(Xa).all(1), ~np.isnan(Xs).any(1)
    mA = model().fit(Xa[okA & okS], yl[okA & okS])
    mS = model().fit(Xs[okA & okS], yl[okA & okS])
    pos = dict(zip(pts["조각ID"], vi))
    rows1 = []
    for grade in ("A", "B"):
        g = v[v["등급"].eq(grade) & v["임상"].isin(FT3) & v["검증블록"].eq(1) & v["라벨유형"].isin(["관측", "유지추정"])]
        for y in YRS:
            gy = g[g["연도"].eq(y)]
            if len(gy) < 30:
                continue
            idx = gy["조각ID"].map(pos).to_numpy()
            xa, xs_ = ae_state(E, idx, y), s2_state(Sx, idx, y)
            ok = ~np.isnan(xa).all(1) & ~np.isnan(xs_).any(1)
            yt = gy["임상"].to_numpy()[ok]
            pa, ps = mA.predict(xa[ok]), mS.predict(xs_[ok])
            rows1.append({"등급": grade, "연도": y, "표본": int(ok.sum()),
                          "AE_macroF1": round(f1_score(yt, pa, labels=FT3, average="macro"), 3), "S2_macroF1": round(f1_score(yt, ps, labels=FT3, average="macro"), 3),
                          "AE_정확도": round(float((pa == yt).mean()), 3), "S2_정확도": round(float((ps == yt).mean()), 3),
                          "학습연도여부": y in (2019, 2025)})
    v1 = pd.DataFrame(rows1)
    v1.to_csv(OUT / "V1_연도별_임상분류.csv", index=False, encoding="utf-8-sig")
    res["V1_연도별_임상분류"] = v1.to_dict("records")
    print(v1.to_string(index=False), flush=True)

    # ---- 연속연도 신호 (모든 검증 표본점)
    aeD = np.stack([1 - (E[y - 1][vi] * E[y][vi]).sum(1) for y in range(2018, 2026)], 1)       # 열 0 = 2017→2018
    nd = np.stack([Sx[y][vi, 0] for y in YRS], 1)
    s2D = -(np.diff(nd, axis=1))                                                               # NDVI 감소량
    s2A = np.abs(np.diff(nd, axis=1))                                                          # 증감 모두
    years_end = np.arange(2018, 2026)

    # ---- V2 안정 조각 연도별 오탐률
    st = pd.read_parquet(V / "검증_안정조각.parquet")
    lab = v.set_index(["조각ID", "연도"])
    rows2 = []
    sidx = st["조각ID"].map(pos).to_numpy()
    for j, y in enumerate(years_end):
        # 그 해와 전년이 모두 A등급 같은 상태로 확인된 안정 조각만
        both = [(lab.loc[(c, y), "등급"] == "A") and (lab.loc[(c, y - 1), "등급"] == "A") if (c, y - 1) in lab.index else False
                for c in st["조각ID"]]
        m = sidx[np.asarray(both)]
        if len(m) < 30:
            continue
        rows2.append({"구간": f"{y - 1}→{y}", "안정조각": len(m),
                      "AE_오탐률": round(float(np.nanmean(aeD[m, j] > T_AE)), 3), "S2_NDVI감소_오탐률": round(float(np.nanmean(s2D[m, j] > T_S2)), 3)})
    v2 = pd.DataFrame(rows2)
    v2.to_csv(OUT / "V2_안정조각_연도별_오탐률.csv", index=False, encoding="utf-8-sig")
    res["V2_안정조각_오탐률"] = v2.to_dict("records")
    print(v2.to_string(index=False), flush=True)

    # ---- V3 변화 판별력, V4 시점
    ch = pd.read_parquet(V / "검증_변화구간.parquet")
    ch["전이"] = np.where(ch["이전상태"].str.startswith("입목지") & ch["이후상태"].str.startswith("무립목지"), "입목지→무립목지",
                 np.where(ch["이전상태"].str.startswith("무립목지") & ch["이후상태"].str.startswith("입목지"), "무립목지→입목지",
                          np.where(ch["이전상태"].str.startswith("입목지") & ch["이후상태"].str.startswith("입목지"), "임상 간 전이", "기타")))
    cidx = ch["조각ID"].map(pos).to_numpy()
    rows3 = []
    for tname in ("입목지→무립목지", "무립목지→입목지", "임상 간 전이"):
        m = ch["전이"].eq(tname).to_numpy()
        for nm, sig in (("AE 연속거리", aeD), ("S2 NDVI 증감", s2A)):
            pos_s, neg_s = np.nanmax(sig[cidx[m]], 1), np.nanmax(sig[sidx], 1)
            ok_p, ok_n = np.isfinite(pos_s), np.isfinite(neg_s)
            y_ = np.r_[np.ones(ok_p.sum()), np.zeros(ok_n.sum())]
            rows3.append({"전이": tname, "신호": nm, "변화조각": int(ok_p.sum()), "안정조각": int(ok_n.sum()),
                          "AUROC": round(float(roc_auc_score(y_, np.r_[pos_s[ok_p], neg_s[ok_n]])), 3)})
    v3 = pd.DataFrame(rows3)
    v3.to_csv(OUT / "V3_변화판별력.csv", index=False, encoding="utf-8-sig")
    res["V3_변화판별력"] = v3.to_dict("records")
    print(v3.to_string(index=False), flush=True)

    tm = ch[ch["시점평가용"]].copy()
    tidx = tm["조각ID"].map(pos).to_numpy()
    rows4 = []
    for tname in ("입목지→무립목지", "무립목지→입목지", "임상 간 전이", "전체"):
        m = np.ones(len(tm), bool) if tname == "전체" else tm["전이"].eq(tname).to_numpy()
        if m.sum() < 10:
            continue
        s0, s1 = tm["구간시작"].to_numpy()[m], tm["구간끝"].to_numpy()[m]
        rnd = np.mean([(min(b, 2025) - max(a, 2017)) / 8 for a, b in zip(s0, s1)])           # 2018~2025 중 무작위 연도의 기대 적중률
        row = {"전이": tname, "변화구간": int(m.sum()), "무작위_기대적중": round(float(rnd), 3)}
        for nm, sig in (("AE", aeD), ("S2_NDVI", s2A)):
            sv = sig[tidx[m]]
            ok = np.isfinite(sv).any(1)
            peak = years_end[np.nanargmax(np.where(np.isfinite(sv), sv, -1), 1)]
            hit = (peak > s0) & (peak <= s1)
            row[f"{nm}_구간적중"] = round(float(hit[ok].mean()), 3)
            row[f"{nm}_구간±1년"] = round(float(((peak > s0 - 1) & (peak <= s1 + 1))[ok].mean()), 3)
        rows4.append(row)
    v4 = pd.DataFrame(rows4)
    v4.to_csv(OUT / "V4_변화시점_구간적중.csv", index=False, encoding="utf-8-sig")
    res["V4_변화시점"] = v4.to_dict("records")
    print(v4.to_string(index=False), flush=True)

    # ---- V5 변화유형 2019→2025 (CH 학습 블록 모델)
    trc = (S["task"] == "CH") & (S["test"] == 0)
    XcA, XcS = ae_change(E, si[trc]), s2_change(Sx, si[trc])
    ycl = S["label"][trc]
    ok = ~np.isnan(XcA).all(1) & ~np.isnan(XcS).all(1)
    mcA, mcS = model().fit(XcA[ok], ycl[ok]), model().fit(XcS[ok], ycl[ok])
    sel = ch[ch["전이"].isin(["입목지→무립목지", "무립목지→입목지"]) & ch["구간시작"].ge(2019) & ch["구간끝"].le(2025) & ch["검증블록"].eq(1)]
    stb = st[st["검증블록"].eq(1)]
    idx5 = np.r_[sel["조각ID"].map(pos).to_numpy(), stb["조각ID"].map(pos).to_numpy()]
    y5 = np.r_[np.where(sel["전이"].eq("입목지→무립목지"), "벌채·갱신", "산림증가"), np.repeat("무변화", len(stb))]
    xa5, xs5 = ae_change(E, idx5), s2_change(Sx, idx5)
    ok5 = ~np.isnan(xa5).all(1) & ~np.isnan(xs5).all(1)
    L = ["무변화", "벌채·갱신", "산림증가"]
    pa5, ps5 = mcA.predict(xa5[ok5]), mcS.predict(xs5[ok5])
    res["V5_변화유형"] = {"표본": {l: int((y5[ok5] == l).sum()) for l in L},
                       "AE": {"macroF1": round(f1_score(y5[ok5], pa5, labels=L, average="macro"), 3),
                              "클래스F1": dict(zip(L, f1_score(y5[ok5], pa5, labels=L, average=None).round(3).tolist()))},
                       "S2": {"macroF1": round(f1_score(y5[ok5], ps5, labels=L, average="macro"), 3),
                              "클래스F1": dict(zip(L, f1_score(y5[ok5], ps5, labels=L, average=None).round(3).tolist()))}}
    print(json.dumps(res["V5_변화유형"], ensure_ascii=False), flush=True)
    res["소요_초"] = round(time.time() - t0)
    (OUT / "검증결과.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("저장:", OUT, res["소요_초"], "s")


if __name__ == "__main__":
    main()
