"""AlphaEarth 분류 모델 3종(산지 피복·임상·변화유형)을 국유림 소반·산지전용 필지에 적용하고 소반별로 집계한다.

모델: ae_class_models/ (Colab 학습). Colab 과 scikit-learn 버전이 달라 저장 모델을 열 수 없으면, 같은 표본(output/ae_class_samples.npz)·
      같은 특징·같은 설정으로 여기서 다시 학습하고 독립 블록 검증 성능을 Colab report.json 과 대조한다.
과업  LC 산지 피복(산림·나지·초지·시설지, 임상도 속성 라벨: 입목지 / 미립목지 / 관목덤불·초지 / 제지·주거지)
      FT 임상(침엽수림·활엽수림·혼효림)          — 2019·2025 각각 예측
      CH 변화유형 2019→2025(무변화·벌채·갱신·산림증가, 산지전용 제외)
출력: output/ae_class_maps/{LC_2019,LC_2025,FT_2019,FT_2025,CH_2019_2025}.tif (밴드1 클래스, 밴드2 확률×100, 255=범위 밖)
      forest_decision.sqlite 테이블 "소반_AE분류", build_meta.json "AE분류" 항목
"""
import json
import sqlite3
import time
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "forest-change-agent" / "models" / "ae_class_models"
OUT = ROOT / "output" / "ae_class_maps"
DB_DIR = ROOT / "output" / "의사결정DB"
AE = {y: ROOT / "output" / f"Chuncheon_AE_{y}_64Bands_int8.tif" for y in range(2017, 2026)}
YRS = list(AE)
STRIP = 64
LABELS = {"LC": ["산림", "나지", "초지", "시설지"], "FT": ["침엽수림", "활엽수림", "혼효림"], "CH": ["무변화", "벌채·갱신", "산림증가"]}


def unit(v):
    v = v.astype("float32") / 127.0
    bad = (v == -128 / 127.0).any(1) | (v == 0).all(1)
    v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-6, None)
    v[bad] = np.nan
    return v


def state_feats(E, y, mean):
    return np.concatenate([E[y], E[max(y - 1, 2017)], E[min(y + 1, 2025)], mean], 1)


def change_feats(E):
    e19, e25 = E[2019], E[2025]
    cons = np.stack([1 - (E[y - 1] * E[y]).sum(1) for y in range(2018, 2026)], 1)
    win = cons[:, 2:]
    return np.concatenate([e19, e25, e25 - e19, (1 - (e19 * e25).sum(1))[:, None], cons, win.max(1, keepdims=True),
                           (2020 + np.nanargmax(np.nan_to_num(win, nan=-1), 1))[:, None], cons[:, :2].max(1, keepdims=True)], 1).astype("float32")


def strips(H):
    for r0 in range(0, H, STRIP):
        yield r0, min(STRIP, H - r0)


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)


def load_or_train(S, EMB):
    try:
        ms = {t: joblib.load(MD / f"{t}_model.joblib")["model"] for t in LABELS}
        return ms, {"출처": "Colab 저장 모델"}
    except Exception as e:  # 버전 불일치
        note = {"출처": f"로컬 재학습 (Colab 모델 로드 실패: {type(e).__name__}: {e})"}
    mean = np.nanmean(np.stack([EMB[y] for y in YRS]), 0)
    ref = json.loads((MD / "report.json").read_text(encoding="utf-8"))
    ms = {}
    for t, labels in LABELS.items():
        sel = S["task"] == t
        y = S["label"][sel]
        te = S["test"][sel] == 1
        if t == "CH":
            X = change_feats({k: v[sel] for k, v in EMB.items()})
        else:
            yr = S["year"][sel]
            X = np.full((sel.sum(), 256), np.nan, "float32")
            for yy in np.unique(yr):
                m = yr == yy
                X[m] = state_feats({k: v[sel][m] for k, v in EMB.items()}, int(yy), mean[sel][m])
        ok = ~np.isnan(X).all(1)
        p = model().fit(X[~te & ok], y[~te & ok]).predict(X[te & ok])
        cr = classification_report(y[te & ok], p, labels=labels, output_dict=True, zero_division=0)
        note[t] = {"로컬_macro_F1": round(cr["macro avg"]["f1-score"], 3), "Colab_macro_F1": ref[t]["다년(2017~2025)"]["macro_F1"],
                   "클래스별_F1": {l: round(cr[l]["f1-score"], 3) for l in labels}}
        print(f"[{t}] 독립 검증 macro-F1 로컬 {note[t]['로컬_macro_F1']} / Colab {note[t]['Colab_macro_F1']}", flush=True)
        ms[t] = model().fit(X[ok], y[ok])
        joblib.dump({"model": ms[t], "labels": labels}, MD / f"{t}_model_local.joblib")
    return ms, note


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = dict(np.load(ROOT / "output" / "ae_class_samples.npz"))
    with rasterio.open(AE[2025]) as ds:
        H, W, crs, tf = ds.height, ds.width, ds.crs, ds.transform
    # 적용 범위: 국유림 소반 + 산지전용 필지
    su = gpd.read_file(DB_DIR / "forest_decision.gpkg", layer="소반").to_crs(crs)
    pc = gpd.read_file(DB_DIR / "forest_decision.gpkg", layer="산지전용").to_crs(crs)
    zid = rasterize(zip(su.geometry, range(1, len(su) + 1)), out_shape=(H, W), transform=tf, fill=0, dtype="int32")
    area = (zid > 0) | rasterize(((g, 1) for g in pc.geometry), out_shape=(H, W), transform=tf, fill=0, dtype="uint8", all_touched=True).astype(bool)
    print(f"적용 화소 {int(area.sum()):,} ({area.sum() / 100:,.0f} ha)", flush=True)

    # 1차: 표본 임베딩
    EMB = {y: np.full((len(S["row"]), 64), np.nan, "float32") for y in YRS}
    srow, scol = S["row"].astype(int), S["col"].astype(int)
    srcs = {y: rasterio.open(p) for y, p in AE.items()}
    for r0, h in strips(H):
        m = (srow >= r0) & (srow < r0 + h)
        if not m.any():
            continue
        for y, ds in srcs.items():
            a = ds.read(window=Window(0, r0, W, h))
            EMB[y][m] = unit(a[:, srow[m] - r0, scol[m]].T)
    print(f"표본 임베딩 {time.time() - t0:.0f}s", flush=True)
    ms, note = load_or_train(S, EMB)

    # 2차: 적용 범위 화소 예측
    prof = dict(driver="GTiff", width=W, height=H, count=2, dtype="uint8", crs=crs, transform=tf, nodata=255, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    outs = {k: np.full((2, H, W), 255, "uint8") for k in ("LC_2019", "LC_2025", "FT_2019", "FT_2025", "CH_2019_2025")}
    for r0, h in strips(H):
        m = area[r0:r0 + h]
        if not m.any():
            continue
        rr, cc = np.nonzero(m)
        E = {y: unit(ds.read(window=Window(0, r0, W, h))[:, rr, cc].T) for y, ds in srcs.items()}
        ok = ~np.isnan(E[2025]).any(1) & ~np.isnan(E[2019]).any(1)
        if not ok.any():
            continue
        E = {y: v[ok] for y, v in E.items()}
        rr, cc = rr[ok] + r0, cc[ok]
        mean = np.nanmean(np.stack([E[y] for y in YRS]), 0)
        jobs = {"LC_2019": ("LC", state_feats(E, 2019, mean)), "LC_2025": ("LC", state_feats(E, 2025, mean)),
                "FT_2019": ("FT", state_feats(E, 2019, mean)), "FT_2025": ("FT", state_feats(E, 2025, mean)), "CH_2019_2025": ("CH", change_feats(E))}
        for k, (t, X) in jobs.items():
            P = ms[t].predict_proba(X)
            cls = list(ms[t].classes_)
            order = np.array([cls.index(l) for l in LABELS[t]])
            P = P[:, order]
            outs[k][0, rr, cc] = P.argmax(1)
            outs[k][1, rr, cc] = (P.max(1) * 100).round()
        if r0 % (STRIP * 10) == 0:
            print(f"  행 {r0}/{H} {time.time() - t0:.0f}s", flush=True)
    for ds in srcs.values():
        ds.close()
    for k, a in outs.items():
        with rasterio.open(OUT / f"{k}.tif", "w", **prof) as dst:
            dst.write(a)
            dst.set_band_description(1, "class")
            dst.set_band_description(2, "prob_x100")
            t = k.split("_")[0]
            dst.update_tags(classes=json.dumps(dict(enumerate(LABELS[t])), ensure_ascii=False))

    # 소반별 집계
    n = len(su)
    z = zid.ravel()
    rows = {"단위ID": su["단위ID"].to_numpy()}
    for k, a in outs.items():
        t = k.split("_")[0]
        c = a[0].ravel()
        valid = (z > 0) & (c != 255)
        tot = np.bincount(z[valid], minlength=n + 1)[1:]
        for i, l in enumerate(LABELS[t]):
            cnt = np.bincount(z[valid & (c == i)], minlength=n + 1)[1:]
            key = f"{k}_{l}" if t != "CH" else f"AE_{l}"
            rows[key + "_ha"] = (cnt * 0.01).round(2)
            if t != "CH":
                rows[key + "_비율"] = np.where(tot > 0, cnt / np.maximum(tot, 1), np.nan).round(3)
        pr = a[1].ravel().astype(float)
        s = np.bincount(z[valid], weights=pr[valid], minlength=n + 1)[1:]
        rows[f"{k}_평균확률"] = np.where(tot > 0, s / np.maximum(tot, 1) / 100, np.nan).round(3)
    df = pd.DataFrame(rows)
    for y in (2019, 2025):
        cols = [f"FT_{y}_{l}_비율" for l in LABELS["FT"]]
        df[f"AE_우세임상_{y}"] = np.where(df[cols].notna().all(1), np.array(LABELS["FT"])[df[cols].fillna(-1).to_numpy().argmax(1)], None)
    df["AE_산림비율_변화"] = (df["LC_2025_산림_비율"] - df["LC_2019_산림_비율"]).round(3)
    df["AE_시설지나지_증가"] = ((df["LC_2025_시설지_비율"] + df["LC_2025_나지_비율"]) - (df["LC_2019_시설지_비율"] + df["LC_2019_나지_비율"])).round(3)
    with sqlite3.connect(DB_DIR / "forest_decision.sqlite") as con:
        df.to_sql("소반_AE분류", con, index=False, if_exists="replace")
    meta = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    meta["AE분류"] = {"모델": note, "적용범위": "국유림 소반 + 산지전용 필지", "지도": str(OUT),
                    "라벨": "임상도 속성만: 산림=입목지, 나지=미립목지, 초지=관목덤불·초지, 시설지=제지·주거지; 변화유형은 산지전용 제외",
                    "주의": ["초지 F1 0.11 — 판단에 쓰지 말 것", "시설지는 임도·묘지·송전탑 등 산지 안 산림 외 용도(제지)", "혼효림은 침엽·활엽과 혼동이 큼"]}
    (DB_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df[["LC_2025_산림_비율", "AE_벌채·갱신_ha", "AE_산림증가_ha"]].describe().round(3).to_string())
    print(f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
