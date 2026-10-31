"""GEE 추출값(wide table) → 모형 입력 특징 그룹.

AE   : AlphaEarth 연도쌍 차이벡터(64) + 코사인 거리 + 연속 연도 거리 시계열(8) 요약
S2   : Sentinel-2 성장기 NDVI·NBR 두 시기 및 차이 (농림위성 대체 분광특징)
TERR : 표고·경사
ATTR : 2019 임상도 사전 상태(영급·밀도·수종군) — 2019 산림인 표본에만 정의됨
"""
import numpy as np
import pandas as pd

from . import config as C


def _ae(df: pd.DataFrame, y: int) -> np.ndarray:
    e = df[[f"ae{y}_{b}" for b in C.AE_BANDS]].to_numpy("float32")
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-6, None)


def build(df: pd.DataFrame, y0: int = C.YEAR_FROM, y1: int = C.YEAR_TO) -> tuple[pd.DataFrame, dict]:
    F = {}
    e0, e1 = _ae(df, y0), _ae(df, y1)
    for i, b in enumerate(C.AE_BANDS):
        F[f"ae_diff_{b}"] = e1[:, i] - e0[:, i]
    F["ae_cos_dist"] = 1 - (e0 * e1).sum(1)

    years = [y for y in C.AE_YEARS if f"ae{y}_A00" in df]
    embs = {y: _ae(df, y) for y in years}
    consec = {y: 1 - (embs[y - 1] * embs[y]).sum(1) for y in years[1:]}
    for y, d in consec.items():
        F[f"ae_d{y}"] = d
    win = [y for y in consec if y0 < y <= y1]
    pre = [y for y in consec if y <= y0]
    W = np.stack([consec[y] for y in win], 1)
    F["ae_consec_max"] = W.max(1)
    F["ae_peak_year"] = np.array(win)[W.argmax(1)]
    F["ae_pre_noise"] = np.stack([consec[y] for y in pre], 1).max(1) if pre else np.nan

    for idx in ("ndvi", "nbr"):
        F[f"{idx}_{y0}"] = df[f"{idx}{y0}"].to_numpy()
        F[f"{idx}_{y1}"] = df[f"{idx}{y1}"].to_numpy()
        F[f"d{idx}"] = F[f"{idx}_{y1}"] - F[f"{idx}_{y0}"]

    F["elev"], F["slope"] = df["elev"].to_numpy(), df["slope"].to_numpy()

    if f"AGCLS_{y0}" in df:
        F["agcls_pre"] = pd.to_numeric(df[f"AGCLS_{y0}"], errors="coerce").to_numpy()
        F["dnst_pre"] = df[f"DNST_{y0}"].map({"A": 1, "B": 2, "C": 3}).to_numpy(dtype="float")
        F["koftr_pre"] = pd.to_numeric(df[f"KOFTR_{y0}"], errors="coerce").to_numpy()

    X = pd.DataFrame(F, index=df.index)
    groups = {
        "AE": [c for c in X if c.startswith("ae_") and c != "ae_peak_year"],
        "S2": [c for c in X if c.startswith(("ndvi", "nbr", "dndvi", "dnbr"))],
        "TERR": ["elev", "slope"],
        "ATTR": [c for c in ("agcls_pre", "dnst_pre", "koftr_pre") if c in X],
    }
    return X, groups
