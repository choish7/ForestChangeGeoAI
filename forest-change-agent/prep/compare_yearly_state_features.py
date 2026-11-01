# [통합 사본] 원본: scripts/compare_yearly_state_features.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:\src\ForestChangeGeoAI\scripts")
from validate_yearly_ae_s2 import YRS, model, read_ae
from sklearn.metrics import f1_score
V = r"C:\src\ForestChangeGeoAI\output\연도별라벨\검증용"
v = pd.read_parquet(V + r"\검증_표본점_연도별.parquet")
v["클래스"] = np.where(v["입목여부"].eq("무립목지"), "무립목지", v["임상"])
pts = v.drop_duplicates("조각ID")[["조각ID", "행", "열"]].reset_index(drop=True)
E = read_ae(pts["행"].to_numpy(), pts["열"].to_numpy()); pos = dict(zip(pts["조각ID"], range(len(pts))))
CLS4 = ["침엽수림", "활엽수림", "혼효림", "무립목지"]
def F(kind, idx, y):
    if kind == "단일연도": return E[y][idx]
    if kind == "당해+전년": return np.concatenate([E[y][idx], E[max(y-1, 2017)][idx]], 1)
    mean = np.nanmean(np.stack([E[k][idx] for k in YRS]), 0)
    return np.concatenate([E[y][idx], E[max(y-1, 2017)][idx], E[min(y+1, 2025)][idx], mean], 1)
def feats(kind, df):
    X = None
    for y in YRS:
        m = (df["연도"] == y).to_numpy()
        if m.any():
            f = F(kind, df.loc[m, "조각ID"].map(pos).to_numpy(), y)
            X = np.full((len(df), f.shape[1]), np.nan, "float32") if X is None else X
            X[m] = f
    return X
good = v["등급"].eq("A") & v["라벨유형"].isin(["관측", "유지추정"]) & v["클래스"].isin(CLS4)
tr, te = v[good & v["검증블록"].eq(0)], v[good & v["검증블록"].eq(1)]
ch = pd.read_parquet(V + r"\검증_변화구간.parquet")
cut = ch[ch["이전상태"].str.startswith("입목지") & ch["이후상태"].str.startswith("무립목지") & ch["검증블록"].eq(1)]
gro = ch[ch["이전상태"].str.startswith("무립목지") & ch["이후상태"].str.startswith("입목지") & ch["검증블록"].eq(1)]
for kind in ("다년(평균포함)", "당해+전년", "단일연도"):
    Xtr, Xte = feats(kind, tr), feats(kind, te); ok = ~np.isnan(Xtr).all(1)
    m = model().fit(Xtr[ok], tr["클래스"].to_numpy()[ok]); okt = ~np.isnan(Xte).all(1)
    p = m.predict(Xte[okt]); yt = te["클래스"].to_numpy()[okt]
    f4 = f1_score(yt, p, labels=CLS4, average="macro"); m3 = np.isin(yt, CLS4[:3]); f3 = f1_score(yt[m3], p[m3], labels=CLS4[:3], average="macro")
    def after(df, target):   # 구간 끝 연도(관측 연도)에 새 상태로 예측하는 비율
        idx = df["조각ID"].map(pos).to_numpy(); ys = df["구간끝"].astype(int).to_numpy()
        pr = np.array([m.predict(F(kind, idx[i:i+1], y))[0] for i, y in enumerate(ys)])
        return round(float(np.isin(pr, target).mean()), 3)
    print(f"{kind:<12} 4클래스 {f4:.3f} 임상3 {f3:.3f} | 벌채후 무립목지 예측 {after(cut, ['무립목지'])} (n={len(cut)}) | 회복후 입목 예측 {after(gro, CLS4[:3])} (n={len(gro)})", flush=True)
