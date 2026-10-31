"""임상도 연도별 관측 라벨(2017~2025)을 의사결정 DB에 연결.

1) 연도별 임상 추정 모델(학습용): 성능은 학습 블록(검증블록 0)으로 학습해 검증 블록에서 평가하고, 배포 모델은 A등급 전체로 다시 학습
   클래스 = 침엽수림·활엽수림·혼효림·무립목지, 특징 = 해당 연도·앞뒤 연도 임베딩 + 2017~2025 평균 (분류모델과 같음)
   검증 블록(검증블록 1)에서 연도별 macro-F1 을 2019·2025 라벨만으로 학습한 모델과 비교 (검증 블록은 학습에 쓰지 않음)
2) 모든 조각 표본점 × 2017~2025 예측 → 소반별 연도별 임상 구성(조각 면적 가중)
3) 소반별 임상도 조사 이력: 조사연도, 마지막 조사연도, 연도별 상태(면적 최다), 입목↔무립목 변화구간, 임상 간 전이(판독 수정 가능)
   위성 변화연도(최종지도·AE)가 마지막 조사 이후면 '임상도 미반영 변화' → 재조사 우선
4) 연도 신뢰도: 안정 조각 연도별 오탐률(검증결과 V2)을 build_meta 에 저장 — AE 단독 신호 보정에 사용
출력: forest_decision.sqlite 테이블 "소반_연도별임상", "소반_임상도이력", ae_class_models/FT4_yearly_model_local.joblib, build_meta.json
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_yearly_ae_s2 import YRS, ae_state, model, read_ae  # noqa: E402

V = ROOT / "output" / "연도별라벨" / "검증용"
R = ROOT / "output" / "연도별라벨" / "검증결과"
DB_DIR = ROOT / "output" / "의사결정DB"
CLS4 = ["침엽수림", "활엽수림", "혼효림", "무립목지"]


def main():
    t0 = time.time()
    v = pd.read_parquet(V / "검증_표본점_연도별.parquet")
    v["클래스"] = np.where(v["입목여부"].eq("무립목지"), "무립목지", v["임상"])
    pts = v.drop_duplicates("조각ID")[["조각ID", "행", "열", "검증블록", "면적_ha", "국유림소반"]].reset_index(drop=True)
    E = read_ae(pts["행"].to_numpy(), pts["열"].to_numpy())
    pos = dict(zip(pts["조각ID"], range(len(pts))))
    print(f"AE 추출 {time.time() - t0:.0f}s", flush=True)

    # 1) 학습 (학습 블록 A등급) / 검증 (검증 블록 A등급)
    good = v["등급"].eq("A") & v["라벨유형"].isin(["관측", "유지추정"]) & v["클래스"].isin(CLS4)

    def feats(df):
        X = np.full((len(df), 256), np.nan, "float32")
        for y in YRS:
            m = (df["연도"] == y).to_numpy()
            if m.any():
                X[m] = ae_state(E, df.loc[m, "조각ID"].map(pos).to_numpy(), y)
        return X
    tr, te = v[good & v["검증블록"].eq(0)], v[good & v["검증블록"].eq(1)]
    Xtr, Xte = feats(tr), feats(te)
    ok = ~np.isnan(Xtr).all(1)
    m4 = model().fit(Xtr[ok], tr["클래스"].to_numpy()[ok])
    joblib.dump({"model": m4, "labels": CLS4, "학습": "임상도 연도별 A등급(관측·유지추정), 검증블록 0"},
                ROOT / "ae_class_models" / "FT4_yearly_model_local.joblib")
    okt = ~np.isnan(Xte).all(1)
    pt = m4.predict(Xte[okt])
    te = te[okt].assign(예측=pt)
    v1 = pd.read_csv(R / "V1_연도별_임상분류.csv")
    rows = []
    for y in YRS:
        g = te[te["연도"].eq(y)]
        g3 = g[g["클래스"].isin(CLS4[:3])]
        old = v1[(v1["등급"] == "A") & (v1["연도"] == y)]
        rows.append({"연도": y, "검증표본": len(g), "macroF1_4클래스": round(f1_score(g["클래스"], g["예측"], labels=CLS4, average="macro"), 3),
                     "macroF1_임상3클래스": round(f1_score(g3["클래스"], g3["예측"], labels=CLS4[:3], average="macro"), 3),
                     "기존(2019·2025 학습)_임상3클래스": float(old["AE_macroF1"].iloc[0]) if len(old) else None,
                     "무립목지_F1": round(f1_score(g["클래스"], g["예측"], labels=["무립목지"], average="macro"), 3)})
    ev = pd.DataFrame(rows)
    print(ev.to_string(index=False), flush=True)
    # 배포 모델: 학습용으로 전환 — A등급 전체(모든 블록)로 다시 학습. 성능은 위 블록 분할 결과로 보고
    Xall = np.concatenate([Xtr, Xte])
    yall = np.concatenate([tr["클래스"].to_numpy(), v[good & v["검증블록"].eq(1)]["클래스"].to_numpy()])
    oka = ~np.isnan(Xall).all(1)
    m4 = model().fit(Xall[oka], yall[oka])
    joblib.dump({"model": m4, "labels": CLS4, "학습": "임상도 연도별 A등급(관측·유지추정) 전체 블록",
                 "성능": "학습 블록으로 학습·검증 블록 평가 결과(build_meta 연도별임상모델)"}, ROOT / "ae_class_models" / "FT4_yearly_model_local.joblib")
    print(f"배포 모델 학습 표본 {int(oka.sum()):,}", flush=True)

    # 2) 모든 표본점 × 연도 예측 → 소반별 연도별 구성
    nf = pts[pts["국유림소반"].notna()].reset_index(drop=True)
    idx = nf["조각ID"].map(pos).to_numpy()
    out = []
    for y in YRS:
        X = ae_state(E, idx, y)
        okp = ~np.isnan(X).all(1)
        P = np.full((len(idx), 4), np.nan)
        P[okp] = m4.predict_proba(X[okp])[:, [list(m4.classes_).index(c) for c in CLS4]]
        out.append(pd.DataFrame(P, columns=CLS4).assign(단위ID=nf["국유림소반"], 연도=y, 면적=nf["면적_ha"], 예측=np.array(CLS4)[np.nan_to_num(P, nan=-1).argmax(1)]))
    pr = pd.concat(out)
    w = pr.dropna(subset=CLS4)
    yearly = (w.assign(**{c: (w["예측"] == c) * w["면적"] for c in CLS4})
               .groupby(["단위ID", "연도"])[CLS4 + ["면적"]].sum())
    for c in CLS4:
        yearly[f"{c}_비율"] = (yearly[c] / yearly["면적"]).round(3)
    yearly = yearly[[f"{c}_비율" for c in CLS4]].reset_index()
    yearly["표본조각"] = w.groupby(["단위ID", "연도"]).size().values
    yearly["우세"] = np.array(CLS4)[yearly[[f"{c}_비율" for c in CLS4]].to_numpy().argmax(1)]

    # 3) 소반별 임상도 조사 이력
    gp = ROOT / "output" / "연도별라벨" / "검증용" / "연도별관측라벨_2017_2025.gpkg"
    import geopandas as gpd
    pc = gpd.read_file(gp, layer="조각", ignore_geometry=True)
    pc = pc[pc["국유림소반"].notna()]
    ucols = [c for c in pc.columns if c.startswith("u_")]
    surv = pc.melt(id_vars=["조각ID", "국유림소반", "area_ha"], value_vars=ucols, value_name="u").dropna(subset=["u"])
    sy = surv.groupby("국유림소반")["u"].agg(lambda s: sorted(set(int(x) for x in s)))
    last = surv.groupby("국유림소반").apply(lambda g: int(np.average(g["u"], weights=g["area_ha"]).round()) if len(g) else None)
    lastmax = surv.groupby("국유림소반")["u"].max().astype(int)
    lv = v[v["국유림소반"].notna()]
    st = (lv.assign(a=lv["면적_ha"]).groupby(["국유림소반", "연도", "라벨유형"])["a"].sum().reset_index()
            .sort_values("a").groupby(["국유림소반", "연도"]).last().reset_index())
    hist = st.pivot(index="국유림소반", columns="연도", values="라벨유형")
    hist = hist.apply(lambda r: ", ".join(f"{y}:{t}" for y, t in r.items() if isinstance(t, str)), axis=1)
    ch = pd.read_parquet(V / "검증_변화구간.parquet")
    ch = ch[ch["국유림소반"].notna()].copy()
    ch["전이"] = np.where(ch["이전상태"].str.startswith("입목지") & ch["이후상태"].str.startswith("입목지"), "임상 간(판독 수정 가능)",
                 ch["이전상태"].str.split("|").str[0] + "→" + ch["이후상태"].str.split("|").str[0])
    ch["구간"] = ch["구간시작"].astype(str) + "~" + ch["구간끝"].astype(str)
    chs = ch.groupby("국유림소반").apply(lambda g: "; ".join(f"{r.전이} ({r.구간}, {r.면적_ha:.2f} ha)" for r in g.itertuples()))
    big = ch[~ch["전이"].str.startswith("임상 간")].groupby("국유림소반").agg(입목변화_ha=("면적_ha", "sum"),
                                                                   입목변화_구간끝=("구간끝", "max"))
    with sqlite3.connect(DB_DIR / "forest_decision.sqlite") as con:
        u = pd.read_sql('select u."단위ID", u."FM_변화연도", u."FM_변화면적_ha", u."신호일치수", p."정점연도" from "소반" u join "소반_경로" p using("단위ID")', con)
    h = pd.DataFrame({"단위ID": u["단위ID"]}).set_index("단위ID")
    h["조사연도"] = sy.map(lambda s: ",".join(map(str, s)))
    h["마지막조사_면적가중"] = last
    h["마지막조사_최근"] = lastmax
    h["연도별라벨유형"] = hist
    h["변화구간"] = chs
    h = h.join(big).reset_index()
    h = h.merge(u, on="단위ID", how="left")
    sat_year = h["FM_변화연도"].fillna(h["정점연도"])
    h["위성변화연도"] = sat_year
    h["조사후_경과연수(2025기준)"] = 2025 - h["마지막조사_면적가중"]
    h["임상도_미반영변화"] = (h["FM_변화면적_ha"].fillna(0) >= 0.5) & (sat_year > h["마지막조사_최근"])
    h["재조사_우선"] = np.select([h["임상도_미반영변화"], h["조사후_경과연수(2025기준)"].ge(7)],
                             ["위성 변화가 마지막 조사 이후 — 임상도 미반영", "마지막 조사 7년 이상 경과"], default="")
    # 연도별 추정 임상의 우세 클래스 전환
    ys = yearly.pivot(index="단위ID", columns="연도", values="우세")
    h["AE연도별_우세임상"] = h["단위ID"].map(ys.apply(lambda r: "→".join(f"{y}:{c}" for y, c in r.items()
                                                                   if isinstance(c, str) and (y == r.first_valid_index() or c != r.get(y - 1))), axis=1))
    with sqlite3.connect(DB_DIR / "forest_decision.sqlite") as con:
        yearly.to_sql("소반_연도별임상", con, index=False, if_exists="replace")
        h.drop(columns=["FM_변화연도", "FM_변화면적_ha", "신호일치수", "정점연도"]).to_sql("소반_임상도이력", con, index=False, if_exists="replace")

    # 4) 연도 신뢰도
    v2 = pd.read_csv(R / "V2_안정조각_연도별_오탐률.csv")
    meta = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    meta["연도신뢰도"] = {"출처": "임상도 연도별 관측 라벨 안정 조각(검증 전용) 연도별 오탐률",
                      "AE_오탐률": {r["구간"].split("→")[1]: r["AE_오탐률"] for _, r in v2.iterrows()},
                      "S2_오탐률": {r["구간"].split("→")[1]: r["S2_NDVI감소_오탐률"] for _, r in v2.iterrows()},
                      "AE_고오탐연도": [int(r["구간"].split("→")[1]) for _, r in v2.iterrows() if r["AE_오탐률"] >= 0.05]}
    meta["연도별임상모델"] = {"배포모델_학습": "임상도 연도별 A등급 전체 블록", "성능평가": "검증블록 0 학습 → 검증블록 1 평가",
                          "검증(검증블록 1)": ev.to_dict("records")}
    (DB_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(h["재조사_우선"].value_counts().to_dict(), "| 임상도 미반영 변화", int(h["임상도_미반영변화"].sum()))
    print("AE 고오탐 연도", meta["연도신뢰도"]["AE_고오탐연도"], f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
