"""E5b·E5c 독립 사건 기준 변화 시점 검증 (임상도 갱신년도 대신 실제 행정 사건 연도).

E5b 국유림 수확·갱신 사업: 소반별 2020~2024 최초 사건연도
     A등급 = 임목생산 주벌, B등급 = 조림예정지정리·인공갱신(벌채 직후 작업)
     비교 신호: 소반 평균 AlphaEarth 연속연도 거리 최대 구간의 끝 연도(2018~2025 전체 범위, 제한 없음),
               소반 평균 Sentinel-2 NDVI 최대 감소 연도, 최종지도 변화연도(화소 최빈)
E5c 보전산지 해제 고시(전용확정 필지): 고시연도는 전용 완료 후 고시 → 변화 시점의 상한
결과: results/E5b_harvest_timing.csv, E5c_conversion_timing.csv, timing_summary.json
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\src\ForestChangeGeoAI")
sys.path.insert(0, str(ROOT))
from forest_decision_agent import tools as T  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def signal_years(**key):
    t = T.unit_timeseries(T.Session("timing", run_dir=T.RUNS / "_scratch"), **key)["data"]
    ae = {int(k.split("→")[1]): v for k, v in t["AlphaEarth_연속연도_임베딩거리"].items() if v is not None}
    nd = {int(y): v["NDVI"] for y, v in t["Sentinel-2_연별_평균"].items() if v["NDVI"] is not None}
    drops = {y: nd[y - 1] - nd[y] for y in nd if y - 1 in nd}
    fy = t["최종지도(AE+S2)"]["변화연도_화소"]
    return {"AE_정점": max(ae, key=ae.get) if ae else None, "AE_정점거리": max(ae.values()) if ae else None,
            "NDVI_최대감소": max(drops, key=drops.get) if drops else None, "NDVI_감소폭": round(max(drops.values()), 3) if drops else None,
            "최종지도_연도": int(max(fy, key=fy.get)) if fy else None}


def agree(df, ev, col):
    d = df.dropna(subset=[col])
    lag = d[col] - d[ev]
    return {"n": int(len(d)), "정확일치": round(float((lag == 0).mean()), 3), "±1년": round(float((lag.abs() <= 1).mean()), 3),
            "시차분포(신호−사건)": {int(k): int(v) for k, v in lag.value_counts().sort_index().items()}}


def main():
    con = sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)
    r = pd.read_sql('select * from "사업실적"', con)
    r = r[pd.to_numeric(r["연도"], errors="coerce").between(2020, 2024)]
    harv = r[(r["사업구분"].eq("임목생산") & r["종별"].eq("주벌")) | (r["사업구분"].eq("조림") & r["종별"].isin(["조림예정지정리", "인공갱신"]))].copy()
    harv["등급"] = np.where(harv["종별"].eq("주벌"), "A(주벌)", "B(조림예정지정리·인공갱신)")
    harv["연도"] = harv["연도"].astype(int)
    ev = harv.sort_values(["연도", "등급"]).groupby("단위ID").first().reset_index()[["단위ID", "연도", "등급", "종별"]]
    rows = []
    for _, e in ev.iterrows():
        rows.append({**e.to_dict(), **signal_years(unit_id=e["단위ID"])})
    h = pd.DataFrame(rows).rename(columns={"연도": "사건연도"})
    h.to_csv(OUT / "E5b_harvest_timing.csv", index=False, encoding="utf-8-sig")
    out = {"E5b_수확갱신사업": {}}
    for g, sub in [("전체", h), ("A(주벌)", h[h["등급"].str.startswith("A")]), ("B", h[h["등급"].str.startswith("B")])]:
        out["E5b_수확갱신사업"][g] = {s: agree(sub, "사건연도", s) for s in ("AE_정점", "NDVI_최대감소", "최종지도_연도")}
    # 신호가 약한 소반(변화 없음) 비율 — 소반 전체 평균이라 일부 사업이 희석
    out["E5b_수확갱신사업"]["AE_정점거리_중앙값"] = round(float(h["AE_정점거리"].median()), 3)

    p = pd.read_sql('select * from "산지전용"', con)
    p = p[p["라벨"].eq("전용확정")]
    rows = [{"레코드ID": x.레코드ID, "고시연도": int(x.고시연도), **signal_years(record_id=x.레코드ID)} for x in p.itertuples()]
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "E5c_conversion_timing.csv", index=False, encoding="utf-8-sig")
    lag = c["고시연도"] - c["AE_정점"]
    lag2 = c["고시연도"] - c["NDVI_최대감소"]
    out["E5c_산지전용고시"] = {"필지": int(len(c)),
                           "AE_정점≤고시연도": round(float((lag >= 0).mean()), 3), "AE_고시 0~2년 전": round(float(lag.between(0, 2).mean()), 3),
                           "NDVI_최대감소≤고시연도": round(float((lag2 >= 0).mean()), 3), "NDVI_고시 0~2년 전": round(float(lag2.between(0, 2).mean()), 3),
                           "AE 시차분포(고시−정점)": {int(k): int(v) for k, v in lag.value_counts().sort_index().items()},
                           "NDVI 시차분포(고시−감소)": {int(k): int(v) for k, v in lag2.value_counts().sort_index().items()}}
    (OUT / "timing_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
