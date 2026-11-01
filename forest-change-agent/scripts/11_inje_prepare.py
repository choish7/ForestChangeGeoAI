# [통합 사본] 원본: paper/revision/inje_prepare.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""인제군 전이 시험 준비 (로컬): 임상도 2019·2025 변화 라벨 → 층화 표본점, 사업실적 사건 표본점, Colab 전송용 payload.

춘천과 같은 규칙(fcd.labels.classify, 0.1 ha 조각, 15 m 내부 버퍼, 10 m 중복 제거)을 쓰고,
춘천 표본(work/samples.parquet)도 같은 형식으로 묶어 Colab 에서 두 지역 특징을 같은 코드로 추출한다.
  python forest-change-agent/scripts/11_inje_prepare.py
산출: forest-change-agent/work/inje/{inje_change_polygons.parquet, inje_samples.parquet, inje_events.parquet, payload_*.txt}
"""
import base64
import gzip
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
sys.path.insert(0, str(ROOT / "forest-change-agent"))
from fcd import config as C                      # noqa: E402
from fcd.labels import TRAIN_CLASSES, classify  # noqa: E402

SRC = DATA / "타지역 테스트(강원도 인제군)"
OUT = Path(__file__).resolve().parents[1] / "work" / "inje"
OUT.mkdir(parents=True, exist_ok=True)
MAPS = {2019: SRC / "임상도 51810 (2019 인제)" / "51810.shp", 2025: SRC / "임상도 51810(2025 인제)" / "51810.shp"}
PLAN_XLS = next(SRC.glob("(조회용)경영계획부_인제_9차_*.xls"))
STANDS = next((SRC / "인제(9)_20260921임소반도").glob("*.shp"))
PER_CLASS = {"no_change": 1000, "harvest": 500, "conversion": 300, "gain": 300}   # 평가 전용(춘천 비율 유지, 1/3 규모)
CODE = {c: i for i, c in enumerate(TRAIN_CLASSES)}


def load_map(year):
    g = gpd.read_file(MAPS[year], columns=["FRTP_CD", "KOFTR_GROU", "AGCLS_CD", "DNST_CD", "갱신년도"], engine="pyogrio")
    g = g.rename(columns={"FRTP_CD": "FRTP", "KOFTR_GROU": "KOFTR", "AGCLS_CD": "AGCLS", "DNST_CD": "DNST", "갱신년도": "UPDATE_YEAR"})
    g["geometry"] = g.geometry.make_valid()
    return g.to_crs(C.CRS_WORK)


def change_polygons():
    p = OUT / "inje_change_polygons.parquet"
    if p.exists():
        return gpd.read_parquet(p)
    t = time.time()
    a = load_map(2019).drop(columns=["UPDATE_YEAR"])
    b = load_map(2025)
    a.columns = [f"{c}_2019" if c != "geometry" else c for c in a.columns]
    b.columns = [f"{c}_2025" if c not in ("geometry", "UPDATE_YEAR") else c for c in b.columns]
    print(f"임상도 2019 {len(a):,} / 2025 {len(b):,} 폴리곤 중첩 중…", flush=True)
    ov = gpd.overlay(a, b, how="intersection", keep_geom_type=True).explode(index_parts=False, ignore_index=True)
    ov["area_ha"] = ov.area / 1e4
    ov = classify(ov[ov["area_ha"] >= C.MIN_PIECE_HA].reset_index(drop=True), 2019, 2025)
    ov.to_parquet(p)
    print(f"중첩 {time.time() - t:.0f}s, 조각 {len(ov):,}개", flush=True)
    return ov


def sample_class(polys, n, rng):
    inner = polys.copy()
    inner["geometry"] = inner.buffer(-C.INNER_BUFFER_M)
    inner = inner[~inner.is_empty & inner.is_valid & (inner.area > 100)]
    if inner.empty:
        return inner
    w = inner.area.to_numpy()
    pick = rng.choice(len(inner), size=min(n * 2, max(n, len(inner) * 5)), p=w / w.sum())
    chosen = inner.iloc[pick]
    out = chosen.drop(columns="geometry").set_geometry(chosen.sample_points(1, rng=rng).values, crs=inner.crs)
    key = (out.geometry.x // 10).astype(int).astype(str) + "_" + (out.geometry.y // 10).astype(int).astype(str)
    return out[~key.duplicated()].head(n)


def samples(cp):
    rng = np.random.default_rng(C.SEED)
    s = pd.concat([sample_class(cp[cp["change_type"] == c], PER_CLASS[c], rng) for c in TRAIN_CLASSES], ignore_index=True)
    s = gpd.GeoDataFrame(s, geometry="geometry", crs=C.CRS_WORK)
    s["block_id"] = "B" + (s.geometry.x // C.BLOCK_M).astype(int).astype(str) + "_" + (s.geometry.y // C.BLOCK_M).astype(int).astype(str)
    ll = s.to_crs(C.CRS_GEE).geometry
    s["lon"], s["lat"] = ll.x.round(6), ll.y.round(6)
    s["sample_id"] = [f"I{i:05d}" for i in range(len(s))]
    return s


def events():
    """국유림 수확·갱신 사건(2020~2024): A=임목생산 주벌, B=조림예정지정리·인공갱신. 소반 내부 최대 5점."""
    rows = []
    for sh, kinds in (("임목생산실적", {"주벌"}), ("조림실적", {"조림예정지정리", "인공갱신"})):
        d = pd.read_excel(PLAN_XLS, sheet_name=sh).iloc[1:]
        d = d[d["종별"].isin(kinds) & pd.to_numeric(d["연도"], errors="coerce").between(2020, 2024)]
        rows.append(pd.DataFrame({"cmblc_id": d["임반"], "sblt_id": d["소반"], "year": d["연도"].astype(int),
                                  "grade": np.where(d["종별"].eq("주벌"), "A", "B"), "kind": d["종별"]}))
    ev = pd.concat(rows).sort_values("year").groupby(["cmblc_id", "sblt_id"], as_index=False).first()
    st = pyogrio.read_dataframe(STANDS, encoding="latin1", columns=["cmblc_id", "sblt_id"]).to_crs(C.CRS_WORK)
    st = st.merge(ev, on=["cmblc_id", "sblt_id"])
    st["geometry"] = st.buffer(-C.INNER_BUFFER_M)
    st = st[~st.is_empty & (st.area > 100)]
    rng = np.random.default_rng(C.SEED)
    pts = []
    for r in st.itertuples():
        n = int(min(5, max(1, r.geometry.area // 2000)))
        for p in gpd.GeoSeries([r.geometry], crs=C.CRS_WORK).sample_points(n, rng=rng).explode(index_parts=False):
            pts.append({"unit": f"{r.cmblc_id}/{r.sblt_id}", "year": r.year, "grade": r.grade, "geometry": p})
    e = gpd.GeoDataFrame(pts, crs=C.CRS_WORK)
    ll = e.to_crs(C.CRS_GEE).geometry
    e["lon"], e["lat"] = ll.x.round(6), ll.y.round(6)
    e["sample_id"] = [f"E{i:05d}" for i in range(len(e))]
    return e


def payload(df, cols, name):
    """gzip+base64 CSV — Colab 셀 코드에 그대로 붙여 넣는다."""
    raw = df[cols].to_csv(index=False).encode()
    txt = base64.b64encode(gzip.compress(raw, 9)).decode()
    (OUT / f"payload_{name}.txt").write_text(txt)
    print(f"payload_{name}: {len(df):,}행, {len(txt) / 1024:.0f} KB", flush=True)


def main():
    cp = change_polygons()
    print(cp.groupby("change_type")["area_ha"].agg(["count", "sum"]).round(1).to_string(), flush=True)
    s = samples(cp)
    s.drop(columns="geometry").to_parquet(OUT / "inje_samples.parquet")
    print(s["change_type"].value_counts().to_string(), "| 블록", s["block_id"].nunique(), flush=True)
    e = events()
    e.drop(columns="geometry").to_parquet(OUT / "inje_events.parquet")
    print(f"사건 소반 {e['unit'].nunique()}개, 점 {len(e)}개", e.groupby("grade")["unit"].nunique().to_dict(), flush=True)

    ch = pd.read_parquet(C.SAMPLES)[["sample_id", "change_type", "change_year", "block_id", "split", "lon", "lat"]]
    for d in (ch, s):
        d["cls"] = d["change_type"].map(CODE).astype(int)
    s["split"], s["change_year"] = "inje", np.nan
    payload(ch, ["sample_id", "cls", "change_year", "block_id", "split", "lon", "lat"], "chuncheon")
    payload(s, ["sample_id", "cls", "change_year", "block_id", "split", "lon", "lat"], "inje")
    payload(e, ["sample_id", "unit", "year", "grade", "lon", "lat"], "inje_events")


if __name__ == "__main__":
    main()
