"""소반별 변화 궤적·해석 근거 구축 (build_db.py 다음에 실행).

임베딩 각 차원의 물리적 의미를 정하지 않고, 아래 근거를 결합해 '탐지된 변화가 어떤 산림변화와 부합하는지' 설명한다.
  관측 : AlphaEarth 연속연도 임베딩 거리(변화 시기·강도), Sentinel-2 NDVI·NDMI·NDBI 연간 추이(변화 방향·지속성)
  맥락 : 임상도 2019·2025 우세 속성(임상·수종·영급·밀도), 경영실적, 재해 이력
변화 화소(AE 변화 화소가 5개 이상)가 있으면 그 화소만으로, 없으면 소반 전체로 연간 값을 낸다(소반 평균 희석 완화).

추가 테이블 (forest_decision.sqlite)
  소반_연별    : 단위ID, 연도(2017~2025), AE_연속거리(전년→당해), AE_고변화비율, NDVI, NDMI, NDBI, NDWI, 집계범위
  소반_임베딩  : 단위ID, 연도(2019·2025), A00~A63 (소반 평균 단위벡터) — 유사 지역 검색용
  소반_경로    : 단위ID, 변화경로, 변화구간, NDVI_변화전/최저/2025, 회복률, NDMI_감소, NDBI_증가, 반복교란, 고변화연도,
                 임상·수종·영급·밀도(2019·2025), 임상도_속성변화, 관측/해석/추가확인/불확실성 문장
  임베딩_지수_관계 : 임베딩 차원과 지수의 Spearman 상관(탐색용, 차원 '명명' 근거 아님)
판정 기준값은 춘천 자료로 정한 잠정값이며, 지역 검증 전에는 '부합' 수준의 해석으로만 쓴다.
"""
import json
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "forest-change-agent" / "prep"))   # combine_thinning_forest_type
from combine_thinning_forest_type import DNST, FRTP, KOFTR, norm_code  # noqa: E402

OUT = ROOT / "output"
DB_DIR = OUT / "의사결정DB"
DB, GPKG = DB_DIR / "forest_decision.sqlite", DB_DIR / "forest_decision.gpkg"
AE_DIST = OUT / "Chuncheon_AE_consec_dist_2017_2025.tif"
AE_FILE = OUT / "Chuncheon_AE_{}_64Bands_int8.tif"
S2 = OUT / "Chuncheon_S2_2017_2025_54Bands.tif"
MASK = OUT / "Chuncheon_forestmask_AE_10m.tif"
YEARS = list(range(2017, 2026))
IDX = ["NDVI", "NDMI", "NDBI", "NDWI"]
DROP, NDMI_DROP, RISE, BUILT, REC = 0.08, 0.06, 0.05, 0.05, 0.5     # 잠정 판정 기준
MIN_CHG_PX = 5


def zonal_mean(zid, arr, n, sel=None):
    a = arr.ravel().astype("float64")
    z = zid.ravel()
    ok = np.isfinite(a) & (z > 0)
    if sel is not None:
        ok &= sel.ravel()
    s = np.bincount(z[ok], weights=a[ok], minlength=n + 1)[1:]
    c = np.bincount(z[ok], minlength=n + 1)[1:]
    return np.where(c > 0, s / np.maximum(c, 1), np.nan), c


def dominant(units, year):
    f = gpd.read_file(ROOT / "forest-change-agent" / "data" / f"임상도 {year} 51110" / "51110.shp", encoding="cp949",
                      columns=["FRTP_CD", "KOFTR_GROU", "AGCLS_CD", "DNST_CD", "geometry"])
    for c in ("FRTP_CD", "KOFTR_GROU", "AGCLS_CD", "DNST_CD"):
        f[c] = f[c].map(norm_code)
    f = f.to_crs(units.crs)
    f["geometry"] = f.geometry.make_valid()
    x = gpd.overlay(units[["단위ID", "geometry"]], f, how="intersection", keep_geom_type=True)
    x["a"] = x.area
    out = {}
    for col, name, table in [("FRTP_CD", "임상", FRTP), ("KOFTR_GROU", "수종", KOFTR), ("AGCLS_CD", "영급", None), ("DNST_CD", "밀도", DNST)]:
        d = x.groupby(["단위ID", col])["a"].sum().reset_index().sort_values("a").groupby("단위ID").last()[col]
        out[f"{name}_{year}"] = d.map(table) if table else d.map(lambda v: f"{v}영급" if v and v != "0" else None)
    return pd.DataFrame(out)


def classify(r):
    """관측 패턴 → 가능한 해석 (판정 규칙이 아니라 근거 정리용)."""
    if not r["변화관측"]:
        if pd.notna(r["NDVI_추세"]) and r["NDVI_추세"] >= RISE:
            return "식생 증가·성장 추정"
        return "안정"
    if r["NDVI_감소"] >= DROP:
        if r["정점연도"] >= 2024:
            return "최근 교란(회복 판단 이름)"
        if r["회복률"] >= REC:
            return "교란 후 회복"
        if r.get("FM_주변화유형") == "산지전용" or (pd.notna(r["NDBI_증가"]) and r["NDBI_증가"] >= 2 * BUILT and r["NDVI_2025"] < 0.5):
            # 벌채 직후에도 NDBI 가 오르므로, 학습모형 전용 판정 또는 큰 NDBI 증가 + 2025년까지 낮은 NDVI 일 때만
            return "지속 손실·전환 의심"
        return "교란 후 회복 지연"
    if pd.notna(r["NDMI_감소"]) and r["NDMI_감소"] >= NDMI_DROP:
        return "수분 스트레스 등 식생 이상 의심"
    if r["NDVI_후기상승"] >= RISE:
        return "식생 증가·성장 (변화 신호 동반)"
    return "임베딩 변화·지수 안정"


INTERP = {
    "교란 후 회복": ("교란(벌채·산불 등) 후 식생이 회복되는 과정과 부합한다.", "벌채·산불·조림 기록, 조림 수종과 원래 수종의 차이"),
    "교란 후 회복 지연": ("교란 후 식생 회복이 더디게 진행되는 상태와 부합한다.", "조림·복구 사업 시행 여부, 토사 유출, 현장 활착 상태"),
    "지속 손실·전환 의심": ("산림이 다른 피복(시설·도로·나지)으로 지속적으로 바뀐 상태와 부합한다.", "고해상도 영상, 산지전용 인허가, 토지이용 자료"),
    "최근 교란(회복 판단 이름)": ("최근 교란이 관측되었으나 이후 경과를 판단할 자료가 부족하다.", "다음 해 영상, 벌채·재해 기록"),
    "수분 스트레스 등 식생 이상 의심": ("녹색도보다 수분 신호가 더 크게 줄어, 수분 스트레스·병해충 등 식생 이상 가능성이 있다.", "강수·기온, 병해충 조사, 관측 품질(구름 잔재)"),
    "식생 증가·성장 (변화 신호 동반)": ("식생이 늘어나는 변화(조림지 생장, 산림증가)와 부합한다.", "초지·관목·산림 구분, 조림 기록"),
    "임베딩 변화·지수 안정": ("식생지수로는 설명되지 않는 구조·피복 변화이거나 자료 영향일 수 있다.", "원영상, 관측 조건, 다른 지표"),
    "식생 증가·성장 추정": ("변화 신호는 약하지만 식생지수가 꾸준히 높아져 생장과 부합한다.", "영급·밀도 변화"),
    "안정": ("뚜렷한 변화 신호가 없다.", "임상도가 그대로라는 사실만으로 무변화를 확정하지 않음"),
}


def main():
    meta = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    t_ae = meta["임계값"]["AE_거리"]
    units = gpd.read_file(GPKG, layer="소반")
    n = len(units)
    ids = units["단위ID"].to_numpy()

    # --- AlphaEarth 연속연도 거리 (10 m)
    with rasterio.open(AE_DIST) as ds:
        ae_crs, ae_tf, shp = ds.crs, ds.transform, (ds.height, ds.width)
        zid = rasterize(zip(units.to_crs(ae_crs).geometry, range(1, n + 1)), out_shape=shp, transform=ae_tf, fill=0, dtype="int32")
        bands = {int(d.split("_")[1]): i + 1 for i, d in enumerate(ds.descriptions)}     # d2017_2018 → 2018
        dist = {}
        for y, b in bands.items():
            a = ds.read(b).astype("float32")
            a[(a == ds.nodata) | (a <= 0)] = np.nan
            dist[y] = a
    chg = np.zeros(shp, bool)
    for y in range(2020, 2026):
        chg |= np.nan_to_num(dist[y]) > t_ae
    nchg = np.bincount(zid.ravel()[chg.ravel()], minlength=n + 1)[1:]
    use_chg = nchg >= MIN_CHG_PX
    rows = []
    for y in range(2018, 2026):
        m_all, _ = zonal_mean(zid, dist[y], n)
        m_chg, _ = zonal_mean(zid, dist[y], n, chg)
        hi, cnt = zonal_mean(zid, (np.nan_to_num(dist[y]) > t_ae).astype("float32") + np.where(np.isfinite(dist[y]), 0, np.nan), n)
        rows.append(pd.DataFrame({"단위ID": ids, "연도": y, "AE_연속거리": np.where(use_chg, m_chg, m_all), "AE_고변화비율": hi}))
    del dist
    # --- Sentinel-2 연간 지수 (20 m): 변화 마스크를 20 m 로 (max)
    with rasterio.open(S2) as ds:
        s2_crs, s2_tf, s2_shp = ds.crs, ds.transform, (ds.height, ds.width)
        desc = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        zid2 = rasterize(zip(units.to_crs(s2_crs).geometry, range(1, n + 1)), out_shape=s2_shp, transform=s2_tf, fill=0, dtype="int32")
        chg2 = np.zeros(s2_shp, "uint8")
        reproject(chg.astype("uint8"), chg2, src_transform=ae_tf, src_crs=ae_crs, dst_transform=s2_tf, dst_crs=s2_crs, resampling=Resampling.max)
        chg2 = chg2.astype(bool)
        nchg2 = np.bincount(zid2.ravel()[chg2.ravel()], minlength=n + 1)[1:]
        use2 = use_chg & (nchg2 >= 2)
        s2rows = []
        for y in YEARS:
            d = {"단위ID": ids, "연도": y}
            for k in IDX:
                a = ds.read(desc[f"{k}_{y}"], masked=True).filled(np.nan).astype("float32")
                ma, _ = zonal_mean(zid2, a, n)
                mc, _ = zonal_mean(zid2, a, n, chg2)
                d[k] = np.where(use2, mc, ma)
                d[f"{k}_전체"] = ma
            s2rows.append(pd.DataFrame(d))
    yr = pd.DataFrame({"단위ID": np.repeat(ids, len(YEARS)), "연도": np.tile(YEARS, n)})
    yr = yr.merge(pd.concat(rows), on=["단위ID", "연도"], how="left").merge(pd.concat(s2rows), on=["단위ID", "연도"], how="left")
    yr["집계범위"] = np.repeat(np.where(use2, "변화 화소", "소반 전체"), len(YEARS))

    # --- 소반 평균 임베딩 (2019·2025) — 밴드별로 읽어 메모리 절약
    embrows = []
    for y in (2019, 2025):
        with rasterio.open(Path(str(AE_FILE).format(y))) as ds:
            E = np.zeros((n, 64))
            for b in range(64):
                a = ds.read(b + 1).astype("float32")
                a[a == -128] = np.nan
                E[:, b], _ = zonal_mean(zid, a / 127.0, n)
        E = E / np.linalg.norm(np.nan_to_num(E), axis=1, keepdims=True).clip(1e-9)
        embrows.append(pd.DataFrame(E, columns=[f"A{i:02d}" for i in range(64)]).assign(단위ID=ids, 연도=y))
    emb = pd.concat(embrows)

    # --- 임베딩 차원–지수 관계 (탐색): 산림 화소 3,000개 표본
    from scipy.stats import spearmanr
    with rasterio.open(MASK) as ds:
        fm = ds.read(1) == 1
    rng = np.random.default_rng(42)
    rr, cc = np.nonzero(fm)
    pick = rng.choice(len(rr), 3000, replace=False)
    rr, cc = rr[pick], cc[pick]
    xs, ys = rasterio.transform.xy(ae_tf, rr, cc)
    rel = []
    vals = {}
    for y in (2019, 2025):
        with rasterio.open(Path(str(AE_FILE).format(y))) as ds:
            v = np.array(list(ds.sample(zip(xs, ys))), dtype="float32")
        v[v == -128] = np.nan
        vals[y] = v / 127.0
    with rasterio.open(S2) as ds:
        desc = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        sv = {f"{k}_{y}": np.array([p[0] for p in ds.sample(zip(xs, ys), indexes=desc[f"{k}_{y}"])], dtype="float32")
              for k in ("NDVI", "NDMI", "NDBI") for y in (2019, 2025)}
    for k in ("NDVI", "NDMI", "NDBI"):
        for kind, X, Y in [("2025 수준", vals[2025], sv[f"{k}_2025"]), ("2019→2025 변화", vals[2025] - vals[2019], sv[f"{k}_2025"] - sv[f"{k}_2019"])]:
            ok = np.isfinite(Y) & np.isfinite(X).all(1) & (Y != -9999)
            for i in range(64):
                rho = spearmanr(X[ok, i], Y[ok]).statistic
                rel.append({"차원": f"A{i:02d}", "지수": k, "구분": kind, "rho": round(float(rho), 3), "표본": int(ok.sum())})
    rel = pd.DataFrame(rel)

    # --- 경로 판정
    u = pd.read_sql('select * from "소반"', sqlite3.connect(DB))
    tr = []
    for uid, g in yr.groupby("단위ID", sort=False):
        g = g.set_index("연도")
        d = g["AE_연속거리"].loc[2018:2025]
        peak = int(d.idxmax()) if d.notna().any() else None
        nd = g["NDVI"]
        hi_years = [int(y) for y, v in g["AE_고변화비율"].loc[2018:2025].items() if v >= 0.10]
        groups = 0
        for i, y in enumerate(hi_years):
            if i == 0 or y - hi_years[i - 1] > 1:
                groups += 1
        if peak:
            pre = nd.loc[:peak - 1].mean()
            post = nd.loc[peak:min(peak + 1, 2025)].min()
            last = nd.loc[2025]
            drop = pre - post
            rec = (last - post) / drop if drop > 0.01 else np.nan
            ndmi_drop = g["NDMI"].loc[:peak - 1].mean() - g["NDMI"].loc[peak:min(peak + 1, 2025)].min()
            ndbi_up = g["NDBI"].loc[2025] - g["NDBI"].loc[:peak - 1].mean()
            late_rise = last - pre
        else:
            pre = post = last = drop = rec = ndmi_drop = ndbi_up = late_rise = np.nan
        tr.append({"단위ID": uid, "정점연도": peak, "변화구간": f"{peak - 1}→{peak}" if peak else None,
                   "정점거리": round(float(d.max()), 4) if peak else None,
                   "NDVI_변화전": round(float(pre), 3), "NDVI_최저": round(float(post), 3), "NDVI_2025": round(float(last), 3),
                   "NDVI_감소": round(float(drop), 3), "회복률": round(float(rec), 2) if pd.notna(rec) else None,
                   "NDMI_감소": round(float(ndmi_drop), 3), "NDBI_증가": round(float(ndbi_up), 3), "NDVI_후기상승": round(float(late_rise), 3),
                   "NDVI_추세": round(float(g["NDVI_전체"].loc[2025] - g["NDVI_전체"].loc[2019]), 3),
                   "고변화연도": ",".join(map(str, hi_years)), "반복교란": groups >= 2, "집계범위": g["집계범위"].iloc[0]})
    tr = pd.DataFrame(tr).merge(u[["단위ID", "판정", "FM_변화면적_ha", "AE_변화면적_ha", "AE_변화비율", "FM_주변화유형", "FM_변화연도",
                                  "실적구분_2020_2025", "실적연도", "실적종별", "임상도_변화비율", "합의라벨", "산불_500m_2019_2025",
                                  "산사태_500m_2019_2025"]], on="단위ID")
    tr["변화관측"] = (tr["FM_변화면적_ha"] >= 0.5) | ((tr["AE_변화면적_ha"] >= 0.5) & (tr["AE_변화비율"] >= 0.15))
    tr["NDVI_감소"] = tr["NDVI_감소"].fillna(0)
    tr["회복률"] = tr["회복률"].fillna(0)
    tr["NDVI_후기상승"] = tr["NDVI_후기상승"].fillna(0)
    tr["변화경로"] = tr.apply(classify, axis=1)
    attrs = units[["단위ID", "geometry"]].merge(dominant(units, 2019), left_on="단위ID", right_index=True, how="left") \
                                       .merge(dominant(units, 2025), left_on="단위ID", right_index=True, how="left")
    tr = tr.merge(pd.DataFrame(attrs.drop(columns="geometry")), on="단위ID", how="left")
    tr["임상도_속성변화"] = (tr["임상_2019"] != tr["임상_2025"]) | (tr["수종_2019"] != tr["수종_2025"])

    def sentences(r):
        f = lambda v: "-" if pd.isna(v) else v
        obs = (f"AlphaEarth 임베딩 변화는 {r['변화구간']}에 가장 컸고(거리 {f(r['정점거리'])}), NDVI는 변화 전 {f(r['NDVI_변화전'])} → 최저 {f(r['NDVI_최저'])} → 2025년 {f(r['NDVI_2025'])}"
               f"(회복률 {f(r['회복률'])}), NDMI 감소 {f(r['NDMI_감소'])}, NDBI 증가 {f(r['NDBI_증가'])}였다."
               + (f" 고변화 연도: {r['고변화연도']}." if r["고변화연도"] else "")
               + (f" 학습모형은 {r['FM_주변화유형']}({f(r['FM_변화연도'])}년)으로 분류하였다." if isinstance(r["FM_주변화유형"], str) else ""))
        rec_ = (f"사업실적: {r['실적구분_2020_2025'] or '실적없음'}"
                + (f"({r['실적연도']}, {r['실적종별']})" if isinstance(r["실적연도"], str) else "")
                + f"; 임상도 2019 {f(r['임상_2019'])}·{f(r['수종_2019'])}·{f(r['영급_2019'])} → 2025 {f(r['임상_2025'])}·{f(r['수종_2025'])}·{f(r['영급_2025'])}"
                + f" (변화비율 {f(r['임상도_변화비율'])}); 500 m 내 산불 {r['산불_500m_2019_2025']}·산사태 {r['산사태_500m_2019_2025']}건.")
        it, chk = INTERP[r["변화경로"]]
        unc = [f"연간 값은 {r['집계범위']} 평균(20 m 지수, 10 m 임베딩)"]
        if r["집계범위"] == "소반 전체":
            unc.append("변화가 소반 일부면 신호가 희석됨")
        if r["반복교란"]:
            unc.append("여러 해에 걸친 반복 변화 — 집중 감시 대상")
        unc.append("판정 기준값은 춘천 자료의 잠정값")
        return pd.Series({"관측": obs, "기록": rec_, "해석": it, "추가확인": chk, "불확실성": "; ".join(unc)})

    tr = pd.concat([tr, tr.apply(sentences, axis=1)], axis=1)
    keep = ["단위ID", "변화경로", "변화관측", "변화구간", "정점연도", "정점거리", "NDVI_변화전", "NDVI_최저", "NDVI_2025", "NDVI_감소", "회복률",
            "NDMI_감소", "NDBI_증가", "NDVI_후기상승", "NDVI_추세", "고변화연도", "반복교란", "집계범위", "임상_2019", "수종_2019", "영급_2019", "밀도_2019",
            "임상_2025", "수종_2025", "영급_2025", "밀도_2025", "임상도_속성변화", "관측", "기록", "해석", "추가확인", "불확실성"]
    with sqlite3.connect(DB) as con:
        for name, df in [("소반_연별", yr.drop(columns=[f"{k}_전체" for k in IDX])), ("소반_임베딩", emb), ("소반_경로", tr[keep]), ("임베딩_지수_관계", rel)]:
            df.to_sql(name, con, index=False, if_exists="replace")
    m = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    m["변화경로_분포"] = tr["변화경로"].value_counts().to_dict()
    m["반복교란_소반"] = int(tr["반복교란"].sum())
    m["경로_기준"] = {"NDVI_감소": DROP, "NDMI_감소": NDMI_DROP, "상승": RISE, "NDBI_증가": BUILT, "회복률": REC}
    (DB_DIR / "build_meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tr["변화경로"].value_counts().to_string())
    print("반복교란", int(tr["반복교란"].sum()), "| 임베딩-지수 |rho| 최대", rel.assign(a=rel.rho.abs()).sort_values("a").tail(3)[["차원", "지수", "구분", "rho"]].to_dict("records"))


if __name__ == "__main__":
    main()
