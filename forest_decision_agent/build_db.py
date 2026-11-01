"""의사결정 근거 DB 구축: 위성 변화신호(AlphaEarth·Sentinel-2) + 행정기록 + 재해 + 지형·접근성 → SQLite + GPKG.

분석 단위: 국유림 9차 소반(1,136개, 춘천_임소반_산림사업실적.gpkg) + 산지전용 고시 필지(516개)
위성 신호 (2019→2025)
  AlphaEarth : 연속 연도 임베딩 거리(Chuncheon_AE_consec_dist_2017_2025.tif)의 2019~2025 최댓값, 최대 연도
  Sentinel-2 : 연간 NDVI 최대 감소(−min ΔNDVI), 2025−2019 ΔNDVI
  변화 화소 = 검증된 무변화 산림 화소(TRAIN_P2 = 0) 상위 5% 를 넘는 값 (두 신호 모두 같은 기준 → 오탐률 5%)
행정·환경: 2020~2025 사업실적(합의라벨 비교 결과), 산불·산사태 2019~2025(500 m 이내), 임도·도로 거리, 경사·표고
판정(코드 규칙, 변화관측 = AE 변화면적 ≥ 0.5 ha & 변화비율 ≥ 0.15): 기록일치 / 미기록 변화(점검) / 기타실적·변화 / 실적·변화 미관측(이행확인) / 안정

출력: output/의사결정DB/forest_decision.sqlite, forest_decision.gpkg, build_meta.json
"""
import json
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DB_DIR = OUT / "의사결정DB"
GIS = ROOT / "forest-change-agent" / "data" / "춘천_임도_도로.gpkg"   # 임도·도로 (원본 C:/src/ai_test/공간데이터/춘천_산림공간정보.gpkg 의 두 레이어)
AE_DIST = OUT / "Chuncheon_AE_consec_dist_2017_2025.tif"
S2 = OUT / "Chuncheon_S2_2019_2025_42Bands.tif"
LAB_AE, LAB_S2 = OUT / "Chuncheon_labels_AE_10m.tif", OUT / "Chuncheon_labels_S2_20m.tif"
AE_BANDS = {2020: "d2019_2020", 2021: "d2020_2021", 2022: "d2021_2022", 2023: "d2022_2023", 2024: "d2023_2024", 2025: "d2024_2025"}
CHANGE_HA = 0.5          # 변화 관측 기준 (임상도 최소 구획 0.5 ha)
CHANGE_RATIO = 0.15      # 그리고 소반 화소의 15% 이상 (무변화 소반 중앙값 0.026 의 약 6배, 수확실적 소반 29/31 포함)
CRS = 5186
# 학습된 변화탐지 모델 결과 (Colab 학습 → 내려받음)
TSFM_DIR = ROOT / "forest-change-agent" / "models" / "s2_tsfm_chuncheon"            # Sentinel-2 S2-TSFM (20 m, 2019~2025 42밴드, 지수+GFM 임베딩 분류기)
FM_DIR = ROOT / "final_change_2019_2025"        # AlphaEarth + S2 지수 최종 변화지도 (10 m, 산림 마스크·최소면적 적용)
MASK_S2 = OUT / "Chuncheon_forestmask_S2_20m.tif"
CLS = {1: "벌채·갱신", 2: "산지전용", 3: "산림증가"}


def ae_signal():
    with rasterio.open(AE_DIST) as ds:
        idx = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        mx = np.full((ds.height, ds.width), np.nan, "float32")
        yr = np.zeros((ds.height, ds.width), "int16")
        for y, b in AE_BANDS.items():
            a = ds.read(idx[b]).astype("float32")
            a[(a == ds.nodata) | (a <= 0)] = np.nan
            better = a > np.nan_to_num(mx, nan=-1)
            mx[better], yr[better] = a[better], y
        return mx, yr, ds.crs, ds.transform


def s2_signal():
    with rasterio.open(S2) as ds:
        idx = {d: i + 1 for i, d in enumerate(ds.descriptions)}
        nd = np.stack([ds.read(idx[f"NDVI_{y}"], masked=True).filled(np.nan) for y in range(2019, 2026)]).astype("float32")
        slope = ds.read(idx["Slope_2019"], masked=True).filled(np.nan)
        elev = ds.read(idx["Elevation_2019"], masked=True).filled(np.nan)
        with np.errstate(invalid="ignore"):
            drop = -np.nanmin(np.diff(nd, axis=0), axis=0)
        return drop, nd[0] - nd[-1], slope, elev, ds.crs, ds.transform


def model_layers():
    """모델 예측 래스터 → 지표 레이어. S2-TSFM 은 산림 마스크 밖을 0(변화 아님)으로, 최종지도는 이미 마스크·최소면적 적용."""
    with rasterio.open(FM_DIR / "final_change_class.tif") as d:
        fc, fm_crs, fm_tf = d.read(1), d.crs, d.transform
    with rasterio.open(FM_DIR / "final_change_prob.tif") as d:
        fp = 1 - d.read(1, masked=True).filled(np.nan)
    with rasterio.open(FM_DIR / "final_change_year.tif") as d:
        fy = d.read(1).astype("int16")
    with rasterio.open(TSFM_DIR / "change_class.tif") as d:
        mc, mg_crs, mg_tf = d.read(1), d.crs, d.transform
    with rasterio.open(TSFM_DIR / "change_prob.tif") as d:
        mp = 1 - d.read(1, masked=True).filled(np.nan)
    with rasterio.open(MASK_S2) as d:
        forest = d.read(1) == 1
    mc = np.where(forest, mc, 0)
    mp = np.where(forest, mp, np.nan)
    fm = {f"FM_{v}": (fc == k).astype("float32") for k, v in CLS.items()}
    fm["FM_변화확률"] = np.where(fc != 255, fp, np.nan).astype("float32")
    mg = {f"MG_{v}": (mc == k).astype("float32") for k, v in CLS.items()}
    mg["MG_변화확률"] = mp.astype("float32")
    return fm, fy, fc, fm_crs, fm_tf, mg, mg_crs, mg_tf


def threshold(score, label_path):
    with rasterio.open(label_path) as lb:
        p2 = lb.read(list(lb.descriptions).index("TRAIN_P2") + 1)
    v = score[(p2 == 0) & np.isfinite(score)]
    chg = score[np.isin(p2, [1, 2, 3]) & np.isfinite(score)]
    t = float(np.percentile(v, 95))
    return t, {"무변화화소": int(v.size), "변화화소": int(chg.size), "변화화소_탐지율": round(float((chg > t).mean()), 3)}


def zonal(gdf, crs, tf, shape, layers, all_touched=False):
    """gdf 행마다 화소 수, 각 레이어 평균(nan 제외). layers: {이름: 배열}."""
    g = gdf.to_crs(crs)
    zid = rasterize(zip(g.geometry, np.arange(1, len(g) + 1)), out_shape=shape, transform=tf, fill=0, dtype="int32",
                    all_touched=all_touched)
    flat = zid.ravel()
    n = len(g) + 1
    out = {"화소수": np.bincount(flat, minlength=n)[1:]}
    for k, a in layers.items():
        a = a.ravel().astype("float64")
        ok = np.isfinite(a) & (flat > 0)
        s = np.bincount(flat[ok], weights=a[ok], minlength=n)[1:]
        c = np.bincount(flat[ok], minlength=n)[1:]
        out[k] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out, zid


def mode_year(zid, yr, mask, n):
    out = np.zeros(n, int)
    f, y = zid.ravel(), yr.ravel()
    sel = mask.ravel() & (f > 0)
    if sel.any():
        tab = pd.crosstab(f[sel], y[sel])
        out[tab.index.values - 1] = tab.idxmax(axis=1).values
    return out


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    # 단위
    su = gpd.read_file(OUT / "춘천_임소반_산림사업실적.gpkg", layer="임소반_실적요약").to_crs(CRS)
    su["단위ID"] = su["임반"] + "/" + su["소반"]
    su["면적_ha"] = su.area / 1e4
    emd = gpd.read_file(OUT / "춘천시_읍면동경계.shp").to_crs(CRS)
    rp = su.representative_point()
    su["읍면동"] = gpd.sjoin(gpd.GeoDataFrame(geometry=rp, crs=CRS), emd[["EMD_NM", "geometry"]], how="left")["EMD_NM"].groupby(level=0).first()
    units = su[["단위ID", "임반", "소반", "읍면동", "lctn_daddr", "면적_ha", "geometry"]].rename(columns={"lctn_daddr": "소재지"})

    cmp_ = gpd.read_file(OUT / "춘천_경영실적_임상도_변화라벨_비교.gpkg", layer="단위별_비교", ignore_geometry=True)
    cmp_ = cmp_[cmp_["구간"] == "P2019_2025"][["단위ID", "실적라벨", "실적연도", "실적종별", "실적건수", "합의라벨", "임상도_변화비율"]]
    units = units.merge(cmp_.rename(columns={"실적라벨": "실적구분_2020_2025"}), on="단위ID", how="left")

    # 위성 신호
    ae, ae_yr, ae_crs, ae_tf = ae_signal()
    t_ae, cal_ae = threshold(ae, LAB_AE)
    s2d, s2_dndvi, slope, elev, s2_crs, s2_tf = s2_signal()
    t_s2, cal_s2 = threshold(s2d, LAB_S2)
    ae_chg = (ae > t_ae).astype("float32")
    ae_chg[~np.isfinite(ae)] = np.nan
    s2_chg = (s2d > t_s2).astype("float32")
    s2_chg[~np.isfinite(s2d)] = np.nan
    print(f"[임계값] AE 거리 {t_ae:.4f} {cal_ae} | S2 NDVI감소 {t_s2:.4f} {cal_s2}")

    def add_signals(gdf, all_touched):
        za, zid = zonal(gdf, ae_crs, ae_tf, ae.shape, {"AE_변화비율": ae_chg, "AE_거리평균": ae}, all_touched)
        zs, _ = zonal(gdf, s2_crs, s2_tf, s2d.shape, {"S2_변화비율": s2_chg, "S2_dNDVI": s2_dndvi, "경사_도": slope, "표고_m": elev}, all_touched)
        gdf = gdf.copy()
        gdf["AE_화소수"] = za["화소수"]
        for k in ("AE_변화비율", "AE_거리평균"):
            gdf[k] = za[k]
        for k in ("S2_변화비율", "S2_dNDVI", "경사_도", "표고_m"):
            gdf[k] = zs[k]
        gdf["AE_변화연도"] = mode_year(zid, ae_yr, ae > t_ae, len(gdf))
        gdf["AE_변화연도"] = gdf["AE_변화연도"].astype("Int64").replace(0, pd.NA)
        # 학습 모델: 최종지도(AE+S2, 10 m) · S2-TSFM(S2, 20 m)
        zf, zidf = zonal(gdf, fm_crs, fm_tf, fc.shape, fm, all_touched)
        zm, _ = zonal(gdf, mg_crs, mg_tf, mg["MG_변화확률"].shape, mg, all_touched)
        for pre, z, px in (("FM", zf, 0.01), ("MG", zm, 0.04)):
            tot = 0
            for v in CLS.values():
                a = np.nan_to_num(z[f"{pre}_{v}"]) * z["화소수"] * px
                gdf[f"{pre}_{v}_ha"] = a
                tot = tot + a
            gdf[f"{pre}_변화면적_ha"] = tot
            gdf[f"{pre}_변화확률"] = z[f"{pre}_변화확률"]
            dom = np.argmax(np.stack([gdf[f"{pre}_{v}_ha"] for v in CLS.values()]), axis=0)
            gdf[f"{pre}_주변화유형"] = np.where(tot > 0, np.asarray(list(CLS.values()))[dom], None)
        gdf["FM_변화연도"] = mode_year(zidf, fy, np.isin(fc, [1, 2, 3]), len(gdf))
        gdf["FM_변화연도"] = gdf["FM_변화연도"].astype("Int64").replace(0, pd.NA)
        return gdf

    fm, fy, fc, fm_crs, fm_tf, mg, mg_crs, mg_tf = model_layers()
    assert (fm_crs, fm_tf, fc.shape) == (ae_crs, ae_tf, ae.shape), "최종지도 격자가 AlphaEarth 격자와 다름"

    units = add_signals(units, False)
    units["AE_변화면적_ha"] = units["AE_변화비율"].fillna(0) * units["AE_화소수"] * 0.01
    units["S2_변화면적_ha"] = units["S2_변화비율"].fillna(0) * units["면적_ha"]

    # 재해·접근성
    fire = gpd.read_file(OUT / "춘천시_산불_산사태.gpkg", layer="산불").to_crs(CRS)
    slide = gpd.read_file(OUT / "춘천시_산불_산사태.gpkg", layer="산사태").to_crs(CRS)
    dis = pd.concat([fire.assign(유형="산불")[["유형", "연도", "geometry"]], slide.assign(유형="산사태")[["유형", "연도", "geometry"]]])
    dis = dis[pd.to_numeric(dis["연도"], errors="coerce").between(2019, 2025)].reset_index(drop=True)
    buf = units[["단위ID", "geometry"]].copy()
    buf["geometry"] = buf.buffer(500)
    hit = gpd.sjoin(dis, buf, predicate="within")
    cnt = hit.groupby(["단위ID", "유형"]).size().unstack(fill_value=0)
    for t in ("산불", "산사태"):
        units[f"{t}_500m_2019_2025"] = units["단위ID"].map(cnt[t] if t in cnt else {}).fillna(0).astype(int)
    road = gpd.read_file(GIS, layer="임도").to_crs(CRS)
    rd = gpd.read_file(GIS, layer="도로_1000").to_crs(CRS)
    cen = gpd.GeoDataFrame(units[["단위ID"]], geometry=units.representative_point(), crs=CRS)
    units["임도거리_m"] = gpd.sjoin_nearest(cen, road[["geometry"]], distance_col="d").groupby(level=0)["d"].min().round(0)
    units["도로거리_m"] = gpd.sjoin_nearest(cen, rd[["geometry"]], distance_col="d").groupby(level=0)["d"].min().round(0)
    dis = gpd.sjoin_nearest(dis, units[["단위ID", "geometry"]], distance_col="소반거리_m").drop(columns="index_right")

    # 판정 (규칙): 변화관측 = 학습 모델 최종지도(AlphaEarth+S2)의 변화면적 ≥ 0.5 ha
    #   신호일치수 = AE 임베딩거리 / S2-TSFM / 최종지도 중 변화관측으로 본 신호 수 (불확실성 지표)
    obs_ae = (units["AE_변화면적_ha"] >= CHANGE_HA) & (units["AE_변화비율"] >= CHANGE_RATIO)
    obs_mg = (units["MG_변화면적_ha"] >= CHANGE_HA) & (units["MG_변화면적_ha"] / units["면적_ha"] >= CHANGE_RATIO)
    obs = units["FM_변화면적_ha"] >= CHANGE_HA
    units["신호일치수"] = obs_ae.astype(int) + obs_mg.astype(int) + obs.astype(int)
    units["판정_AE거리"] = np.where(obs_ae, "변화", "안정")
    rec = units["실적구분_2020_2025"].fillna("실적없음")
    units["판정"] = np.select(
        [obs & rec.eq("수확·갱신"), obs & rec.eq("실적없음"), obs, rec.eq("수확·갱신")],
        ["기록일치(수확실적·변화관측)", "미기록 변화(점검 필요)", "기타실적·변화관측(확인)", "실적있으나 변화 미관측(이행 확인)"],
        default="안정")

    # 산지전용 필지
    conv = gpd.read_file(OUT / "산지전용" / "산지전용_라벨.gpkg").to_crs(CRS)
    conv = add_signals(conv[["레코드ID", "고시번호", "고시연도", "주소", "읍면동", "지목", "전용용도", "해제면적_m2",
                             "라벨", "임지비율_2019", "임지비율_2025", "geometry"]], True)

    # 사업실적 원장 (2019~2025)
    recs = gpd.read_file(OUT / "춘천_임소반_산림사업실적.gpkg", layer="산림사업실적", ignore_geometry=True)
    recs = recs[pd.to_numeric(recs["연도"], errors="coerce").between(2019, 2025)]
    recs = recs.assign(단위ID=recs["임반"] + "/" + recs["소반"])[["단위ID", "연도", "사업구분", "종별", "면적_ha", "사업비_천원"]]

    # 저장
    num = units.select_dtypes("float").columns
    units[num] = units[num].round(4)
    tables = {"소반": units, "산지전용": conv, "사업실적": recs, "재해": dis}
    gp = DB_DIR / "forest_decision.gpkg"
    gp.unlink(missing_ok=True)
    for name in ("소반", "산지전용", "재해"):
        tables[name].to_file(gp, layer=name, driver="GPKG")
    emd.to_file(gp, layer="읍면동경계", driver="GPKG")
    db = DB_DIR / "forest_decision.sqlite"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        for name, t in tables.items():
            df = pd.DataFrame(t.drop(columns="geometry", errors="ignore"))
            if name == "재해":
                df["x"], df["y"] = t.geometry.x.round(1), t.geometry.y.round(1)
            df.to_sql(name, con, index=False)
    meta = {"임계값": {"AE_거리": round(t_ae, 4), "S2_NDVI감소": round(t_s2, 4), "기준": "검증 무변화 산림화소 상위 5%"},
            "보정": {"AE": cal_ae, "S2": cal_s2}, "변화관측_기준": f"학습 모델 최종지도(AlphaEarth+S2) 변화면적 ≥ {CHANGE_HA} ha",
            "신호일치_기준": f"AE 임베딩거리·S2-TSFM 은 변화면적 ≥ {CHANGE_HA} ha 이고 비율 ≥ {CHANGE_RATIO}",
            "모델": {"최종지도": json.loads((FM_DIR / "final_report.json").read_text(encoding="utf-8"))["final_map_holdout"],
                   "S2-TSFM": json.loads((TSFM_DIR / "report.json").read_text(encoding="utf-8"))["classifier"]["지수+GFM"]},
            "신호일치수_분포": units["신호일치수"].value_counts().sort_index().to_dict(),
            "판정_분포": units["판정"].value_counts().to_dict(), "건수": {k: len(v) for k, v in tables.items()}}
    (DB_DIR / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
