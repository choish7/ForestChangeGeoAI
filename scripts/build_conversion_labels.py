"""보전산지 해제 고시(산지전용) 필지 → 산지전용 라벨 데이터 + 라벨 검증 테스트.

입력: output/산지전용/춘천시_산지전용_보전산지해제.gpkg (build_mountain_conversion_dataset.py, 연속지적도 결합 필지)
라벨 (필지 단위, 영상 기간 2019~2025 기준)
  1 전용확정 : 고시 2020~2024 + 임상도 2019 임지비율 ≥ 0.5 + 2025 < 0.5 (고시와 임상도가 모두 전용 확인)
               + 전체 해제 + 지적면적 대조 ±2% (필지 모양이 고시 당시와 같음)
  2 전용후보 : 고시 2020~2024 중 위 조건 미충족 (2019 이미 비임지, 2025 임지 잔존, 일부 해제, 필지 분할 등)
  3 전용이전 : 고시 2017~2019 — 전용이 2019 무렵 이전에 끝나 2019~2025 에는 변화가 적어야 하는 대조군
  발생연도 상한 = 고시연도 (보전산지 해제는 전용 완료 후 고시)
래스터: Sentinel 20 m(Chuncheon_labels_S2_20m 격자)·AlphaEarth 10 m 격자, 화소 중심이 필지 안이면 그 라벨 (255 = 없음)

테스트 (로컬 Sentinel-2 2019~2025 42밴드 지수)
  - 양성: 전용확정 화소, 음성: 합의 라벨의 검증된 무변화 산림 화소(TRAIN_P2 = 0)
  - 점수: −ΔNDVI(2025−2019), 최대 연간 NDVI 감소, −ΔNDMI, ΔNDBI → AUROC·AP, 음성 상위 1%/5% 임계값에서 필지 탐지율
  - 전용이전 필지는 2019~2025 변화가 작아야 함 (대조)
  - 기존 임상도 변화 라벨(TRAIN_P2 = 2 산지전용)과의 겹침

출력: output/산지전용/
  산지전용_라벨.gpkg (필지 + 라벨·점수), Chuncheon_convlabels_S2_20m.tif, Chuncheon_convlabels_AE_10m.tif,
  산지전용_라벨_테스트.json / .csv
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_training_label_rasters import AE_GRID  # noqa: E402

D = ROOT / "output" / "산지전용"
PARCELS = D / "춘천시_산지전용_보전산지해제.gpkg"
S2 = ROOT / "output" / "Chuncheon_S2_2019_2025_42Bands.tif"
LAB_S2 = ROOT / "output" / "Chuncheon_labels_S2_20m.tif"
CODES = {"전용확정": 1, "전용후보": 2, "전용이전": 3}


def label_parcels():
    g = gpd.read_file(PARCELS, layer="산지전용_필지")
    g = g[g.geometry.notna()].copy()
    g["고시연도"] = g["고시번호"].str.extract(r"(\d{4})")[0].astype(int)
    shape_ok = g["지적면적_대조"].between(0.98, 1.02)
    full = g["해제구분"].eq("전체")
    confirmed = (g["임지비율_2019"] >= 0.5) & (g["임지비율_2025"] < 0.5)
    recent = g["고시연도"].between(2020, 2024)
    g["라벨"] = np.select([recent & confirmed & full & shape_ok, recent, g["고시연도"] <= 2019],
                         ["전용확정", "전용후보", "전용이전"], default="전용후보")
    g["라벨코드"] = g["라벨"].map(CODES)
    g["발생연도_상한"] = g["고시연도"]
    g["후보사유"] = np.where(g["라벨"].ne("전용후보"), "", np.select(
        [~shape_ok, ~full, g["임지비율_2019"] < 0.5, g["임지비율_2025"] >= 0.5],
        ["필지 모양 변경(면적 대조 벗어남)", "일부 해제", "2019 이미 비임지", "2025 임지 잔존"], default="기타"))
    # 같은 필지가 여러 고시에 나오면 가장 이른 고시 기준 하나로 (래스터용)
    g = g.sort_values(["고시연도", "레코드ID"])
    return g


def burn(g, crs, tf, w, h):
    gg = g.drop_duplicates("PNU").to_crs(crs).sort_values("라벨코드", ascending=False)  # 확정이 위에 오도록
    shapes = list(zip(gg.geometry, gg["라벨코드"]))
    lab = rasterize(shapes, out_shape=(h, w), transform=tf, fill=255, dtype="uint8")
    yr = rasterize(list(zip(gg.geometry, gg["발생연도_상한"] - 2000)), out_shape=(h, w), transform=tf, fill=255, dtype="uint8")
    return lab, yr


def write(path, lab, yr, crs, tf):
    prof = dict(driver="GTiff", width=lab.shape[1], height=lab.shape[0], count=2, dtype="uint8", crs=crs, transform=tf,
                nodata=255, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.stack([lab, yr]))
        dst.set_band_description(1, "CONV_LABEL")
        dst.set_band_description(2, "CONV_YEAR_MAX")
        dst.update_tags(CONV_LABEL="1 전용확정, 2 전용후보, 3 전용이전 (보전산지 해제 고시 필지)", CONV_YEAR_MAX="고시연도-2000")


def s2_scores():
    with rasterio.open(S2) as ds:
        desc = ds.descriptions
        idx = {d: i + 1 for i, d in enumerate(desc)}
        years = sorted({int(d.split("_")[1]) for d in desc})
        rd = lambda b: ds.read(idx[b], masked=True).filled(np.nan).astype("float32")
        ndvi = np.stack([rd(f"NDVI_{y}") for y in years])
        ndmi0, ndmi1 = rd(f"NDMI_{years[0]}"), rd(f"NDMI_{years[-1]}")
        ndbi0, ndbi1 = rd(f"NDBI_{years[0]}"), rd(f"NDBI_{years[-1]}")
        crs, tf, w, h = ds.crs, ds.transform, ds.width, ds.height
    steps = np.diff(ndvi, axis=0)
    with np.errstate(invalid="ignore"):
        drop_i = np.nanargmin(np.where(np.isnan(steps), np.inf, steps), axis=0)
    scores = {"−ΔNDVI(2025−2019)": ndvi[0] - ndvi[-1], "최대 연간 NDVI 감소": -np.nanmin(steps, axis=0),
              "−ΔNDMI": ndmi0 - ndmi1, "ΔNDBI": ndbi1 - ndbi0}
    drop_year = np.asarray(years[1:])[drop_i].astype("float32")
    return scores, drop_year, ndvi, crs, tf, w, h, years


def auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = np.isfinite(s)
    return round(float(roc_auc_score(y[ok], s[ok])), 4), round(float(average_precision_score(y[ok], s[ok])), 4)


def main():
    g = label_parcels()
    print("[라벨] 필지:", g["라벨"].value_counts().to_dict(), "| 후보 사유:",
          g.loc[g["라벨"].eq("전용후보"), "후보사유"].value_counts().to_dict())

    scores, drop_year, ndvi, crs, tf, w, h, years = s2_scores()
    lab_s2, yr_s2 = burn(g, crs, tf, w, h)
    write(D / "Chuncheon_convlabels_S2_20m.tif", lab_s2, yr_s2, crs, tf)
    ae_crs = rasterio.crs.CRS.from_string(AE_GRID["crs"])
    lab_ae, yr_ae = burn(g, ae_crs, AE_GRID["transform"], AE_GRID["width"], AE_GRID["height"])
    write(D / "Chuncheon_convlabels_AE_10m.tif", lab_ae, yr_ae, ae_crs, AE_GRID["transform"])
    px = {k: {"S2_20m": int((lab_s2 == v).sum()), "AE_10m": int((lab_ae == v).sum())} for k, v in CODES.items()}
    print("[래스터] 화소 수:", px)

    # 필지별 점수 (화소 평균) + 화소를 하나도 못 가진 작은 필지
    gp = g.drop_duplicates("PNU").to_crs(crs).reset_index(drop=True)
    pid = rasterize(list(zip(gp.geometry, gp.index + 1)), out_shape=(h, w), transform=tf, fill=0, dtype="int32")
    for name, s in scores.items():
        v = np.where(np.isfinite(s) & (pid > 0), s, 0)
        n = np.bincount(pid.ravel(), weights=(np.isfinite(s) & (pid > 0)).ravel(), minlength=len(gp) + 1)
        gp[name] = (np.bincount(pid.ravel(), weights=v.ravel(), minlength=len(gp) + 1) / np.maximum(n, 1))[1:]
    gp["S2화소수"] = np.bincount(pid.ravel(), minlength=len(gp) + 1)[1:]
    yv = np.where(pid > 0, drop_year, 0).astype(int)
    gp["NDVI최대감소연도"] = [int(np.bincount(yv[pid == i + 1]).argmax()) if c else None
                         for i, c in enumerate(gp["S2화소수"])]

    # 화소 단위 테스트: 전용확정 vs 검증된 무변화 산림
    with rasterio.open(LAB_S2) as lb:
        assert (lb.width, lb.height, lb.transform) == (w, h, tf), "S2 라벨 격자 불일치"
        train_p2 = lb.read(list(lb.descriptions).index("TRAIN_P2") + 1)
    pos, neg = lab_s2 == 1, (train_p2 == 0) & (lab_s2 == 255)
    rng = np.random.default_rng(42)
    neg_idx = np.flatnonzero(neg)
    neg_idx = rng.choice(neg_idx, min(200000, len(neg_idx)), replace=False)
    pos_idx = np.flatnonzero(pos)
    y = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))].astype(int)
    rows = []
    for name, s in scores.items():
        sv = np.r_[s.ravel()[pos_idx], s.ravel()[neg_idx]]
        auc, ap = auc_ap(y, sv)
        negv = s.ravel()[neg_idx]
        t95, t99 = np.nanpercentile(negv, 95), np.nanpercentile(negv, 99)
        par = gp[gp["라벨"].eq("전용확정") & gp["S2화소수"].gt(0)][name]
        rows.append({"점수": name, "AUROC": auc, "AP": ap, "양성화소": int(len(pos_idx)), "음성화소": int(len(neg_idx)),
                     "필지탐지율_FPR5%": round(float((par >= t95).mean()), 3),
                     "필지탐지율_FPR1%": round(float((par >= t99).mean()), 3),
                     **{f"평균_{k}": round(float(gp.loc[gp["라벨"].eq(k) & gp["S2화소수"].gt(0), name].mean()), 4) for k in CODES},
                     "평균_무변화산림": round(float(np.nanmean(negv)), 4)})
    test = pd.DataFrame(rows)

    conf = gp[gp["라벨"].eq("전용확정") & gp["S2화소수"].gt(0)]
    yr_hit = (conf["NDVI최대감소연도"] <= conf["발생연도_상한"]).mean()
    yr_within = ((conf["발생연도_상한"] - conf["NDVI최대감소연도"]).between(0, 2)).mean()
    overlap = {k: round(float((train_p2[lab_s2 == v] == 2).mean()), 3) if (lab_s2 == v).any() else None
               for k, v in CODES.items()}
    small = int((gp["S2화소수"] == 0).sum())

    out = gp.drop(columns=[]).to_crs(5186)
    if (D / "산지전용_라벨.gpkg").exists():
        (D / "산지전용_라벨.gpkg").unlink()
    out.to_file(D / "산지전용_라벨.gpkg", layer="산지전용_라벨", driver="GPKG")
    test.to_csv(D / "산지전용_라벨_테스트.csv", index=False, encoding="utf-8-sig")
    report = {"필지_라벨": g["라벨"].value_counts().to_dict(), "고유필지": int(len(gp)), "화소수": px,
              "S2화소_없는_소필지(<20m)": small, "점수별_테스트": rows,
              "전용확정_NDVI최대감소연도": {"고시연도_이전_비율": round(float(yr_hit), 3), "고시 0~2년전_비율": round(float(yr_within), 3)},
              "임상도_산지전용라벨(TRAIN_P2=2)_겹침비율": overlap}
    (D / "산지전용_라벨_테스트.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 230)
    print(f"[필지] 고유 {len(gp)}개 (S2 화소 없는 20 m 미만 소필지 {small}개)")
    print(test.to_string(index=False))
    print("[전용확정 NDVI 최대감소 연도] 고시연도 이전:", round(yr_hit, 3), "| 고시 0~2년 전:", round(yr_within, 3))
    print("[임상도 산지전용 라벨과 겹침]", overlap)
    print(f"저장: {D}")


if __name__ == "__main__":
    main()
