#!/usr/bin/env python3
# [colab 정리본] 원본: scripts/gee_export_sentinel_chuncheon.py — 파일명·모듈명만 실험 단계 번호에 맞춤 (계산 로직 동일)
"""
춘천시 Sentinel-2 연도별 지수 스택 내보내기 (gee_export_18bands.js 의 Python 판)
==============================================================================
gee_export_18bands.js 와 같은 방식으로 연도마다 6밴드를 만들어 쌓는다.
  - 생육기 4/1~10/31 Sentinel-2 SR(Harmonized), 구름 비율 70% 미만 장면, SCL 구름·그림자 마스크, 중앙값 합성
  - 밴드: NDVI, NDWI, NDBI, NDMI, Slope, Elevation (Copernicus GLO-30 DEM) — 연도 접미사 _YYYY
바뀐 점
  - 관심지역: 서울 사각형 → 춘천시 경계(step01_chuncheon_boundary.geojson, 없으면 GAUL 2015 level2 폴백)
  - 기간: 2023~2025 → 2019~2025 (7년 × 6밴드 = 42밴드)
  - 결과: Google Drive 로 GeoTIFF(COG) 내보내기. Colab 에서 Drive 를 마운트했다면 완료까지 기다렸다가 경로 출력

Colab 실행:
  !python step01_gee_export_sentinel.py --project ee-choish7838            # 42밴드 한 파일
  !python step01_gee_export_sentinel.py --project ee-choish7838 --per-year # 연도별 6밴드 7개 파일
  !python step01_gee_export_sentinel.py --project ee-choish7838 --dry-run  # 장면 수·밴드만 확인
  !python step01_gee_export_sentinel.py --project ee-choish7838 --start-year 2017   # AlphaEarth 와 같은 2017~2025 (54밴드)
"""

import argparse
import json
import os
import time

import ee

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARY_GEOJSON = os.path.join(HERE, "step01_chuncheon_boundary.geojson")  # 춘천시 경계 (EPSG:4326)

CFG = {
    "years": list(range(2019, 2026)),
    "season_start_month": 4,
    "season_end_exclusive_month": 11,  # 4/1 ~ 10/31
    "max_cloud_pct": 70,
    "export_crs": "EPSG:32652",
    "export_scale": 20,
    "drive_folder": "GEE_Export",
    "file_prefix": "Chuncheon_S2",
}

META = {}


# ---------------------------------------------------------------- 인증 / 관심지역

def init_ee(project):
    """Earth Engine 인증 및 초기화 (첫 실행 시 Colab 인증 창이 뜬다)."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def load_roi():
    """춘천시 경계: 로컬 GeoJSON 우선, 없으면 GAUL level2 폴백 (gee_forest_change.py 와 동일)."""
    if os.path.exists(BOUNDARY_GEOJSON):
        with open(BOUNDARY_GEOJSON, encoding="utf-8") as f:
            gj = json.load(f)
        feats = [ee.Feature(ee.Geometry(ft["geometry"]), ft.get("properties") or {})
                 for ft in gj["features"] if ft.get("geometry")]
        roi = ee.FeatureCollection(feats)
        src = "local_geojson(step01_chuncheon_boundary.geojson, LSMD 읍면동 경계 병합)"
    else:
        roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", "Republic of Korea"),
                ee.Filter.eq("ADM2_NAME", "Chuncheon-si"),
            )
        )
        src = "GAUL2015 level2 (step01_chuncheon_boundary.geojson 없음)"
    area_km2 = roi.geometry().area(maxError=10).divide(1e6).getInfo()
    print(f"[ROI] 춘천시 경계: {src}, 면적 {area_km2:,.1f} km²")
    META.update({"roi_source": src, "roi_area_km2": area_km2})
    return roi.geometry()


# ---------------------------------------------------------------- 합성 영상

def mask_clouds(img):
    """SCL 기반 마스크: 0 No data, 1 Saturated, 3 Cloud shadow, 8/9 Cloud, 10 Cirrus, 11 Snow 제외."""
    scl = img.select("SCL")
    mask = (scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8))
            .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    return img.updateMask(mask).select(["B3", "B4", "B8", "B11"]).multiply(0.0001)


def terrain(aoi):
    demcol = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(aoi)
    demproj = ee.Image(demcol.first()).select("DEM").projection()
    elev = demcol.select("DEM").mosaic().setDefaultProjection(demproj).rename("Elevation")
    slope = ee.Terrain.slope(elev).rename("Slope")
    return elev, slope


def make_year(year, aoi, elev, slope):
    start = ee.Date.fromYMD(year, CFG["season_start_month"], 1)
    stop = ee.Date.fromYMD(year, CFG["season_end_exclusive_month"], 1)
    scenes = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi).filterDate(start, stop)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CFG["max_cloud_pct"]))
              .map(mask_clouds))
    composite = scenes.median()
    image = ee.Image.cat([
        composite.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{year}"),
        composite.normalizedDifference(["B3", "B8"]).rename(f"NDWI_{year}"),
        composite.normalizedDifference(["B11", "B8"]).rename(f"NDBI_{year}"),
        composite.normalizedDifference(["B8", "B11"]).rename(f"NDMI_{year}"),
        slope.rename(f"Slope_{year}"),
        elev.rename(f"Elevation_{year}"),
    ]).toFloat().clip(aoi)
    return image, scenes


def summarize(year, image, scenes, aoi):
    """연도별 장면 수, 유효 화소 비율, 화소별 구름 없는 관측 횟수(합성 품질) 확인.
    유효 비율은 NDVI 마스크를 0/1 로 펼쳐 같은 격자에서 평균 → 분자·분모 격자 불일치로 100% 를 넘던 문제 수정."""
    ndvi = image.select(f"NDVI_{year}")
    clear = scenes.select("B4").count().rename("clear").unmask(0).clip(aoi)
    stats = ee.Dictionary({
        "scenes": scenes.size(),
        "ndvi_mean": ndvi.reduceRegion(ee.Reducer.mean(), aoi, 100, maxPixels=1e9, bestEffort=True).get(f"NDVI_{year}"),
        "valid": ndvi.mask().rename("v").unmask(0).clip(aoi).reduceRegion(
            ee.Reducer.mean(), aoi, 100, maxPixels=1e9, bestEffort=True).get("v"),
        "clear": clear.reduceRegion(ee.Reducer.percentile([5, 50]), aoi, 100, maxPixels=1e9, bestEffort=True),
    }).getInfo()
    c5, c50 = stats["clear"]["clear_p5"], stats["clear"]["clear_p50"]
    print(f"[S2] {year}: 장면 {stats['scenes']}개, NDVI 평균 {stats['ndvi_mean']:.3f}, "
          f"유효 화소 {stats['valid']:.1%}, 화소별 맑은 관측 중앙값 {c50:.0f}회 (하위 5% {c5:.0f}회)")
    return {"year": year, "scenes": stats["scenes"], "ndvi_mean": stats["ndvi_mean"],
            "valid_ratio": stats["valid"], "clear_obs_p50": c50, "clear_obs_p5": c5}


# ---------------------------------------------------------------- 내보내기

def export(image, aoi, name):
    task = ee.batch.Export.image.toDrive(
        image=image.unmask(-9999).toFloat(),
        description=name,
        folder=CFG["drive_folder"],
        fileNamePrefix=name,
        region=aoi,
        scale=CFG["export_scale"],
        crs=CFG["export_crs"],
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True, "noData": -9999},
    )
    task.start()
    print(f"[EXPORT] 시작: {name} (task {task.id}) → Drive/{CFG['drive_folder']}/{name}*.tif")
    return task


def wait(tasks, poll_sec=30):
    pending = {t.id: t for t in tasks}
    while pending:
        for tid, t in list(pending.items()):
            st = t.status()
            if st["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                msg = st.get("error_message", "")
                print(f"[EXPORT] {st['description']}: {st['state']} {msg}")
                pending.pop(tid)
        if pending:
            print(f"[EXPORT] 진행 중 {len(pending)}개 … {poll_sec}초 후 다시 확인")
            time.sleep(poll_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ee-choish7838", help="GEE 클라우드 프로젝트 ID")
    ap.add_argument("--start-year", type=int, default=CFG["years"][0])
    ap.add_argument("--end-year", type=int, default=CFG["years"][-1])
    ap.add_argument("--scale", type=int, default=CFG["export_scale"], help="내보내기 해상도 (m)")
    ap.add_argument("--per-year", action="store_true", help="연도별 6밴드 파일로 나눠 내보내기")
    ap.add_argument("--dry-run", action="store_true", help="장면 수·밴드만 확인하고 내보내지 않음")
    ap.add_argument("--no-wait", action="store_true", help="내보내기 작업 완료를 기다리지 않음")
    args, _ = ap.parse_known_args()  # Colab 커널 인자 무시

    CFG.update({"years": list(range(args.start_year, args.end_year + 1)), "export_scale": args.scale})
    init_ee(args.project)
    aoi = load_roi()
    elev, slope = terrain(aoi)

    images, summary = [], []
    for year in CFG["years"]:
        image, scenes = make_year(year, aoi, elev, slope)
        images.append(image)
        summary.append(summarize(year, image, scenes, aoi))
    stack = ee.Image.cat(images)
    bands = stack.bandNames().getInfo()
    print(f"[STACK] 밴드 {len(bands)}개: {bands}")
    META.update({"params": CFG, "per_year": summary, "bands": bands})

    span = f"{CFG['years'][0]}_{CFG['years'][-1]}"
    with open(os.path.join(HERE, f"{CFG['file_prefix']}_{span}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2, default=str)

    if args.dry_run:
        print("[*] dry-run — 내보내기는 하지 않았습니다.")
        return

    if args.per_year:
        tasks = [export(img, aoi, f"{CFG['file_prefix']}_{y}_6Bands") for y, img in zip(CFG["years"], images)]
    else:
        tasks = [export(stack, aoi, f"{CFG['file_prefix']}_{span}_{len(bands)}Bands")]
    if not args.no_wait:
        wait(tasks)
    print("[*] 완료 — Google Drive 의 GEE_Export 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
