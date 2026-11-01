#!/usr/bin/env python3
# [colab 정리본] 원본: scripts/gee_export_alphaearth_chuncheon.py — 파일명·모듈명만 실험 단계 번호에 맞춤 (계산 로직 동일)
"""
춘천시 AlphaEarth Foundations 연간 임베딩(64차원) 내보내기 (2017~2025)
======================================================================
step01_gee_export_sentinel.py 와 같은 틀로, forest-change-agent/fcd/gee.py 의 export_rasters 규칙을 따른다.
  - 자료: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (연간 64밴드 A00~A63, 단위벡터, 10 m)
  - 연도별 파일: Chuncheon_AE_{year}_64Bands_int8 — 임베딩 × 127 반올림 int8 (용량 1/4, 코사인 오차 < 1e-3)
                  복원: float = int8 / 127, 결측 = -128
  - 변화 파일:   Chuncheon_AE_consec_dist_2017_2025 — 연속 연도쌍 코사인 거리 1 − cos(e[y-1], e[y]) 8밴드
  - 관심지역: 춘천시 경계(step01_chuncheon_boundary.geojson, 없으면 GAUL 2015 level2 폴백)
  - 좌표계: EPSG:32652 (AlphaEarth·Sentinel-2 원격자 UTM 52N — 재투영 없음, Sentinel 42밴드 스택과 동일)

Colab 실행:
  %run step01_gee_export_alphaearth.py --dry-run          # 가용 연도·결측률·예상 용량만 확인
  %run step01_gee_export_alphaearth.py                    # 연도별 9개 + 변화 1개 Drive 내보내기
  %run step01_gee_export_alphaearth.py --scale 20         # 20 m (용량 1/4)
  %run step01_gee_export_alphaearth.py --float            # float32 원값 (용량 4배)
"""

import argparse
import json
import os
import time

import ee

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARY_GEOJSON = os.path.join(HERE, "step01_chuncheon_boundary.geojson")

AE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AE_BANDS = [f"A{i:02d}" for i in range(64)]
INT8_NODATA = -128

CFG = {
    "years": list(range(2017, 2026)),
    "export_crs": "EPSG:32652",
    "export_scale": 10,
    "drive_folder": "GEE_Export",
    "file_prefix": "Chuncheon_AE",
}

META = {}


# ---------------------------------------------------------------- 인증 / 관심지역

def init_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def load_roi():
    if os.path.exists(BOUNDARY_GEOJSON):
        with open(BOUNDARY_GEOJSON, encoding="utf-8") as f:
            gj = json.load(f)
        roi = ee.FeatureCollection([ee.Feature(ee.Geometry(ft["geometry"]), ft.get("properties") or {})
                                    for ft in gj["features"] if ft.get("geometry")])
        src = "local_geojson(step01_chuncheon_boundary.geojson, LSMD 읍면동 경계 병합)"
    else:
        roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", "Republic of Korea"), ee.Filter.eq("ADM2_NAME", "Chuncheon-si")))
        src = "GAUL2015 level2 (step01_chuncheon_boundary.geojson 없음)"
    area_km2 = roi.geometry().area(maxError=10).divide(1e6).getInfo()
    print(f"[ROI] 춘천시 경계: {src}, 면적 {area_km2:,.1f} km²")
    META.update({"roi_source": src, "roi_area_km2": area_km2})
    return roi.geometry(), area_km2


# ---------------------------------------------------------------- 임베딩

def ae_year(year, aoi):
    """연도별 64밴드 임베딩 (춘천 경계에 걸친 타일 모자이크)."""
    col = (ee.ImageCollection(AE_COLLECTION).filterBounds(aoi)
           .filter(ee.Filter.calendarRange(year, year, "year")))
    return col.mosaic().select(AE_BANDS), col.size()


def cos_distance(e0, e1):
    """1 − cos(e0, e1). 연간 임베딩은 단위벡터이므로 내적 = 코사인 유사도."""
    return ee.Image(1).subtract(e0.multiply(e1).reduce(ee.Reducer.sum()))


def available_years(aoi):
    return (ee.ImageCollection(AE_COLLECTION).filterBounds(aoi)
            .aggregate_array("system:time_start").map(lambda t: ee.Date(t).get("year"))
            .distinct().sort().getInfo())


def summarize(year, emb, n_tiles, aoi):
    """타일 수, 유효 픽셀 비율, 벡터 노름(≈1 이어야 정상) 확인."""
    norm = emb.pow(2).reduce(ee.Reducer.sum()).sqrt().rename("norm")
    s = ee.Dictionary({
        "tiles": n_tiles,
        "norm": norm.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                                  aoi, 100, maxPixels=1e9, bestEffort=True),
        "total": ee.Image.constant(1).reduceRegion(ee.Reducer.count(), aoi, 100, maxPixels=1e9, bestEffort=True),
    }).getInfo()
    valid = min(s["norm"]["norm_count"] / max(s["total"]["constant"], 1), 1.0)
    print(f"[AE] {year}: 타일 {s['tiles']}개, 유효 픽셀 {valid:.1%}, 평균 노름 {s['norm']['norm_mean']:.4f}")
    return {"year": year, "tiles": s["tiles"], "valid_ratio": valid, "norm_mean": s["norm"]["norm_mean"]}


# ---------------------------------------------------------------- 내보내기

def export(image, aoi, name, nodata):
    task = ee.batch.Export.image.toDrive(
        image=image, description=name, folder=CFG["drive_folder"], fileNamePrefix=name,
        region=aoi, scale=CFG["export_scale"], crs=CFG["export_crs"], maxPixels=1e13,
        fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True, "noData": nodata},
    )
    task.start()
    print(f"[EXPORT] 시작: {name} (task {task.id})")
    return task


def wait(tasks, poll_sec=60):
    pending = {t.id: t for t in tasks}
    while pending:
        for tid, t in list(pending.items()):
            st = t.status()
            if st["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                print(f"[EXPORT] {st['description']}: {st['state']} {st.get('error_message', '')}")
                pending.pop(tid)
        if pending:
            print(f"[EXPORT] 진행 중 {len(pending)}개 … {poll_sec}초 후 다시 확인", flush=True)
            time.sleep(poll_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ee-choish7838", help="GEE 클라우드 프로젝트 ID")
    ap.add_argument("--start-year", type=int, default=CFG["years"][0])
    ap.add_argument("--end-year", type=int, default=CFG["years"][-1])
    ap.add_argument("--scale", type=int, default=CFG["export_scale"], help="내보내기 해상도 (m, 원해상도 10)")
    ap.add_argument("--crs", default=CFG["export_crs"])
    ap.add_argument("--float", action="store_true", help="int8 양자화 대신 float32 원값으로 내보내기")
    ap.add_argument("--no-dist", action="store_true", help="연속 연도 코사인 거리 파일 생략")
    ap.add_argument("--dry-run", action="store_true", help="가용 연도·결측·예상 용량만 확인")
    ap.add_argument("--no-wait", action="store_true", help="내보내기 완료를 기다리지 않음")
    args, _ = ap.parse_known_args()  # Colab 커널 인자 무시

    CFG.update({"years": list(range(args.start_year, args.end_year + 1)),
                "export_scale": args.scale, "export_crs": args.crs, "dtype": "float32" if args.float else "int8"})
    init_ee(args.project)
    aoi, area_km2 = load_roi()

    avail = available_years(aoi)
    missing = [y for y in CFG["years"] if y not in avail]
    print(f"[AE] 가용 연도 {avail}" + (f" — 요청했지만 없는 연도 {missing} 제외" if missing else ""))
    years = [y for y in CFG["years"] if y in avail]

    embs, summary = {}, []
    for y in years:
        emb, n = ae_year(y, aoi)
        embs[y] = emb
        summary.append(summarize(y, emb, n, aoi))

    px = area_km2 * 1e6 / CFG["export_scale"] ** 2
    per_year_gb = px * 64 * (4 if args.float else 1) / 1e9
    dist_gb = px * (len(years) - 1) * 4 / 1e9
    total_gb = per_year_gb * len(years) + (0 if args.no_dist else dist_gb)
    print(f"[SIZE] 연도별 약 {per_year_gb:.2f} GB × {len(years)} + 변화 {dist_gb:.2f} GB "
          f"= 약 {total_gb:.1f} GB (압축 전, {CFG['export_scale']} m, {CFG['dtype']})")
    META.update({"params": CFG, "years": years, "per_year": summary, "estimated_gb": round(total_gb, 2)})

    span = f"{years[0]}_{years[-1]}"
    with open(os.path.join(HERE, f"{CFG['file_prefix']}_{span}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2, default=str)

    if args.dry_run:
        print("[*] dry-run — 내보내기는 하지 않았습니다.")
        return

    tasks = []
    for y in years:
        if args.float:
            img, nodata, name = embs[y].toFloat().unmask(-9999), -9999, f"{CFG['file_prefix']}_{y}_64Bands_float"
        else:
            img = embs[y].multiply(127).round().toInt8().unmask(INT8_NODATA)
            nodata, name = INT8_NODATA, f"{CFG['file_prefix']}_{y}_64Bands_int8"
        tasks.append(export(img.clip(aoi), aoi, name, nodata))
    if not args.no_dist and len(years) > 1:
        dist = ee.Image.cat([cos_distance(embs[a], embs[b]).rename(f"d{a}_{b}")
                             for a, b in zip(years, years[1:])]).toFloat().unmask(-9999).clip(aoi)
        tasks.append(export(dist, aoi, f"{CFG['file_prefix']}_consec_dist_{span}", -9999))
    print(f"[EXPORT] {len(tasks)}개 작업 → Drive/{CFG['drive_folder']}/ "
          "(진행상황: https://code.earthengine.google.com/tasks)")
    if not args.no_wait:
        wait(tasks)
    print("[*] 완료 — Google Drive 의 GEE_Export 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
