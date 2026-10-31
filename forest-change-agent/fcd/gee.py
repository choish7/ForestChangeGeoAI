"""Google Earth Engine 연계: AlphaEarth 연간 임베딩·Sentinel-2·DEM 특징 추출과 래스터 export.

인증(최초 1회):  python -c "import ee; ee.Authenticate()"
"""
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd

from . import config as C

NODATA = -999.0
_READY = False


def init():
    global _READY
    if not _READY:
        ee.Initialize(project=C.GEE_PROJECT)
        _READY = True


# ------------------------------------------------------------------ 이미지 구성

def ae_year(year: int) -> ee.Image:
    """연도별 AlphaEarth 임베딩(64밴드, 단위벡터). 타일 모자이크."""
    col = ee.ImageCollection(C.AE_COLLECTION).filter(ee.Filter.calendarRange(year, year, "year"))
    return col.mosaic().select(C.AE_BANDS)


def ae_stack(years=C.AE_YEARS) -> ee.Image:
    """ae{year}_A00 … 형태로 전 연도를 하나의 이미지로 결합."""
    imgs = [ae_year(y).rename([f"ae{y}_{b}" for b in C.AE_BANDS]) for y in years]
    return ee.Image.cat(imgs)


def cos_distance(e0: ee.Image, e1: ee.Image) -> ee.Image:
    """1 − cos(e0, e1). 연간 임베딩은 단위벡터이므로 내적 = 코사인 유사도."""
    return ee.Image(1).subtract(e0.multiply(e1).reduce(ee.Reducer.sum())).rename("cos_dist")


def s2_summer(year: int) -> ee.Image:
    """Sentinel-2 L2A 성장기(6–9월) 중앙값 NDVI·NBR (농림위성 가용 전 대체 분광특징)."""
    def mask(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok)
    med = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(f"{year}-06-01", f"{year}-09-30")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
           .map(mask).median())
    ndvi = med.normalizedDifference(["B8", "B4"]).rename(f"ndvi{year}")
    nbr = med.normalizedDifference(["B8", "B12"]).rename(f"nbr{year}")
    return ndvi.addBands(nbr)


def terrain() -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    return dem.rename("elev").addBands(ee.Terrain.slope(dem).rename("slope"))


def feature_image(years=C.AE_YEARS, s2_years=(C.YEAR_FROM, C.YEAR_TO)) -> ee.Image:
    img = ae_stack(years)
    for y in s2_years:
        img = img.addBands(s2_summer(y))
    return img.addBands(terrain()).unmask(NODATA)


# ------------------------------------------------------------------ 점 추출

def _extract_chunk(img: ee.Image, chunk: pd.DataFrame, id_col: str) -> list[dict]:
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {id_col: getattr(r, id_col)})
             for r in chunk.itertuples()]
    fc = img.sampleRegions(collection=ee.FeatureCollection(feats), properties=[id_col],
                           scale=10, tileScale=4, geometries=False)
    return [f["properties"] for f in fc.getInfo()["features"]]


def extract_points(points: pd.DataFrame, id_col: str = "sample_id", chunk: int = 200,
                   workers: int = 4, img: ee.Image | None = None, verbose: bool = True) -> pd.DataFrame:
    """lon/lat 점에서 특징 이미지 값을 추출해 DataFrame 반환 (NODATA → NaN)."""
    init()
    img = img or feature_image()
    chunks = [points.iloc[i:i + chunk] for i in range(0, len(points), chunk)]
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(lambda c: _extract_chunk(img, c, id_col), chunks), 1):
            rows.extend(res)
            if verbose:
                print(f"  GEE 추출 {i}/{len(chunks)} 청크 ({len(rows):,}점)", flush=True)
    df = pd.DataFrame(rows)
    num = df.columns.drop(id_col)
    df[num] = df[num].astype("float32").replace(NODATA, np.nan)
    return df


# ------------------------------------------------------------------ 래스터 export (Drive)

def export_rasters(region: ee.Geometry, folder: str = "aef_chuncheon_forest", crs: str = "EPSG:5186"):
    """시범지역 전체 래스터를 Google Drive 로 export (비동기 task 반환).
    - ae_{year}_int8 : 64밴드 임베딩을 ×127 정수화 (용량 1/4, 코사인 오차 <1e-3)
    - ae_consec_dist : 연속 연도쌍 코사인 거리 8밴드 (2017→18 … 2024→25)
    """
    init()
    tasks = []
    for y in C.AE_YEARS:
        img = ae_year(y).multiply(127).round().toInt8()
        tasks.append(ee.batch.Export.image.toDrive(
            image=img, description=f"ae_{y}_int8", folder=folder, fileNamePrefix=f"ae_{y}_int8",
            region=region, scale=10, crs=crs, maxPixels=1e10))
    dist = ee.Image.cat([cos_distance(ae_year(y - 1), ae_year(y)).rename(f"d{y - 1}_{y}")
                         for y in C.AE_YEARS[1:]]).toFloat()
    tasks.append(ee.batch.Export.image.toDrive(
        image=dist, description="ae_consec_dist", folder=folder, fileNamePrefix="ae_consec_dist",
        region=region, scale=10, crs=crs, maxPixels=1e10))
    for t in tasks:
        t.start()
    return tasks


# ------------------------------------------------------------------ 에이전트용 즉석 분석

def to_ee_geometry(gdf_4326) -> ee.Geometry:
    init()
    geom = gdf_4326.union_all().simplify(0.0002)
    return ee.Geometry(geom.__geo_interface__)


def change_scan(region: ee.Geometry, y0: int, y1: int, threshold: float, max_patches: int = 15) -> dict:
    """영역 내 산림(ESA WorldCover 2021 수목) 픽셀의 AlphaEarth 코사인 거리 요약 + 변화 패치 추출."""
    init()
    forest = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(10)
    dist = cos_distance(ae_year(y0), ae_year(y1)).updateMask(forest)
    area = ee.Image.pixelArea().divide(1e4)
    changed = dist.gt(threshold).selfMask()

    stats = ee.Dictionary({
        "forest_ha": area.updateMask(forest).reduceRegion(ee.Reducer.sum(), region, 10, maxPixels=1e10, tileScale=4).get("area"),
        "changed_ha": area.updateMask(changed).reduceRegion(ee.Reducer.sum(), region, 10, maxPixels=1e10, tileScale=4).get("area"),
        "dist_pct": dist.reduceRegion(ee.Reducer.percentile([50, 90, 99]), region, 30, maxPixels=1e9, tileScale=4),
    }).getInfo()

    vec = changed.addBands(area).reduceToVectors(
        geometry=region, scale=10, geometryType="centroid", eightConnected=True,
        labelProperty="flag", reducer=ee.Reducer.sum(), maxPixels=1e10, tileScale=4)
    top = vec.filter(ee.Filter.gte("sum", 0.2)).sort("sum", False).limit(max_patches).getInfo()
    patches = [{"lon": round(f["geometry"]["coordinates"][0], 5), "lat": round(f["geometry"]["coordinates"][1], 5),
                "area_ha": round(f["properties"]["sum"], 2)} for f in top["features"]]
    return {**stats, "patches": patches}
