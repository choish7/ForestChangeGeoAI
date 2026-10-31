"""연도별 산림사업 공간정보를 QGIS 프로젝트(시간 슬라이더 설정)와 개요 지도 PNG로 만든다.

QGIS 내장 파이썬으로 실행:
  "C:/Program Files/QGIS 3.34.9/bin/python-qgis-ltr.bat" scripts/make_yearly_map.py
QGIS에서 프로젝트를 열고 [보기 > 패널 > 시간 컨트롤러]에서 재생하면 1년 단위로 그 해 사업지만 표시된다.
"""
from datetime import datetime
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsCategorizedSymbolRenderer, QgsDateTimeRange, QgsFillSymbol, QgsGraduatedSymbolRenderer,
    QgsInterval, QgsLayoutExporter, QgsProject, QgsRendererCategory, QgsRendererRange, QgsSingleSymbolRenderer,
    QgsUnitTypes, QgsVectorLayer, QgsVectorLayerTemporalProperties,
)
from qgis.PyQt.QtCore import QDateTime

from make_chuncheon_thinning_map import label
from make_combined_map import make_layout

ROOT = Path(__file__).resolve().parent.parent
GPKG = ROOT / "output" / "춘천시_연도별_산림사업.gpkg"
PROJECT = ROOT / "output" / "춘천시_연도별_산림사업.qgz"
PNG = ROOT / "output" / "춘천시_사업지_최근사업연도_지도.png"
FIRST_YEAR, LAST_YEAR = 2000, 2025

CATEGORY_COLORS = {
    "조림": "#1b9e77",
    "숲가꾸기": "#66a61e",
    "임목생산": "#d95f02",
    "시설": "#7570b3",
    "소득": "#e6ab02",
    "병해충방제및산림피해복구": "#e7298a",
}
RECENT_RANGES = [  # (하한, 상한, 색, 라벨)
    (2000, 2009, "#fdd49e", "2000~2009 (7차)"),
    (2010, 2018, "#fc8d59", "2010~2018 (8차)"),
    (2019, 2025, "#b30000", "2019~2025 (9차)"),
]


def layer(name, title):
    lyr = QgsVectorLayer(f"{GPKG}|layername={name}", title, "ogr")
    assert lyr.isValid(), name
    return lyr


def fill(**props):
    return QgsFillSymbol.createSimple({k: str(v) for k, v in props.items()})


def yearly_temporal(lyr):
    tp = lyr.temporalProperties()
    tp.setIsActive(True)
    tp.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeStartAndEndFromFields)
    tp.setStartField("사업시작일")
    tp.setEndField("사업종료일")


def main():
    qgs = QgsApplication([], False)
    qgs.initQgis()
    project = QgsProject.instance()

    city = layer("춘천시경계", "춘천시 경계")
    city.setRenderer(QgsSingleSymbolRenderer(fill(color="0,0,0,0", style="no", outline_color="200,30,30", outline_width=0.9)))
    project.setCrs(city.crs())
    umd = layer("춘천시_읍면동경계", "읍면동 경계")
    umd.setRenderer(QgsSingleSymbolRenderer(fill(color="0,0,0,0", style="no", outline_color="90,90,90",
                                                 outline_width=0.25, outline_style="dash")))
    label(umd, "EMD_NM", 7, "#444444")

    records = layer("연도별_산림사업", "연도별 산림사업(사업구분)")
    records.setRenderer(QgsCategorizedSymbolRenderer("사업구분", [
        QgsRendererCategory(k, fill(color=v + "cc", outline_color=v, outline_width=0.2), k)
        for k, v in CATEGORY_COLORS.items()]))
    yearly_temporal(records)

    site_year = layer("연도별_사업지요약", "연도별 사업지 요약")
    site_year.setRenderer(QgsSingleSymbolRenderer(fill(color="214,0,126,160", outline_color="120,0,70", outline_width=0.2)))
    yearly_temporal(site_year)

    history = layer("사업지_연도이력", "사업지 최근 사업연도")
    ranges = [QgsRendererRange(lo, hi + 0.99, fill(color=c, outline_color="80,80,80", outline_width=0.1), lab)
              for lo, hi, c, lab in RECENT_RANGES]
    history.setRenderer(QgsGraduatedSymbolRenderer("최근사업연도", ranges))

    idle = layer("사업지_연도이력", "사업 없는 국유림 소반")
    idle.setSubsetString('"사업건수" = 0')
    idle.setRenderer(QgsSingleSymbolRenderer(fill(color="225,225,225", outline_color="160,160,160", outline_width=0.1)))

    for lyr in (city, umd, records, site_year, history):
        lyr.saveStyleToDatabase(lyr.name(), "ForestChangeGeoAI 기본 스타일", True, "")

    order = [city, umd, records, site_year, history, idle]
    project.addMapLayers(order, False)
    root = project.layerTreeRoot()
    for lyr in order:
        root.addLayer(lyr)
    for lyr in (site_year, history, idle):
        root.findLayer(lyr.id()).setItemVisibilityChecked(False)

    ts = project.timeSettings()
    ts.setTemporalRange(QgsDateTimeRange(QDateTime(datetime(FIRST_YEAR, 1, 1)), QDateTime(datetime(LAST_YEAR, 12, 31, 23, 59))))
    ts.setTimeStep(1)
    ts.setTimeStepUnit(QgsUnitTypes.TemporalYears)
    ts.setFramesPerSecond(1)

    extent = city.extent()
    extent.grow(max(extent.width(), extent.height()) * 0.03)
    layout = make_layout(project, 0, [city, umd, history, idle], extent)
    layout.setName("사업지 최근 사업연도")
    title = [i for i in layout.items() if i.__class__.__name__ == "QgsLayoutItemLabel"][0]
    title.setText("춘천시 산림사업지 최근 사업연도 (국유림 7·8·9차 실적 + 2015 숲가꾸기)")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 200
    assert QgsLayoutExporter(layout).exportToImage(str(PNG), settings) == QgsLayoutExporter.Success
    print("저장:", PNG)

    assert project.write(str(PROJECT))
    print("저장:", PROJECT)
    qgs.exitQgis()


if __name__ == "__main__":
    main()
