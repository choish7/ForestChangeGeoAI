"""숲가꾸기·임상도(2019/2025)·국유림 임소반 결합 결과를 QGIS 프로젝트(.qgz)와 연도별 지도 PNG로 만든다.

QGIS 내장 파이썬으로 실행:
  "C:/Program Files/QGIS 3.34.9/bin/python-qgis-ltr.bat" scripts/make_combined_map.py
"""
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsCategorizedSymbolRenderer, QgsFillSymbol, QgsLayoutExporter, QgsLayoutItemLabel,
    QgsLayoutItemLegend, QgsLayoutItemMap, QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize, QgsPrintLayout,
    QgsProject, QgsRendererCategory, QgsSingleSymbolRenderer, QgsTextFormat, QgsUnitTypes, QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor, QFont

from make_chuncheon_thinning_map import label

ROOT = Path(__file__).resolve().parent.parent
GPKG = ROOT / "output" / "춘천시_숲가꾸기_임상도_사업실적_결합.gpkg"
PROJECT = ROOT / "output" / "춘천시_숲가꾸기_임상도_사업실적_결합.qgz"
YEARS = (2019, 2025)

FOREST_TYPE_COLORS = {
    "침엽수림": "#2e7d32",
    "활엽수림": "#a5d6a7",
    "혼효림": "#81b29a",
    "죽림": "#c0ca33",
    "무립목지/비산림": "#eeeeee",
}


def layer(name, title):
    lyr = QgsVectorLayer(f"{GPKG}|layername={name}", title, "ogr")
    assert lyr.isValid(), name
    return lyr


def simple(lyr, **props):
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({k: str(v) for k, v in props.items()})))
    return lyr


def text(layout, s, x, y, w, h, size, bold=False):
    item = QgsLayoutItemLabel(layout)
    item.setText(s)
    fmt = QgsTextFormat()
    fmt.setFont(QFont("Malgun Gothic", size, QFont.Bold if bold else QFont.Normal))
    fmt.setSize(size)
    item.setTextFormat(fmt)
    item.attemptMove(QgsLayoutPoint(x, y))
    item.attemptResize(QgsLayoutSize(w, h))
    layout.addLayoutItem(item)


def make_layout(project, year, layers, extent):
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(f"{year} 임상도 + 숲가꾸기")
    layout.pageCollection().page(0).setPageSize(QgsLayoutSize(297, 297, QgsUnitTypes.LayoutMillimeters))
    text(layout, f"춘천시 숲가꾸기 사업지(2015) × {year} 임상도 × 국유림 임소반", 10, 6, 277, 12, 16, bold=True)

    m = QgsLayoutItemMap(layout)
    m.setLayers(layers)
    m.attemptMove(QgsLayoutPoint(10, 20))
    m.attemptResize(QgsLayoutSize(277, 267))
    m.zoomToExtent(extent)
    m.setFrameEnabled(True)
    layout.addLayoutItem(m)

    legend = QgsLayoutItemLegend(layout)
    legend.setLinkedMap(m)
    legend.setTitle("범례")
    legend.setAutoUpdateModel(False)  # 프로젝트 전체가 아니라 이 지도에 그린 레이어만 범례에 표시
    legend_root = legend.model().rootGroup()
    legend_root.clear()
    for lyr in layers:
        legend_root.addLayer(lyr)
    legend.attemptMove(QgsLayoutPoint(229, 178))
    legend.setBackgroundColor(QColor(255, 255, 255, 230))
    legend.setBackgroundEnabled(True)
    legend.setFrameEnabled(True)
    layout.addLayoutItem(legend)

    bar = QgsLayoutItemScaleBar(layout)
    bar.setStyle("Single Box")
    bar.setLinkedMap(m)
    bar.setUnits(QgsUnitTypes.DistanceKilometers)
    bar.setUnitLabel("km")
    bar.setNumberOfSegments(4)
    bar.setUnitsPerSegment(2.5)
    bar.attemptMove(QgsLayoutPoint(214, 272))
    layout.addLayoutItem(bar)
    project.layoutManager().addLayout(layout)
    return layout


def main():
    qgs = QgsApplication([], False)
    qgs.initQgis()
    project = QgsProject.instance()

    city = simple(layer("춘천시경계", "춘천시 경계"), color="0,0,0,0", style="no",
                  outline_color="200,30,30", outline_width=0.9)
    project.setCrs(city.crs())
    umd = simple(layer("춘천시_읍면동경계", "읍면동 경계"), color="0,0,0,0", style="no",
                 outline_color="90,90,90", outline_width=0.25, outline_style="dash")
    label(umd, "EMD_NM", 7, "#444444")
    stands = simple(layer("국유림_임소반", "국유림 임소반"), color="0,0,0,0", style="no",
                    outline_color="30,90,200", outline_width=0.2)
    thinning = simple(layer("숲가꾸기_결합", "숲가꾸기 사업지(2015)"), color="214,0,126,200",
                      outline_color="120,0,70", outline_width=0.25)

    forest = {}
    for year in YEARS:
        ft = layer(f"임상도{year}", f"임상도 {year}")
        cats = [QgsRendererCategory(k, QgsFillSymbol.createSimple(
            {"color": v, "outline_style": "no"}), k) for k, v in FOREST_TYPE_COLORS.items()]
        ft.setRenderer(QgsCategorizedSymbolRenderer("임상", cats))
        forest[year] = ft

    pieces = {year: layer(f"숲가꾸기x임상도{year}", f"숲가꾸기×임상도{year} 교차") for year in YEARS}
    stand_pieces = layer("숲가꾸기x임소반", "숲가꾸기×임소반 교차")

    for lyr in (city, umd, stands, thinning, *forest.values()):
        lyr.saveStyleToDatabase(lyr.name(), "ForestChangeGeoAI 기본 스타일", True, "")

    order = [city, thinning, stands, umd, forest[2025], forest[2019], *pieces.values(), stand_pieces]
    project.addMapLayers(order, False)
    root = project.layerTreeRoot()
    for lyr in order:
        root.addLayer(lyr)
    for lyr in (forest[2019], *pieces.values(), stand_pieces):
        root.findLayer(lyr.id()).setItemVisibilityChecked(False)

    extent = city.extent()
    extent.grow(max(extent.width(), extent.height()) * 0.03)
    for year in YEARS:
        layout = make_layout(project, year, [city, thinning, stands, umd, forest[year]], extent)
        png = ROOT / "output" / f"춘천시_숲가꾸기_임상도{year}_지도.png"
        settings = QgsLayoutExporter.ImageExportSettings()
        settings.dpi = 200
        res = QgsLayoutExporter(layout).exportToImage(str(png), settings)
        assert res == QgsLayoutExporter.Success, res
        print("저장:", png)

    assert project.write(str(PROJECT))
    print("저장:", PROJECT)
    qgs.exitQgis()


if __name__ == "__main__":
    main()
