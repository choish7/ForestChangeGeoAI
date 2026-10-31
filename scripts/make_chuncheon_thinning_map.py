"""춘천시 숲가꾸기공간정보 + 춘천시 경계를 스타일 지정한 QGIS 프로젝트(.qgz)와 지도 PNG로 만든다.

QGIS 내장 파이썬으로 실행:
  "C:/Program Files/QGIS 3.34.9/bin/python-qgis-ltr.bat" scripts/make_chuncheon_thinning_map.py
GPKG 안에도 각 레이어의 기본 스타일을 저장하므로, GPKG만 QGIS로 열어도 같은 모양으로 표시된다.
"""
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsCategorizedSymbolRenderer, QgsFillSymbol, QgsLayoutExporter, QgsLayoutItemLabel,
    QgsLayoutItemLegend, QgsLayoutItemMap, QgsLayoutItemScaleBar, QgsLayoutPoint, QgsLayoutSize,
    QgsPalLayerSettings, QgsPrintLayout, QgsProject, QgsRendererCategory, QgsTextBufferSettings,
    QgsTextFormat, QgsUnitTypes, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtGui import QColor, QFont

ROOT = Path(__file__).resolve().parent.parent
GPKG = ROOT / "output" / "춘천시_산림경영활동_숲가꾸기공간정보.gpkg"
PROJECT = ROOT / "output" / "춘천시_숲가꾸기공간정보.qgz"
PNG = ROOT / "output" / "춘천시_숲가꾸기공간정보_지도.png"

WORK_COLORS = {
    "풀베기": "#7fbf3f",
    "천연림보육": "#1b7837",
    "솎아베기": "#d95f02",
    "덩굴제거": "#7570b3",
    "어린나무가꾸기": "#e6ab02",
}


def layer(name):
    lyr = QgsVectorLayer(f"{GPKG}|layername={name}", name, "ogr")
    assert lyr.isValid(), name
    return lyr


def label(lyr, field, size, color, bold=False):
    fmt = QgsTextFormat()
    fmt.setFont(QFont("Malgun Gothic", size, QFont.Bold if bold else QFont.Normal))
    fmt.setSize(size)
    fmt.setColor(QColor(color))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.8)
    buf.setColor(QColor("white"))
    fmt.setBuffer(buf)
    s = QgsPalLayerSettings()
    s.fieldName = field
    s.setFormat(fmt)
    s.placement = QgsPalLayerSettings.OverPoint
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(s))
    lyr.setLabelsEnabled(True)


def main():
    qgs = QgsApplication([], False)
    qgs.initQgis()
    project = QgsProject.instance()
    project.setCrs(layer("춘천시경계").crs())

    umd = layer("춘천시_읍면동경계")
    umd.setName("읍면동 경계")
    umd.setRenderer(umd.renderer().__class__(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "style": "no", "outline_color": "150,150,150", "outline_width": "0.25",
         "outline_style": "dash"})))
    label(umd, "EMD_NM", 7, "#666666")

    city = layer("춘천시경계")
    city.setName("춘천시 경계")
    city.setRenderer(city.renderer().__class__(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "style": "no", "outline_color": "200,30,30", "outline_width": "0.9"})))

    thin = layer("숲가꾸기공간정보")
    thin.setName("숲가꾸기 사업지(2015)")
    cats = [QgsRendererCategory(k, QgsFillSymbol.createSimple(
        {"color": v, "outline_color": v, "outline_width": "0.2"}), k) for k, v in WORK_COLORS.items()]
    thin.setRenderer(QgsCategorizedSymbolRenderer("작업종", cats))

    for lyr in (umd, city, thin):
        # GPKG에 기본 스타일로 저장 → GPKG만 열어도 동일하게 표시
        lyr.saveStyleToDatabase(lyr.name(), "ForestChangeGeoAI 기본 스타일", True, "")
    order = [city, thin, umd]  # 위에서부터 시 경계 > 사업지 > 읍면동
    project.addMapLayers(order, False)
    for lyr in order:
        project.layerTreeRoot().addLayer(lyr)

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("춘천시 숲가꾸기")
    page = layout.pageCollection().page(0)
    page.setPageSize(QgsLayoutSize(297, 297, QgsUnitTypes.LayoutMillimeters))

    title = QgsLayoutItemLabel(layout)
    title.setText("춘천시 산림경영활동 숲가꾸기 공간정보 (2015)")
    title_fmt = QgsTextFormat()
    title_fmt.setFont(QFont("Malgun Gothic", 18, QFont.Bold))
    title_fmt.setSize(18)
    title.setTextFormat(title_fmt)
    title.attemptMove(QgsLayoutPoint(10, 6))
    title.attemptResize(QgsLayoutSize(277, 12))
    layout.addLayoutItem(title)

    m = QgsLayoutItemMap(layout)
    m.setLayers(order)
    m.attemptMove(QgsLayoutPoint(10, 20))
    m.attemptResize(QgsLayoutSize(277, 267))
    ext = city.extent()
    ext.grow(max(ext.width(), ext.height()) * 0.03)
    m.zoomToExtent(ext)
    m.setFrameEnabled(True)
    layout.addLayoutItem(m)

    legend = QgsLayoutItemLegend(layout)
    legend.setLinkedMap(m)
    legend.setTitle("범례")
    legend.attemptMove(QgsLayoutPoint(225, 190))  # 지도 우하단 빈 공간(홍천군 쪽)
    legend.setBackgroundColor(QColor(255, 255, 255, 220))
    legend.setBackgroundEnabled(True)
    layout.addLayoutItem(legend)

    bar = QgsLayoutItemScaleBar(layout)
    bar.setStyle("Single Box")
    bar.setLinkedMap(m)
    bar.setUnits(QgsUnitTypes.DistanceKilometers)
    bar.setUnitLabel("km")
    bar.setNumberOfSegments(4)
    bar.setUnitsPerSegment(2.5)
    bar.attemptMove(QgsLayoutPoint(200, 272))
    layout.addLayoutItem(bar)

    project.layoutManager().addLayout(layout)
    exporter = QgsLayoutExporter(layout)
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 200
    res = exporter.exportToImage(str(PNG), settings)
    assert res == QgsLayoutExporter.Success, res

    assert project.write(str(PROJECT))
    print("저장:", PROJECT)
    print("저장:", PNG)
    qgs.exitQgis()


if __name__ == "__main__":
    main()
