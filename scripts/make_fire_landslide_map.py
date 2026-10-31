"""춘천시 산불·산사태 점 공간정보를 QGIS 프로젝트(발생일 시간 슬라이더)와 지도 PNG로 만든다.

QGIS 내장 파이썬으로 실행:
  "C:/Program Files/QGIS 3.34.9/bin/python-qgis-ltr.bat" scripts/make_fire_landslide_map.py
지도 PNG에는 실제 산불(진화완료)과 유효 산사태 신고만 그리고, 신고삭제·오인 건은 프로젝트에 꺼진 레이어로 둔다.
"""
from datetime import datetime
from pathlib import Path

from qgis.core import (
    QgsApplication, QgsDateTimeRange, QgsFillSymbol, QgsLayoutExporter, QgsMarkerSymbol, QgsProject,
    QgsSingleSymbolRenderer, QgsUnitTypes, QgsVectorLayer, QgsVectorLayerTemporalProperties,
)
from qgis.PyQt.QtCore import QDateTime

from make_chuncheon_thinning_map import label
from make_combined_map import make_layout

ROOT = Path(__file__).resolve().parent.parent
GPKG = ROOT / "output" / "춘천시_산불_산사태.gpkg"
STANDS = ROOT / "output" / "춘천_임소반_산림사업실적.gpkg"
PROJECT = ROOT / "output" / "춘천시_산불_산사태.qgz"
PNG = ROOT / "output" / "춘천시_산불_산사태_지도.png"


def layer(name, title, path=GPKG, subset=None):
    lyr = QgsVectorLayer(f"{path}|layername={name}", title, "ogr")
    assert lyr.isValid(), name
    if subset:
        lyr.setSubsetString(subset)
    return lyr


def marker(lyr, **props):
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({k: str(v) for k, v in props.items()})))
    return lyr


def polygon(lyr, **props):
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({k: str(v) for k, v in props.items()})))
    return lyr


def by_date(lyr, field):
    tp = lyr.temporalProperties()
    tp.setIsActive(True)
    tp.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeInstantFromField)
    tp.setStartField(field)
    tp.setFixedDuration(1)
    tp.setDurationUnits(QgsUnitTypes.TemporalYears)  # 슬라이더를 연 단위로 넘길 때 그 해 건이 보이도록


def main():
    qgs = QgsApplication([], False)
    qgs.initQgis()
    project = QgsProject.instance()

    city = polygon(layer("춘천시경계", "춘천시 경계"), color="0,0,0,0", style="no", outline_color="200,30,30", outline_width=0.9)
    project.setCrs(city.crs())
    umd = polygon(layer("춘천시_읍면동경계", "읍면동 경계"), color="0,0,0,0", style="no",
                  outline_color="90,90,90", outline_width=0.25, outline_style="dash")
    label(umd, "EMD_NM", 7, "#444444")
    stands = polygon(layer("임소반_실적요약", "국유림 임소반", STANDS), color="215,230,215", outline_color="170,190,170",
                     outline_width=0.1)

    fire = marker(layer("산불", "산불(진화완료)", subset="\"구분\" = '산불'"), name="circle", color="230,40,30",
                  outline_color="120,0,0", size=2.6)
    fire_other = marker(layer("산불", "산불외 종료(오인 등)", subset="\"구분\" = '산불외 종료(오인 등)'"),
                        name="circle", color="255,170,60", outline_color="150,90,0", size=1.8)
    fire_deleted = marker(layer("산불", "산불 신고삭제", subset="\"구분\" = '신고삭제'"), name="circle",
                          color="170,170,170", outline_color="110,110,110", size=1.4)
    slide = marker(layer("산사태", "산사태(유효신고)", subset="\"구분\" = '유효신고'"), name="triangle",
                   color="120,70,20", outline_color="60,30,0", size=3.2)
    slide_deleted = marker(layer("산사태", "산사태 신고삭제", subset="\"구분\" = '신고삭제'"), name="triangle",
                           color="190,170,150", outline_color="120,100,80", size=2.2)
    for lyr in (fire, fire_other, fire_deleted):
        by_date(lyr, "발생일")
    for lyr in (slide, slide_deleted):
        by_date(lyr, "신고일")

    order = [city, slide, fire, fire_other, slide_deleted, fire_deleted, umd, stands]
    project.addMapLayers(order, False)
    root = project.layerTreeRoot()
    for lyr in order:
        root.addLayer(lyr)
    for lyr in (fire_other, slide_deleted, fire_deleted):
        root.findLayer(lyr.id()).setItemVisibilityChecked(False)

    ts = project.timeSettings()
    ts.setTemporalRange(QgsDateTimeRange(QDateTime(datetime(2016, 1, 1)), QDateTime(datetime(2026, 12, 31, 23, 59))))
    ts.setTimeStep(1)
    ts.setTimeStepUnit(QgsUnitTypes.TemporalYears)
    ts.setFramesPerSecond(1)

    extent = city.extent()
    extent.grow(max(extent.width(), extent.height()) * 0.03)
    layout = make_layout(project, 0, [city, slide, fire, umd, stands], extent)
    layout.setName("산불·산사태")
    title = [i for i in layout.items() if i.__class__.__name__ == "QgsLayoutItemLabel"][0]
    title.setText("춘천시 산불(2016~2026, 진화완료) · 산사태 신고(2023~2026, 유효)")
    settings = QgsLayoutExporter.ImageExportSettings()
    settings.dpi = 200
    assert QgsLayoutExporter(layout).exportToImage(str(PNG), settings) == QgsLayoutExporter.Success
    print("저장:", PNG)
    assert project.write(str(PROJECT))
    print("저장:", PROJECT)
    qgs.exitQgis()


if __name__ == "__main__":
    main()
