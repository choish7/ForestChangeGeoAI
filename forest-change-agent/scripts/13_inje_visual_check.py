# [통합 사본] 원본: paper/revision/inje_visual_check.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""인제군 육안 판독 체크리스트 (E10c): 춘천 모델 예측 유형별 25점 × 4 = 100점.

입력: inje/visual_check_points_raw.csv (Colab 셀 6 — 춘천 AE+S2 모델, 재학습 없음, 예측 유형별 무작위 25점)
산출 (inje/visual_check/):
  checklist_blind.csv   판독자용 — 라벨·예측 숨김, 판독 칸 비움 (Excel 에서 바로 열리도록 UTF-8 BOM)
  answer_key.csv        채점용 — 임상도 라벨, 모델 예측, 확률, AE 거리·정점, NDVI
  location_map.png      인제군 읍면 경계 위 번호 위치도
  points.kml            Google Earth Pro 과거 영상(2019·2025) 판독용 번호 표시
  chips/NNN_2019.png, NNN_2025.png  Sentinel-2 성장기(6–9월) 중앙값 RGB, 1 km × 1 km, 중심 표시
  index.html            번호·좌표·두 시기 영상·링크를 한 화면에
  python forest-change-agent/scripts/13_inje_visual_check.py [--no-chips]
"""
import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "work"   # forest-change-agent/work (inje/ 하위 폴더 사용)
ROOT = HERE.parents[1]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
OUT = Path(__file__).resolve().parents[1] / "outputs" / "inje_visual_check"
BOUNDARY = next((DATA / "LSMD_ADM_SECT_UMD_강원특별자치도").glob("*.shp"))
KO = {"no_change": "무변화", "harvest": "벌채·갱신", "conversion": "전용", "gain": "산림증가"}


def chips(df):
    import ee
    ee.Initialize(project="ee-choish7838")

    def s2(y):
        def mask(img):
            scl = img.select("SCL")
            return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(f"{y}-06-01", f"{y}-09-30")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)).map(mask).median().select(["B4", "B3", "B2"]))
    img = {y: s2(y) for y in (2019, 2025)}
    (OUT / "chips").mkdir(parents=True, exist_ok=True)

    def one(args):
        r, y = args
        f = OUT / "chips" / f"{r.no:03d}_{y}.png"
        if f.exists():
            return
        pt = ee.Geometry.Point([r.lon, r.lat])
        box = pt.buffer(500).bounds()
        center = ee.Image().paint(pt.buffer(20), 1, 2).visualize(palette=["ffff00"])
        vis = img[y].visualize(min=0, max=1500, gamma=1.3).blend(center)
        url = vis.getThumbURL({"region": box, "dimensions": 300, "format": "png"})
        for k in range(3):
            try:
                f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
                return
            except Exception:
                if k == 2:
                    raise

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, [(r, y) for r in df.itertuples() for y in (2019, 2025)]))


def location_map(df):
    b = gpd.read_file(BOUNDARY, encoding="cp949")
    b = b[b["COL_ADM_SE"].astype(str).eq("51810")].to_crs(5186)
    p = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(5186)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    b.plot(ax=ax, facecolor="#f2f5ef", edgecolor="#777", linewidth=0.6)
    p.plot(ax=ax, color="#d7301f", markersize=14, edgecolor="white", linewidth=0.5, zorder=3)
    for r in p.itertuples():
        ax.annotate(str(r.no), (r.geometry.x, r.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=5.5, zorder=4)
    x0, y0, x1, y1 = b.total_bounds
    ax.plot([x0 + 2000, x0 + 12000], [y0 + 2000, y0 + 2000], color="k", lw=2)
    ax.text(x0 + 7000, y0 + 2600, "10 km", ha="center", fontsize=8)
    ax.annotate("N", xy=(x1 - 3000, y1 - 2000), xytext=(x1 - 3000, y1 - 6000), ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color="k"))
    ax.set_title("Inje County visual-interpretation points (n = 100, numbers = checklist No.)", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "location_map.png")
    plt.close(fig)


def kml(df):
    pm = "".join(f"<Placemark><name>{r.no}</name><description>{r.sample_id}</description>"
                 f"<Point><coordinates>{r.lon},{r.lat},0</coordinates></Point></Placemark>" for r in df.itertuples())
    (OUT / "points.kml").write_text('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                                    f"<Document><name>Inje visual check (100)</name>{pm}</Document></kml>", encoding="utf-8")


def html(df):
    rows = "".join(
        f"<tr><td><b>{r.no}</b><br>{r.sample_id}<br>{r.lat:.5f}, {r.lon:.5f}<br>"
        f"<a href='https://www.google.com/maps/@{r.lat},{r.lon},700m/data=!3m1!1e3' target='_blank'>Google 위성</a> · "
        f"<a href='https://livingatlas.arcgis.com/wayback/#mapCenter={r.lon}%2C{r.lat}%2C17' target='_blank'>Esri Wayback</a></td>"
        f"<td><img src='chips/{r.no:03d}_2019.png'></td><td><img src='chips/{r.no:03d}_2025.png'></td></tr>" for r in df.itertuples())
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>인제 육안 판독 100점</title>"
        "<style>body{font-family:sans-serif;margin:16px}td{border-bottom:1px solid #ddd;padding:6px;vertical-align:top;font-size:13px}"
        "img{width:220px}</style><h2>인제군 육안 판독 100점 — Sentinel-2 성장기 RGB (1 km, 노란 원 = 판독 위치)</h2>"
        "<p>체크리스트(checklist_blind.csv)의 No.와 같은 번호입니다. 판독 시 고해상도 과거 영상(Google Earth Pro, Esri Wayback)을 함께 보세요.</p>"
        f"<table><tr><th>No.</th><th>2019</th><th>2025</th></tr>{rows}</table>", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chips", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(HERE / "inje" / "visual_check_points_raw.csv")
    df = raw.sample(frac=1, random_state=7).reset_index(drop=True)          # 판독 순서를 섞어 예측 유형이 드러나지 않게
    df.insert(0, "no", np.arange(1, len(df) + 1))
    blind = df[["no", "sample_id", "lon", "lat"]].copy()
    for c in ("판독자", "판독일", "영상_2019(출처·촬영일)", "영상_2025(출처·촬영일)", "피복_2019", "피복_2025",
              "변화_여부(Y/N/불명)", "판독_유형(무변화/벌채·갱신/전용/산림증가/기타)", "확신도(상/중/하)", "비고"):
        blind[c] = ""
    blind.to_csv(OUT / "checklist_blind.csv", index=False, encoding="utf-8-sig")
    key = df.assign(map_label_ko=df.map_label.map(KO), pred_type_ko=df.pred_type.map(KO))
    key.to_csv(OUT / "answer_key.csv", index=False, encoding="utf-8-sig")
    location_map(df)
    kml(df)
    if not a.no_chips:
        chips(df)
    html(df)
    print(pd.crosstab(df.pred_type, df.map_label, margins=True).to_string())
    print("완료:", OUT)


if __name__ == "__main__":
    sys.exit(main())
