# [통합 사본] 원본: scripts/build_ae_class_samples.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""AlphaEarth 분류 모델용 표본 추출 (Colab 학습용) → output/ae_class_samples.npz

과업
  LC  산지 피복(2019·2025, 임상도 속성만): 산림 = 입목지(STORUNST 1), 나지 = 미립목지(81),
      초지 = 관목덤불(83)·초지(92), 시설지 = 제지(82, 임도·묘지·송전탑 등 산림 외 용도)·주거지(91)
      임상도 범위(산지) 안의 피복 구분이며, 시가지·농경지는 학습 범위 밖
  FT  임상(2019·2025): 침엽수림 / 활엽수림 / 혼효림  — 각 연도 임상도와 같은 연도 임베딩
  CH  변화유형(2019→2025): 무변화 / 벌채·갱신 / 산림증가 (산지전용 제외) — 2013·2019·2025 변화라벨 gpkg, 신뢰도 high
표본: 폴리곤을 10 m 안쪽으로 줄인 뒤(경계 오차) 면적 비례로 점 배치, 폴리곤당 최대 k점, 같은 화소 중복 제거
분할: AlphaEarth 10 m 격자 500×500 화소(5 km) 블록 중 20%(seed 42)를 독립 검증, 폴리곤 대표점의 블록으로 폴리곤 전체 표본을 같은 쪽에 배정
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine, rowcol

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "forest-change-agent" / "data"   # 원자료 (forest-change-agent/data/README.md)
TF = Affine(10, 0, 368350, 0, -10, 4216210)
H, W = 4549, 4647
CRS = 32652
BLOCK, SEED = 500, 42
rng = np.random.default_rng(SEED)
nb_r, nb_c = -(-H // BLOCK), -(-W // BLOCK)
HOLD = rng.random(nb_r * nb_c) < 0.2


def norm(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def sample(g, label, n_max, per_poly, task, year):
    g = g.to_crs(CRS)
    g = g[g.geometry.notna()].copy()
    g["geometry"] = g.geometry.buffer(-10)
    g = g[~g.geometry.is_empty & g.geometry.notna() & (g.area > 100)].reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()
    k = np.minimum(per_poly, np.maximum(1, np.round(g.area / g.area.sum() * n_max * 1.3))).astype(int)
    pts = g.geometry.sample_points(k, rng=SEED)
    rp = g.representative_point()
    pr, pc = rowcol(TF, rp.x.values, rp.y.values)
    rows = []
    for i, mp in enumerate(pts):
        if mp is None or mp.is_empty:
            continue
        geoms = getattr(mp, "geoms", [mp])
        xs, ys = [p.x for p in geoms], [p.y for p in geoms]
        r, c = rowcol(TF, xs, ys)
        b = (np.clip(pr[i], 0, H - 1) // BLOCK) * nb_c + np.clip(pc[i], 0, W - 1) // BLOCK
        for rr, cc in zip(r, c):
            if 0 <= rr < H and 0 <= cc < W:
                rows.append((task, year, label, f"{task}{year}{label}_{i}", rr, cc, int(HOLD[b])))
    d = pd.DataFrame(rows, columns=["task", "year", "label", "poly", "row", "col", "test"]).drop_duplicates(["task", "year", "row", "col"])
    if len(d) > n_max:
        d = d.sample(n_max, random_state=SEED)
    print(f"  {task} {year} {label}: 폴리곤 {len(g):,} → 표본 {len(d):,} (검증 {int(d.test.sum()):,})")
    return d


def main():
    out = []
    ft = {}
    for y in (2019, 2025):
        f = gpd.read_file(DATA / f"임상도 {y} 51110" / "51110.shp", encoding="cp949", columns=["STORUNST", "FRTP_CD", "KOFTR_GROU", "geometry"])
        for c in ("STORUNST", "FRTP_CD", "KOFTR_GROU"):
            f[c] = f[c].map(norm)
        ft[y] = f
        for code, lab in (("1", "침엽수림"), ("2", "활엽수림"), ("3", "혼효림")):
            out.append(sample(f[f.FRTP_CD.eq(code) & f.STORUNST.eq("1")], lab, 6000, 2, "FT", y))
    for y, f in ft.items():          # 임상도 속성만 사용 (산지 범위 안)
        out.append(sample(f[f.STORUNST.eq("1")], "산림", 3000, 1, "LC", y))
        out.append(sample(f[f.KOFTR_GROU.eq("81")], "나지", 3000, 6, "LC", y))
        out.append(sample(f[f.KOFTR_GROU.isin(["83", "92"])], "초지", 3000, 15, "LC", y))
        out.append(sample(f[f.KOFTR_GROU.isin(["82", "91"])], "시설지", 3000, 3, "LC", y))
    ch = gpd.read_file(ROOT / "output" / "춘천시_산림변화라벨_2013_2019_2025.gpkg", layer="변화라벨",
                       columns=["P2019_2025_type", "P2019_2025_conf", "geometry"])
    ch = ch[ch["P2019_2025_conf"].eq("high")]
    for t, lab, n, k in (("no_change", "무변화", 6000, 1), ("harvest", "벌채·갱신", 5000, 6), ("gain", "산림증가", 5000, 10)):
        out.append(sample(ch[ch["P2019_2025_type"].eq(t)], lab, n, k, "CH", 2025))
    d = pd.concat(out, ignore_index=True)
    np.savez_compressed(ROOT / "output" / "ae_class_samples.npz", task=d.task.to_numpy("U2"), year=d.year.to_numpy("int16"),
                        label=d.label.to_numpy("U8"), poly=pd.factorize(d.poly)[0].astype("int32"),
                        row=d.row.to_numpy("int16"), col=d.col.to_numpy("int16"), test=d.test.to_numpy("int8"))
    print(d.groupby(["task", "year", "label"]).size().to_string())
    print("저장:", ROOT / "output" / "ae_class_samples.npz", (ROOT / "output" / "ae_class_samples.npz").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
