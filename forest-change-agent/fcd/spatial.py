"""[4.3] 공간집계·군집 분석 (격자 집계, Getis-Ord Gi*, FDR 보정)."""
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm
from shapely.geometry import box


def grid_aggregate(polys: gpd.GeoDataFrame, extent, cell_m: int, value_col: str = "area_ha") -> gpd.GeoDataFrame:
    """폴리곤 면적을 정방격자에 배분 집계 (면적가중 교차)."""
    xmin, ymin, xmax, ymax = extent.bounds
    xs = np.arange(xmin - xmin % cell_m, xmax + cell_m, cell_m)
    ys = np.arange(ymin - ymin % cell_m, ymax + cell_m, cell_m)
    cells = gpd.GeoDataFrame({"cell_id": range(len(xs) * len(ys))},
                             geometry=[box(x, y, x + cell_m, y + cell_m) for x in xs for y in ys], crs=polys.crs)
    cells = cells[cells.intersects(extent)].reset_index(drop=True)
    if polys.empty:
        cells[value_col] = 0.0
        return cells
    inter = gpd.overlay(polys[["geometry"]], cells, how="intersection", keep_geom_type=True)
    agg = (inter.area / 1e4).groupby(inter["cell_id"]).sum()
    cells[value_col] = cells["cell_id"].map(agg).fillna(0.0)
    return cells


def fdr_bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg: 유의 여부 bool 배열."""
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, n + 1) / n
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    sig = np.zeros(n, bool)
    sig[order[:k]] = True
    return sig


def getis_ord_gi_star(values: np.ndarray, xy: np.ndarray, band_m: float) -> tuple[np.ndarray, np.ndarray]:
    """거리대역 이진가중(자기 자신 포함) Gi* z-score 와 양측 p-value."""
    x = np.asarray(values, float)
    n = len(x)
    xbar, s = x.mean(), x.std()
    if n < 3 or s == 0:
        return np.zeros(n), np.ones(n)
    nbrs = cKDTree(xy).query_ball_point(xy, r=band_m)
    z = np.empty(n)
    for i, nb in enumerate(nbrs):
        w = len(nb)                                   # Σw_ij = Σw_ij² (이진가중)
        num = x[nb].sum() - xbar * w
        den = s * np.sqrt((n * w - w ** 2) / (n - 1))
        z[i] = num / den if den > 0 else 0.0
    return z, 2 * norm.sf(np.abs(z))
