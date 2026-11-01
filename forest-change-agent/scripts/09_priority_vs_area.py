# [통합 사본] 원본: paper/revision/revision_e7.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
ROOT = Path(__file__).resolve().parents[2]; FCA = ROOT / "forest-change-agent"; sys.path.insert(0, str(FCA))
from fcd import config as C
OUT = Path(__file__).resolve().parents[1] / "outputs" / "results"
res = {}
# ---------- E7 변화면적 순위 vs 복합 순위 (원 논문 국유림 112 임반)
from fcd import tools as FT
from fcd.data import change_polygons, load_mgmt_plan
reg = FT._region("nf:51110")
mp = load_mgmt_plan()
mp = mp.iloc[mp.sindex.query(reg.geometry, predicate="intersects")]
mp = mp[mp.representative_point().within(reg.geometry)]
cmp_ = mp.dissolve("cmblc_id", aggfunc={"area_ha": "sum", "stock_m3": "sum"}).reset_index()
cpp = FT._clip(change_polygons(), reg.geometry)
inter = gpd.overlay(cpp[cpp["change_type"].isin(["harvest", "conversion"])][["change_type", "geometry"]],
                    cmp_[["cmblc_id", "geometry"]], how="intersection", keep_geom_type=True)
inter["ha"] = inter.area / 1e4
piv = inter.pivot_table(index="cmblc_id", columns="change_type", values="ha", aggfunc="sum").fillna(0)
tt = cmp_.set_index("cmblc_id")[["area_ha", "stock_m3"]].apply(pd.to_numeric, errors="coerce")
tt["harvest_ha"], tt["conversion_ha"] = piv.get("harvest", 0.0), piv.get("conversion", 0.0)
tt = tt.fillna(0.0)
tt["change_rate"] = (tt.harvest_ha + tt.conversion_ha) / tt.area_ha.replace(0, np.nan)
tt["stock_m3_per_ha"] = tt.stock_m3 / tt.area_ha.replace(0, np.nan)
tt = tt.fillna(0.0)
w = FT.DEFAULT_WEIGHTS
norm = (tt[list(w)] - tt[list(w)].min()) / (tt[list(w)].max() - tt[list(w)].min()).replace(0, 1)
tt["복합점수"] = norm.to_numpy() @ np.array(list(w.values()))
tt["변화면적_ha"] = tt.harvest_ha + tt.conversion_ha
tt["복합순위"] = tt["복합점수"].rank(ascending=False, method="min").astype(int)
tt["면적순위"] = tt["변화면적_ha"].rank(ascending=False, method="min").astype(int)
top_c, top_a = set(tt.nsmallest(10, "복합순위").index), set(tt.nsmallest(10, "면적순위").index)
only_c = tt.loc[sorted(top_c - top_a)][["복합순위", "면적순위", "harvest_ha", "conversion_ha", "change_rate", "stock_m3_per_ha"]]
only_a = tt.loc[sorted(top_a - top_c)][["복합순위", "면적순위", "harvest_ha", "conversion_ha", "change_rate", "stock_m3_per_ha"]]
from scipy.stats import spearmanr
res["E7_순위비교"] = {"임반": int(len(tt)), "상위10_공통": len(top_c & top_a),
                   "spearman_rho": round(float(spearmanr(tt["복합점수"], tt["변화면적_ha"]).statistic), 3),
                   "복합순위에만_있는_임반": json.loads(only_c.round(3).reset_index().to_json(orient="records", force_ascii=False)),
                   "면적순위에만_있는_임반": json.loads(only_a.round(3).reset_index().to_json(orient="records", force_ascii=False))}
print("[E7]", json.dumps(res["E7_순위비교"], ensure_ascii=False)[:800], flush=True)

(OUT / "E7_ranking.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
