"""E9 핫스팟 격자·거리대역 민감도 (원 논문 hotspot_analysis 도구 재사용)."""
import json, sys
from pathlib import Path
ROOT = Path(r"C:\src\ForestChangeGeoAI"); FCA = ROOT / "forest-change-agent"; sys.path.insert(0, str(FCA))
from fcd import config as C
C.MGMT_PLAN_DIR, C.MGMT_PLAN_GLOB = ROOT, "춘천(9)*/*.shp"
from fcd import tools as T
out = []
for cell, band in [(250, 750), (500, 1000), (500, 1500), (500, 2000), (1000, 3000)]:
    s = T.Session(query="민감도")
    d = T.call(s, "hotspot_analysis", {"region_id": "nf:51110", "cell_m": cell, "band_m": band})["data"]
    row = {"격자_m": cell, "거리대역_m": band, **{k: d.get(k) for k in ("n_cells", "n_hot", "n_cold", "hot_changed_ha", "total_changed_ha")}}
    out.append(row); print(row, flush=True)
(Path(__file__).parent / "results" / "E9_hotspot_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
