# ForestChangeGeoAI

Code, derived data and trained models for **"Integrating AlphaEarth Satellite Embeddings and Natural Language GeoAI for Traceable Forest Change Monitoring in South Korea"** (Choi & Jung, *Remote Sensing*, under revision).

The project evaluates annual [AlphaEarth Foundations](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) satellite embeddings against forest-map change labels in Chuncheon, Korea. It then exposes the analysis through tool-based GeoAI that checks every reported number against tool outputs. A transfer test in Inje County checks how the approach carries over to another region.

| Component | Folder | What it does |
|---|---|---|
| Change benchmark | `forest-change-agent/` | Forest-map transition labels → stratified samples → GEE features (AlphaEarth, Sentinel-2, terrain) → gradient boosting with 5 km spatial blocks; Gemini/Claude tool agent with numeric checks |
| Data preparation | `scripts/` | GEE exports, label rasters, yearly forest-type labels, management records, fire/landslide layers |
| Sentinel-2 foundation baseline | `mini_gfm/`, `mini_gfm_chuncheon/` | Mini-GFM masked autoencoder on Sentinel-2 time series; change classifier |
| Final change map | `final_change_2019_2025/` | AlphaEarth + Sentinel-2 change map 2019–2025 (class, year, patches) |
| AlphaEarth classifiers | `ae_class_models/` | Land cover, forest type and change type models trained from forest-map attributes |
| Decision-support service | `forest_decision_agent/` | FastAPI + Leaflet web app (Korean/English), Claude agent or rule-based mode, traceable reports |
| Revision experiments | `paper/revision/` | Reviewer-response experiments E1–E10 including the Inje transfer test and visual-check sheet |
| Docs | `docs/` | Data sources and access, regional extension guide (Korean) |

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate          # Google Earth Engine account and a cloud project are required
set GEE_PROJECT=<your-project>    # default in code: ee-choish7838
set FCD_DATA_DIR=<folder with raw forest maps>   # see docs/DATA.md
```

Raw forest type maps, national-forest management plans and administrative boundaries are **not** redistributed here; `docs/DATA.md` explains how to obtain them and which results can be reproduced without them.

## Sample data

The repository includes the sample-level data needed to rerun the classification benchmark without Earth Engine or raw maps.

| File | Rows | Content |
|---|---|---|
| `forest-change-agent/work/samples.parquet` | 6,027 | Chuncheon points: `change_type` (no_change / harvest / conversion / gain), 2019 forest attributes, `block_id` (5 km), `split` (train / test_spatial), `lon`, `lat` |
| `forest-change-agent/work/features.parquet` | 6,027 | Same points with GEE values: `ae{2017..2025}_A00..A63` (AlphaEarth), `ndvi2019`, `ndvi2025`, `nbr2019`, `nbr2025`, `elev`, `slope` |
| `paper/revision/inje/inje_samples.parquet` | 2,100 | Inje County transfer-test points (same label rules) |
| `paper/revision/inje/visual_check/checklist_blind.csv` | 100 | Blind visual-interpretation sheet; `answer_key.csv` holds map labels and model predictions |

Quick check (about one minute; reproduces the hold-out F1 values of Table 4):

```python
import sys, pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, "forest-change-agent")
from fcd import features
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_parquet("forest-change-agent/work/features.parquet")
df = df[df.change_type.isin(["no_change", "harvest", "conversion", "gain"])].reset_index(drop=True)
X, G = features.build(df)
tr, te, y = df.split.eq("train"), df.split.eq("test_spatial"), df.changed.astype(int)
for name in (["AE"], ["S2"], ["AE", "S2", "TERR"]):
    cols = sum((G[g] for g in name), [])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                       class_weight="balanced", random_state=42).fit(X.loc[tr, cols], y[tr])
    print("+".join(name), round(f1_score(y[te], m.predict(X.loc[te, cols])), 3))   # AE 0.933, S2 0.772, AE+S2+TERR 0.936
```

## Reproducing the paper

| Paper item | Script | Needs raw data? |
|---|---|---|
| Table 2 (transition classes), Fig. 2 | `forest-change-agent/scripts/01_build_change_labels.py`, `02_sample_points.py` | Forest maps |
| Table 4, Fig. 3–5 (feature comparison, confusion, embedding peaks) | `forest-change-agent/scripts/04_train_evaluate.py` | No — uses `forest-change-agent/work/features.parquet` |
| GEE feature extraction | `forest-change-agent/scripts/03_extract_gee_features.py` | GEE only |
| Hotspots, compartment ranking (Table 5, Fig. 6) | `forest-change-agent/fcd/spatial.py`, `fcd/tools.py` | Forest maps, management plan |
| Agent demonstration (Section 4.6, Table 6) | `forest-change-agent/scripts/05_run_agent.py` | All of the above + LLM key |
| E1–E6 fair comparison, polygon overlap, fold composition, false negatives by type, excluded classes | `paper/revision/revision_experiments.py` | Features parquet (E1–E4); forest maps (E6) |
| E5b/E5c timing against dated records | `paper/revision/revision_timing.py` | Management records (E5b), conversion notices (E5c) |
| E7 composite vs. area ranking | `paper/revision/revision_e7.py` | Management plan |
| E8 report checks | `paper/revision/revision_report_check.py` | Saved service runs |
| E9 hotspot sensitivity | `paper/revision/revision_hotspot.py` | Forest maps, management plan |
| E10 Inje transfer | `paper/revision/inje_prepare.py` → `inje_transfer_colab.py` → `inje_visual_check.py` | Inje forest maps (GEE for features) |

Results of each experiment are stored in `paper/revision/results/`. `MANIFEST.csv` lists every file with its SHA-256 checksum.

Main results on the 1,084-point spatial-block hold-out: AlphaEarth features F1 = 0.933, two-date Sentinel-2 indices F1 = 0.772, and annual Sentinel-2 indices (2019–2025) F1 = 0.887. AlphaEarth with only the two endpoint years reaches F1 = 0.926. Without retraining, the Chuncheon model reaches F1 = 0.823 (AlphaEarth + Sentinel-2) in Inje County, or 0.861 after threshold recalibration.

## Decision-support web service

```bash
python -m forest_decision_agent.web     # http://127.0.0.1:8800  (add ?lang=en for English)
```

The service needs the evidence database built by `forest_decision_agent/build_*.py` from the raw data (see `forest_decision_agent/README.md`). Put a Claude API key in `forest_decision_agent/claude_key.txt` or `ANTHROPIC_API_KEY` for agent mode; without a key the service runs in rule-based mode. Screenshots are in `forest_decision_agent/docs/screens/`.

## Notes

- Generative AI (Claude Code, ChatGPT) assisted with code writing, debugging and documentation; all reported numbers come from the deterministic scripts above.
- License: to be decided by the authors before public release.
- Contact: Sin-Hyung Choi (University of Seoul / Korea Forest Service).
