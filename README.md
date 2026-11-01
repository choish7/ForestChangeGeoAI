# ForestChangeGeoAI

This repository holds the code, distributable derived data and trained models for **"Integrating AlphaEarth Satellite Embeddings and Natural Language GeoAI for Traceable Forest Change Monitoring in South Korea"** (Choi & Jung, *Remote Sensing*, under revision).

The study evaluates annual [AlphaEarth Foundations](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) satellite embeddings against forest-map change labels in Chuncheon, Korea. The results are delivered through a tool-based GeoAI agent and a decision-support web service, both of which check every reported number against tool outputs. A transfer test in Inje County checks how the approach carries over to another region.

| Folder | Contents |
|---|---|
| [`forest-change-agent/`](forest-change-agent/) | Benchmark and full experiment pipeline. The 15 steps in paper order are in [`EXPERIMENTS.md`](forest-change-agent/EXPERIMENTS.md); see also the folder [`README.md`](forest-change-agent/README.md) |
| [`forest_decision_agent/`](forest_decision_agent/) | Decision-support web service (FastAPI + Leaflet; Korean/English UI; Claude agent or rule-based mode). See its [`README.md`](forest_decision_agent/README.md), which includes an English summary |

`MANIFEST.csv` lists every file with its size and SHA-256 checksum.

## What is included

| Type | Files |
|---|---|
| Source code | `forest-change-agent/fcd/` (shared modules), `scripts/` (experiments 01–13), `prep/` (data preparation), `colab/` (step-numbered Colab notebooks and modules); `forest_decision_agent/*.py`, `static/index.html` |
| Documents | `forest-change-agent/README.md`, `EXPERIMENTS.md`, `data/README.md` (raw-data sources); `forest_decision_agent/README.md` |
| Sample data | `forest-change-agent/work/samples.parquet` and `features.parquet` (6,027 Chuncheon points with labels and GEE features); `work/inje/inje_samples.parquet` (2,100 Inje points) |
| Preprocessed data | `forest-change-agent/release_preprocessed/` (boundaries, conversion-notice parcels and labels, forest masks, thinning sites). Non-public layers and columns are removed; see its `README.md` |
| Models | `forest-change-agent/outputs/change_model.joblib` (benchmark classifier), `models/ae_class_models/*_local.joblib` (AlphaEarth classifiers), `models/s2_tsfm_chuncheon/` (S2-TSFM weights and classifier) |
| Results | `forest-change-agent/outputs/metrics.json`, `ablation.csv`, `results/` (experiments E1–E10), `model_reports/`, `inje_visual_check/` (blind visual-interpretation sheet) |

## What is not included

| Data | Why | Where to get it |
|---|---|---|
| Raw public data: forest type maps, cadastral map, township boundaries, thinning sites, conversion notices | Downloadable from official portals | Download links in [`forest-change-agent/data/README.md`](forest-change-agent/data/README.md) |
| Satellite data: AlphaEarth, Sentinel-2, SRTM / Copernicus DEM, ESA WorldCover | Public via Google Earth Engine; large | Export with `forest-change-agent/colab/01_위성래스터_내보내기.ipynb` (dataset IDs in `data/README.md`) |
| National-forest management plans and stand maps, and everything derived from them | Non-public (Northern Regional Forest Service) | On request to the data provider |
| Wildfire and landslide reports, forest roads | Internal data | — |
| Large derived files (> 10 MB): change-label polygons, S2-TSFM score and probability rasters, 2013–2025 change-label gpkg | Size; can be regenerated | `scripts/01_build_change_labels.py`, `prep/build_change_labels_2013_2019_2025.py`, `colab/05_S2-TSFM_사전학습.ipynb` |
| Decision-support evidence DB and runtime rasters (`forest_decision_agent/data/`), query records | Contain non-public and internal data | Build them with the steps in `forest_decision_agent/README.md` |

Because of these exclusions, the web service does not run straight from this repository. It needs the evidence DB, which is built from the raw data (including the management plan, used with the provider's permission).

## Setup

```bash
pip install -r requirements.txt
earthengine authenticate              # Google Earth Engine account and cloud project
set GEE_PROJECT=<your-project>        # default in code: ee-choish7838
set FCD_DATA_DIR=<raw-data folder>    # default: forest-change-agent/data/
```

## Quick check without raw data

This snippet reproduces the hold-out F1 values of Table 5 from the included sample features in about one minute:

```python
import sys, pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "forest-change-agent")
from fcd import features

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

## Main results

All figures below are F1 scores for change detection on the 1,084-point spatial-block hold-out unless stated otherwise.

| Comparison | F1 |
|---|---|
| AlphaEarth features | 0.933 |
| Two-date Sentinel-2 indices | 0.772 |
| AlphaEarth, two endpoint years only | 0.926 |
| Annual Sentinel-2 indices (2019–2025) | 0.887 |
| Annual Sentinel-2 indices + S2-TSFM | 0.896 |

- **Change type**: conditional change-type macro-F1 is 0.777; conversion is the hardest class (F1 = 0.621).
- **Inje transfer**: without retraining, the Chuncheon AE+S2 model reaches F1 = 0.823 in Inje County, and 0.861 after threshold calibration.
- **Report checks**: across 14 saved decision-support reports, all 973 checked numerical tokens matched tool outputs.

## Notes

- Generative AI (Claude Code, ChatGPT) assisted with code writing, debugging and documentation. All reported numbers come from the deterministic scripts in this repository.
- License: to be decided by the authors before public release.
- Contact: Sin-Hyung Choi (University of Seoul / Korea Forest Service).
