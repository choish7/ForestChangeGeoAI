# Forest Change Dataset, Benchmark and GeoAI Agent

This folder holds the code and results for the paper *Integrating AlphaEarth Satellite Embeddings and Natural Language GeoAI for Traceable Forest Change Monitoring in South Korea* (`../paper/Choi&Jung_Text_revised.docx`).

> **The full experiment procedure is in [`EXPERIMENTS.md`](EXPERIMENTS.md).** It covers 15 steps in paper order: data preparation, Colab notebooks, scripts 01–13, and a check of each manuscript number against its result file.
The study area is Chuncheon (administrative code 51110), with a transfer test in Inje County (51810). Sentinel-2 stands in for CAS500-4 imagery until a CAS500-4 time series is available.

```
data/ (raw inputs)                                  Google Earth Engine
 ├ forest type maps 2013·2019·2024·2025 ─┐          AlphaEarth annual embeddings 2017–2025 (64-d, 10 m)
 ├ national-forest plan stands + records ─┤          Sentinel-2 L2A growing-season indices (CAS500-4 stand-in)
 └ conversion notices, cadastre, ...      │          SRTM elevation·slope
                                          ▼                      │
 01 map overlay → transition labels ──► 02 stratified samples ──► 03 GEE features ──► 04 train·evaluate (T1–T3)
       (work/change_polygons)              (5 km spatial blocks)     (work/features)      (outputs/)
                                          │                                                 │
                                          ├──► 06 controlled comparison · 07 event timing · 08 hotspot sensitivity
                                          │    09 priority vs area · 10 report checks · 11–13 Inje transfer
                                          └──────────► 05 GeoAI agent (tool use) ◄──────────┘
                                                       query → retrieval → RS screening → GIS → MCDA → decision card
```

## Folder layout

| Folder | Contents |
|---|---|
| `data/` | Raw inputs, kept under their original names: forest type maps, management plan, conversion notices, cadastre, boundaries, thinning records, fire/landslide reports, Inje data. See `data/README.md`. **Contains non-public data; never publish.** |
| `fcd/` | Shared modules: label rules, GEE features, spatial analysis, agent tools, numeric grounding |
| `prep/` | Local preparation of administrative datasets and pixel labels (copies of `../scripts/`), plus `build_release_preprocessed.py` |
| `colab/` | Colab/GPU/GEE notebooks named by experiment step, each with its modules (`stepNN_*.py`): `01_위성래스터_내보내기`, `05_S2-TSFM_사전학습`, `08_최종변화지도`, `12_AE분류모델_서비스용`, `15_인제군_전이시험` |
| `scripts/` | Main experiments 01–13 (see below) |
| `models/` | Trained models (the only copy): `ae_class_models/` (AE classifiers LC·FT·CH) and `s2_tsfm_chuncheon/` (S2-TSFM weights, classifier, change-score rasters) |
| `work/` | Intermediate files: polygons, samples, features, regions, `inje/` transfer samples |
| `outputs/` | `metrics.json`, `ablation.csv`, `change_model.joblib`, `cards/` (agent decision cards), `results/` (E1–E10), `model_reports/`, `inje_visual_check/` |
| `release_preprocessed/` | Distributable preprocessing results from `../output/`. Non-public layers and columns are removed, and fire/landslide data are excluded. See its `README.md` and `MANIFEST.csv` |

The decision-support web service is a separate application in `../forest_decision_agent/`. The S2-TSFM source code is in `../s2_tsfm/`, and the trained model is in `models/s2_tsfm_chuncheon/`. The model was called "Mini-GFM" during development.

The decision-support service (`../forest_decision_agent/`) reads its models from `models/`, its helper modules from `prep/` and its forest roads from `data/`. The service README gives the full run order: raw data → Colab exports → `prep/` → model training → service DB → web app.

## Setup

```bash
pip install -r requirements.txt
python -c "import ee; ee.Authenticate()"    # once; project from GEE_PROJECT (default ee-choish7838)
```

Raw data are read from `data/` by default; set `FCD_DATA_DIR` to use another folder. Large satellite rasters are not in `data/`. Export them with `colab/01_위성래스터_내보내기.ipynb` and place them in `../output/`.

## Running the experiments

```bash
# dataset and benchmark (Sections 3.1–3.3)
python scripts/01_build_change_labels.py      # forest-map overlay and transition labels (~3 min)
python scripts/02_sample_points.py            # 6,027 stratified points, 5 km blocks
python scripts/03_extract_gee_features.py     # AlphaEarth 9 yr × 64 + S2 + DEM (~1–2 min)
python scripts/04_train_evaluate.py           # T1/T1b/T2/T3, feature ablation, spatial validation

# analyses added for the revision
python scripts/06_controlled_comparison.py    # E1–E6: matched time windows, polygon overlap, folds, misses
python scripts/07_event_timing.py             # E5b/E5c: timing vs dated harvest records and conversion notices
python scripts/08_hotspot_sensitivity.py      # E9: Gi* cell size and distance band
python scripts/09_priority_vs_area.py         # E7: composite vs area-only compartment ranking
python scripts/10_report_checks.py            # E8: numeric and ranking-row checks of service reports
python scripts/11_inje_prepare.py             # E10: Inje labels and samples → upload to GEE asset
python scripts/12_inje_transfer.py            #      transfer without retraining (or colab/15_*.ipynb)
python scripts/13_inje_visual_check.py        #      100-point blind visual-interpretation sheet

# GeoAI agent (Section 3.6)
python scripts/05_run_agent.py --demo         # tool pipeline without an LLM
export GEMINI_API_KEY=...                     # Gemini backend (default)
# export ANTHROPIC_API_KEY=... ; add --provider claude
python scripts/05_run_agent.py "북산면 국유림에서 2019~2025년 벌채·산지전용 현황과 현장점검 우선 임반을 알려줘"
python scripts/05_run_agent.py -i             # interactive
```

`EXPERIMENTS.md` gives the full order, including the Colab steps and the dependencies. For example, `07` needs the decision-support database and the final change map.

Re-running scripts 06–13 overwrites the copies of the paper results in `outputs/results/` and `work/inje/`. The originals are kept in `../paper/revision/`.

## 1. Dataset (Section 3.1)

| Item | Description |
|---|---|
| Labels | 2019 × 2025 forest type maps overlaid in EPSG:5179; fragments < 0.1 ha removed; attribute rules in `fcd/labels.py` |
| Classes | `harvest`: forest → unstocked, or age class ≥3 → ≤1<br>`conversion`: forest → other non-forest<br>`gain`: non-forest/unstocked → forest<br>`no_change`: same forest type, age +0–2<br>`type_shift`, `ambiguous`, persistent non-forest: excluded from training |
| Event year | Update year of the 2025 map. This is the map revision date, not the event date |
| Samples | 15 m inward buffer, area-proportional selection, 10 m pixel de-duplication; 12 of 64 blocks (5 km) held out |
| Features | AE: 75 (64 endpoint differences, endpoint cosine distance, 8 consecutive-year distances, two maxima)<br>S2: NDVI and NBR 2019/2025 and their differences<br>TERR: elevation, slope<br>ATTR: 2019 age class, density, species group |

Mapped areas: no change 76,495.4 ha; harvest/regeneration 1,592.6 ha; conversion 277.6 ha; forest gain 487.2 ha; forest-type change 2,206.6 ha (excluded).

Samples: 6,027 in total (3,000 / 1,500 / 727 / 800), with 4,943 for training and 1,084 held out.

## 2. Results (`outputs/metrics.json`, `outputs/ablation.csv`, `outputs/results/`)

**T1 change detection (F1)**: 5-fold spatial-block CV on the training blocks, and the independent spatial-block hold-out.

| Features | CV F1 | Hold-out F1 |
|---|---|---|
| AE (AlphaEarth) | 0.859 ± 0.093 | 0.933 |
| S2 | 0.773 ± 0.036 | 0.772 |
| TERR | 0.524 ± 0.068 | 0.513 |
| AE+S2 | 0.866 ± 0.085 | 0.936 |
| **AE+S2+TERR (final)** | **0.866 ± 0.086** | **0.936** (precision 0.943, recall 0.930) |

**Controlled comparison** (E1, `outputs/results/E1_fair_comparison.csv`), hold-out F1 with the same samples and folds:

| Feature set | F1 |
|---|---|
| AE endpoints only | 0.926 |
| AE annual 2019–2025 | 0.933 |
| S2 four indices, annual | 0.887 |
| S2 annual + S2-TSFM | 0.896 |

The AE advantage is therefore not mainly due to the longer time window.

**Other tasks**

- **T1b** (2019 forest samples): adding forest attributes raises hold-out F1 from 0.896 to 0.905.
- **T2 change type**: hold-out macro-F1 is 0.777. Class F1 is 0.880 for harvest, 0.829 for gain and 0.621 for conversion. Conversion is hardest, and 36.1% of conversion samples are missed as no change.
- **T3 / timing**: the embedding peak matches the map-update year exactly for 10.7% of samples and within one year for 26.4%. Map-update years record when the map was revised, not when the change happened.
- **Dated records**: against harvest records, the AE peak falls within one year for 77.4% of 31 Chuncheon stands and 78.8% of 33 Inje stands. For 44 conversion parcels, the peak precedes or coincides with the notice year in 90.9%.
- **Hotspots and priorities**:
  - Gi* finds 166 hotspot cells with 267.7 ha of mapped change.
  - Hotspot area varies from 174.8 to 389.8 ha with cell size and distance band.
  - The composite compartment ranking and the area-only ranking agree with Spearman ρ = 0.834, and only 4 compartments appear in both top-10 lists.
- **Inje transfer** without retraining: AE+S2 F1 is 0.823, and 0.861 with a 0.2 threshold.
- **Instant scan threshold**: cosine distance 0.0665, chosen to maximise training-set F1 for forest loss.

## 3. GeoAI agent (Section 3.6)

| Stage | Tool (`fcd/tools.py`) | Checks and output |
|---|---|---|
| Query interpretation | `set_analysis_spec` | Region name → region_id, period and class checks, clarification when ambiguous |
| Data retrieval | `query_change_map`, `query_forest_status`, `query_mgmt_plan`, `list_regions` | Source, reference time, area-total checks |
| Remote-sensing screening | `scan_alphaearth_change` (any year pair, GEE), `run_change_model` (trained model, nominal 95% bounds, agreement with map labels) | Domain and forest-mask warnings |
| Spatial analysis | `hotspot_analysis` | Grid aggregation, Getis–Ord Gi*, BH-FDR 5% |
| Alternative evaluation | `rank_priority_units` | Weighted sum per compartment; ±20% weight Monte Carlo stability |
| Reporting | `make_decision_card` | Markdown card + folium map, **numeric-token grounding**, analysis history, "draft – managerial review" status |

- **Division of work**: the LLM selects tools, interprets results and writes the narrative. All areas, fractions and ranks come from deterministic tools, and `verify.grounding` matches every number in a card against tool outputs.
- **Backends** are selected with `--provider`. Both share the same tools, prompt rules and checks, and API keys are read only from environment variables.
  - Gemini (`fcd/agent_gemini.py`): default model `gemini-3.8-flash` (`GEMINI_MODEL`).
  - Claude (`fcd/agent.py`): default model `claude-opus-5` (`FOREST_AGENT_MODEL`), with server-side fallback (`fallbacks="default"`).
- **Demonstration** (`outputs/cards/`):
  - The first card flagged one self-computed number (53.5) as unmatched.
  - The regenerated card matched all 28 checked numbers.
- **Compartment 060 cross-check**:
  - Map query: 49.6 ha.
  - Instant AlphaEarth scan: 49.2 ha.
  - Trained model: 69.4 ha, with nominal bounds 54.8–84.1 ha and agreement 0.92 with map labels.

## Data availability

Two kinds of data are not distributed:

- **Non-public**: the management plan (stands and records). Files derived from it are also excluded from `release_preprocessed/` and from `../make_release.py`.
- **Internal**: the fire and landslide reports.

Forest type maps, AlphaEarth, Sentinel-2, SRTM and ESA WorldCover are public. See `../docs/DATA.md`.

## Limitations and next steps

- **CAS500-4 imagery**: not yet available. Replace the Sentinel-2 composites (`fcd/gee.s2_summer`, `colab/step01_gee_export_sentinel.py`) and retrain S2-TSFM when the archive grows.
- **Township units**: in `fcd/data.build_regions`, townships are still approximated by dissolving national-forest stands. The township boundaries are now in `data/LSMD_ADM_SECT_UMD_강원특별자치도/` and can replace this approximation.
- **Visual interpretation**: the Inje blind sheet (`outputs/inje_visual_check/`) has been prepared but not yet interpreted.
- **Legal checks**: legal and permit compliance is not judged. The agent marks such questions as requiring managerial confirmation.
