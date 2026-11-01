# Full Experiment Procedure (in paper order)

Reference manuscript: `paper/Choi&Jung_Text_revised.docx`
Compiled: 2026-11-01

This folder (`forest-change-agent/`) lets you check every experiment in the paper, step by step.

- **No originals were deleted or moved.** Code, notebooks, results and models from other folders were gathered here as **copies**.
- The first line of each copied script reads `# [통합 사본] 원본: …` ("consolidated copy; original: …").
- Only I/O paths and names were changed in the copies. The computation is unchanged.
  - Paths: absolute paths became paths relative to this folder. Raw data are read from `data/`.
  - Names: S2-TSFM is used, as in the paper, and Colab files carry step numbers.
- The paper's **S2-TSFM** (Sentinel-2 time-series representation model) was called "Mini-GFM" during development. It was renamed across the project on 2026-11-01.
  - Folders: `mini_gfm/` → `../s2_tsfm/`, `mini_gfm_chuncheon/` → `../s2_tsfm_chuncheon/` (now `models/s2_tsfm_chuncheon/`; the root folder was removed)
  - Source: `mini_gfm_ts.py` → `s2_tsfm.py`
  - Not renamed (Section 15, item 8): past run records, and internal keys and column names in model reports and the database.

---

## 0. Folder layout

```
forest-change-agent/
├─ EXPERIMENTS.md        ← this document
├─ README.md             overview (English)
├─ data/                 raw inputs (forest maps, management plan, stands, conversion notices, cadastre,
│                        boundaries, thinning, fire/landslide, Inje) — see data/README.md
├─ fcd/                  shared modules (label rules, GEE features, spatial analysis, agent tools, numeric grounding)
├─ prep/                 [local] administrative datasets and pixel labels (copies of ../scripts/),
│                        plus build_release_preprocessed.py
├─ colab/                [Colab·GPU·GEE] step-numbered notebooks and their modules
│    01_위성래스터_내보내기.ipynb   (satellite raster export)   step01_gee_export_sentinel.py, step01_gee_export_alphaearth.py, step01_chuncheon_boundary.geojson
│    05_S2-TSFM_사전학습.ipynb      (S2-TSFM pre-training)      step05_s2_tsfm.py
│    08_최종변화지도.ipynb          (final change map)          step08_ae_change_classifier.py, step08_final_change_map.py, step08_conv_label_experiment.py
│    12_AE분류모델_서비스용.ipynb   (AE classifiers for the service)
│    15_인제군_전이시험.ipynb       (Inje transfer test)
├─ scripts/              [local] main experiments 01–13
├─ models/               trained models (copies)
│    ae_class_models/      AE classifiers LC·FT·CH (+ *_local re-trained with the local scikit-learn, FT4 yearly model), report.json
│    s2_tsfm_chuncheon/    S2-TSFM: best_model.pt, change_classifier.joblib, change_scores/prob/class.tif, history.json, report.json
├─ work/                 intermediate files (samples, features, regions; inje/ transfer samples)
├─ outputs/
│  ├─ metrics.json, ablation.csv, test_predictions.csv, change_model.joblib   (scripts/04)
│  ├─ cards/             agent decision cards (scripts/05)
│  ├─ results/           E1–E10 result tables (scripts/06–12)
│  ├─ model_reports/     S2-TSFM, final map, AE classifiers and DB build reports (copies)
│  └─ inje_visual_check/ Inje visual-interpretation sheet (scripts/13)
└─ release_preprocessed/ distributable preprocessing results from ../output/ (non-public layers and columns
                         removed; fire/landslide excluded) — README.md, MANIFEST.csv
```

The decision-support service is a separate application, so it was not copied. Its code is in `../forest_decision_agent/` (Section 12), and its README gives the full run order (A–G) linking it to this folder. It reads from this folder:
- the models in `models/`
- the helper modules in `prep/` (`combine_thinning_forest_type`, `validate_yearly_ae_s2`)
- the forest roads in `data/춘천_임도_도로.gpkg`

It shares the derived products in `../output/` and `../final_change_2019_2025/`.

---

## 1. Execution order

| Step | Paper section | Environment | Code | Main outputs |
|---|---|---|---|---|
| **1** Satellite raster export | 2.3 | Colab+GEE | `colab/01_위성래스터_내보내기.ipynb` | `output/Chuncheon_AE_{2017..2025}_64Bands_int8.tif`, `Chuncheon_AE_consec_dist_2017_2025.tif`, `Chuncheon_S2_2019_2025_42Bands.tif`, `Chuncheon_S2_2017_2025_54Bands.tif` |
| **2** Administrative datasets | 2.2 | local | `prep/join_management_results.py` → `build_yearly_projects.py` (+ `clip_thinning_*`, `combine_thinning_*`), `build_mountain_conversion_dataset.py` → `build_conversion_labels.py`, `build_fire_landslide.py` | management records joined to stands, conversion-notice parcels, fire/landslide points |
| **3** Map transition labels and samples | 3.1 | local | `scripts/01_build_change_labels.py` → `02_sample_points.py` | `work/change_polygons.parquet`, `samples.parquet` |
| **4** Sample-point features | 2.3, 3.2 | local+GEE | `scripts/03_extract_gee_features.py` | `work/features.parquet` |
| **5** S2-TSFM pre-training | 2.3 | Colab GPU | `prep/make_s2_label_raster.py` → `colab/05_S2-TSFM_사전학습.ipynb` | `models/s2_tsfm_chuncheon/` (`best_model.pt`, `change_scores.tif`) |
| **6** Classification T1·T1b·T2·T3 | 3.3 | local | `scripts/04_train_evaluate.py` | `outputs/metrics.json`, `ablation.csv` |
| **7** Controlled comparison etc. E1–E6 | 3.2, 4.2 | local | `scripts/06_controlled_comparison.py` | `outputs/results/E1_*.csv`, `E3_*.csv`, `run.log` |
| **8** Final change map | 3.7 | local→Colab | `prep/build_change_labels_2013_2019_2025.py` → `compare_mgmt_vs_forestmap_labels.py` → `build_training_label_rasters.py` → `build_forest_mask.py` → `colab/08_최종변화지도.ipynb` | `../final_change_2019_2025/`, `../compare_runs/` |
| **9** Hotspots + sensitivity | 3.4 | local | tool `hotspot_analysis` (`fcd/spatial.py`), `scripts/08_hotspot_sensitivity.py` | `outputs/results/E9_*.json` |
| **10** Compartment priorities | 3.5 | local | tool `rank_priority_units`, `scripts/09_priority_vs_area.py` | `outputs/results/E7_ranking.json` |
| **11** GeoAI agent demonstration | 3.6, 4.6 | local+LLM | `scripts/05_run_agent.py` | `outputs/cards/` |
| **12** Decision-support service | 3.7, 4.7 | local+Colab | `../forest_decision_agent/build_db.py` → `build_trajectory.py` → `prep/build_ae_class_samples.py` → `colab/12_AE분류모델_서비스용.ipynb` → `build_ae_classes.py` → `prep/build_yearly_labels.py` → `save_yearly_validation_set.py` → `validate_yearly_ae_s2.py` → `build_yearly_history.py` → `web.py` | `../output/의사결정DB/`, `models/ae_class_models/` |
| **13** Timing vs dated events | 3.8, 4.3 | local | `scripts/07_event_timing.py` (needs the DB from step 12) | `outputs/results/E5b_*`, `E5c_*`, `timing_summary.json` |
| **14** Service report checks | 4.6 | local | `scripts/10_report_checks.py` | `outputs/results/E8_report_checks.csv` |
| **15** Inje transfer test | 3.8, 4.8 | local→GEE→local | `scripts/11_inje_prepare.py` → upload to GEE asset → `scripts/12_inje_transfer.py` (= `colab/15_인제군_전이시험.ipynb`) → `scripts/13_inje_visual_check.py` | `work/inje/`, `outputs/results/E10_*.json`, `outputs/inje_visual_check/` |

**Dependencies**
- Step 13 (timing) needs the evidence DB from step 12 and the final change map from step 8.
- Step 7 reads two inputs from other steps:
  - S2-TSFM features from `models/s2_tsfm_chuncheon/change_scores.tif` (step 5).
  - Distances for the excluded categories (E6) from the AE rasters (step 1) and the final map (step 8).

---

## 2. Satellite raster export (step 1)

`colab/01_위성래스터_내보내기.ipynb`

Code-cell titles carry `[01-k]` numbers; the cell column below refers to them.

| Cells | Content | Recorded result |
|---|---|---|
| boundary, auth | Write the Chuncheon boundary `step01_chuncheon_boundary.geojson`; initialise Earth Engine | — |
| 01-1–01-2 | Sentinel-2 2019–2025 annual June–September medians. Six bands per year (NDVI, NDWI, NDBI, NDMI, slope, elevation), 42 bands in total (`step01_gee_export_sentinel.py`) | 97 scenes in 2025; export completed (2:44:04) |
| 01-3–01-6 | AlphaEarth V1 2017–2025: nine yearly 64-band int8 (×127) files + one 8-band consecutive-year cosine-distance file (`step01_gee_export_alphaearth.py`) | 100% valid pixels, mean norm 1.0001, about 6.8 GB |
| 01-7–01-9 | Sentinel-2 2017–2025, 54 bands (for the pixel-level comparison) | completed (0:29:22) |

---

## 3. Administrative datasets (step 2, Section 2.2, Table 1)

| Code (`prep/`) | Content | Output (`../output/`) |
|---|---|---|
| `join_management_results.py` | Join the 7th–9th plan management records to the Chuncheon (9th plan) stand map | `춘천_임소반_산림사업실적.gpkg` |
| `clip_thinning_to_chuncheon.py`, `combine_thinning_forest_type.py` | Clip the thinning dataset to Chuncheon and join forest-map attributes and records | `춘천시_숲가꾸기_임상도_사업실적_결합.gpkg` |
| `build_yearly_projects.py` | Yearly forest-project layer | `춘천시_연도별_산림사업.gpkg` |
| `build_mountain_conversion_dataset.py` | Join conservation forest-land release notices (2017–2024) to the cadastral map | `산지전용/춘천시_산지전용_보전산지해제.gpkg` (549 parcel records; 516 labelled parcels in `산지전용_라벨.gpkg`) |
| `build_conversion_labels.py` | Conversion label rasters and label checks | `산지전용/Chuncheon_convlabels_AE_10m.tif` |
| `build_fire_landslide.py` | Fire and landslide reports inside Chuncheon as points | `춘천시_산불_산사태.gpkg` (internal data, not distributed) |

---

## 4. Map transition labels and samples (step 3, Section 3.1, Table 3, Figure 2)

**Method**
- The 2019 and 2025 forest type maps were overlaid in EPSG:5179, and fragments < 0.1 ha were removed.
- Rules in `fcd/labels.py`:
  - No change: forest at both endpoints, same forest type, age class +0 to +2
  - Harvest/regeneration: forest → unstocked, or age class ≥3 → ≤1
  - Conversion: forest → other non-forest
  - Forest gain: non-forest or unstocked → forest
  - Excluded from training: forest-type change, unresolved, persistent non-forest
- Sampling targets were 3,000 / 1,500 / 800 / 800. Polygons were drawn in proportion to area, points were placed inside a 15 m inward buffer, and duplicates within a 10 m pixel were removed.
- 12 of the 64 5-km blocks were assigned to the hold-out.

| Item | Value |
|---|---|
| Overlay fragments / area | 47,750 / 82,391.3 ha |
| No change / harvest / conversion / gain | 76,495.4 / 1,592.6 / 277.6 / 487.2 ha |
| Forest-type change / unresolved / persistent non-forest (excluded) | 2,206.6 / 995.0 / 337.0 ha |
| Samples | 6,027 (3,000 / 1,500 / **727** / 800). Only 727 conversion points remained after buffering |
| Training / hold-out | 4,943 / 1,084 |
| Changed samples with a valid update year | 3,008, of which 2,095 (69.6%) updated in 2025 |

---

## 5. Sample-point features (step 4, Sections 2.3 and 3.2)

`scripts/03_extract_gee_features.py` extracts the features in GEE at 10 m; 6,027 points take less than a minute.

| Group | Count | Composition |
|---|---|---|
| AE | 75 | 64 differences (2025 − 2019), 1 endpoint cosine distance, 8 consecutive-year distances (2018–2025), maximum over 2020–2025, maximum before 2019 |
| S2 | 6 | NDVI and NBR for 2019 and 2025 and their differences (June–September medians, SCL mask) |
| TERR | 2 | SRTM elevation and slope |
| ATTR | 3 | 2019 age class, density, species group (T1b only) |
| S2-TSFM | 13 | Embedding distances and index differences from the step-5 model (`scripts/06` samples `change_scores.tif` at the points) |

---

## 6. S2-TSFM pre-training (step 5, Section 2.3)

`prep/make_s2_label_raster.py` builds evaluation labels on the 42-band grid. `colab/05_S2-TSFM_사전학습.ipynb` then trains the model. The notebook module is `step05_s2_tsfm.py`, a copy of `../s2_tsfm/s2_tsfm.py`.

- 05-1: Copy the input TIF and labels from Drive.
- 05-2: Quick check (1 epoch × 5 steps).
- 05-3: Full training (30 epochs × 100 steps, Tesla T4, 15.1 min). The run is saved to `GEE_Export/s2_tsfm_chuncheon`, and the trained model is in `models/s2_tsfm_chuncheon/`.
  - Model: a convolutional encoder shared across years (24 dimensions per year) and a masked-reconstruction decoder.
  - Training: 35% of 16-px blocks masked, whole-year masking with probability 0.15, L1 loss, 128-px patches, 512-px spatial validation blocks.
- **Result**: validation MAE **0.0747** (0.075 in the manuscript). See `models/s2_tsfm_chuncheon/history.json` and `report.json`.
  - Unsupervised change-score AUROC: `emb_first_last` 0.869, `|ΔNDVI|` 0.801.

---

## 7. Classification T1·T1b·T2·T3 (step 6, Section 3.3, Tables 5 and 7, Figures 3–5)

`scripts/04_train_evaluate.py`

- Classifier: HistGradientBoosting (300 iterations, learning rate 0.06, 31 leaves, L2 1.0, class_weight balanced)
- Validation: StratifiedGroupKFold(5) on the training blocks, and the independent block hold-out

### T1 change detection (Table 5)
| Features | Count | CV F1 | Hold-out F1 |
|---|---|---|---|
| AE | 75 | 0.859 ± 0.093 | **0.933** |
| S2 | 6 | 0.773 ± 0.036 | 0.772 |
| TERR | 2 | 0.524 ± 0.068 | 0.513 |
| AE+S2 | 81 | 0.866 ± 0.085 | 0.936 |
| AE+TERR | 77 | 0.866 ± 0.082 | 0.935 |
| S2+TERR | 8 | 0.785 ± 0.044 | 0.758 |
| AE+S2+TERR | 83 | 0.866 ± 0.086 | 0.936 |

- Final AE+S2+TERR model: P 0.943, R 0.930, BA 0.938. The confusion matrix is TN 523, FP 30, FN 37, TP 494.
- T1b (samples forested in 2019): adding ATTR raises F1 from 0.896 to 0.905, and CV from 0.838 ± 0.127 to 0.845 ± 0.118.
- Instant-scan threshold: cosine distance 0.0665 (training-set F1 0.819).

### T2 change type (531 hold-out points)
- macro-F1: CV 0.742 ± 0.065; hold-out **0.777**
- Class F1: harvest 0.880, conversion 0.621, gain 0.829
- Confusion matrix (rows = true harvest / conversion / gain): [272,17,43] / [14,36,11] / [0,2,136]

### T3 peak year vs map-update year (3,008 points)
- Exact 10.7%, within one year 26.4%

---

## 8. Controlled comparison and diagnostics E1–E6 (step 7, Sections 3.2 and 4.2, Tables 6–7)

`scripts/06_controlled_comparison.py`. Samples, blocks and classifier settings are the same as in Section 7.

### E1 feature source vs time window (Table 6; hold-out CI from 1,000 block resamples)
| Features | Window | Count | CV F1 | Hold-out F1 [95% CI] | Type macro-F1 |
|---|---|---|---|---|---|
| AE endpoints | 2019, 2025 | 65 | 0.861 ± 0.090 | 0.926 [0.894–0.956] | 0.776 |
| AE annual | 2019–2025 | 72 | 0.863 ± 0.096 | 0.933 [0.911–0.960] | 0.777 |
| AE original set | 2017–2025 | 75 | 0.859 ± 0.093 | 0.933 [0.907–0.962] | 0.795 |
| S2 NDVI·NBR endpoints | 2019, 2025 | 6 | 0.773 ± 0.036 | 0.772 [0.682–0.828] | 0.636 |
| S2 four indices, endpoints | 2019, 2025 | 12 | 0.799 ± 0.016 | 0.811 [0.777–0.843] | 0.667 |
| S2 four indices, annual | 2019–2025 | 44 | 0.846 ± 0.058 | 0.887 [0.846–0.932] | 0.708 |
| S2 annual + S2-TSFM | 2019–2025 | 57 | 0.851 ± 0.052 | 0.896 [0.858–0.933] | 0.712 |
| AE annual + S2 annual | 2019–2025 | 116 | 0.873 ± 0.081 | 0.931 [0.903–0.968] | 0.763 |

### E2 polygons shared by training and hold-out
- 20 shared polygons, contributing 44 hold-out points.
- Excluding those points changes the AE+S2+TERR hold-out F1 from 0.936 to 0.937.

### E3 fold composition
| Set | Conversion share | AE F1 |
|---|---|---|
| Folds 1–5 | 0.175 / 0.065 / 0.082 / **0.277** / 0.066 | 0.823 / 0.943 / 0.914 / **0.692** / 0.924 |
| Hold-out | 0.056 | 0.933 |

### E4 missed changes by type (Table 7)
| Type | Samples | Missed | Miss rate |
|---|---|---|---|
| Harvest/regeneration | 332 | 4 | 1.2% |
| Conversion | 61 | 22 | 36.1% |
| Forest gain | 138 | 11 | 8.0% |
| Total | 531 | 37 | 7.0% |

### E5a by map-update year
| Group | Exact | Within 1 yr | Peak ≤ update |
|---|---|---|---|
| Updated 2025 | 9.5% | 19.7% | 100% |
| Updated 2020–2024 | 13.4% | 41.7% | 72.6% |

### E6 AE distance medians of excluded categories (800 points each)
| Category | Median distance |
|---|---|
| Forest-type change | 0.033 |
| Unresolved | 0.034 |
| No change | 0.032 |
| Harvest/regeneration | 0.199 |

These values support the exclusion argued in Section 3.1.

---

## 9. Final change map (step 8; input to the service and the timing check)

**Label rasters (`prep/`)**
1. `build_change_labels_2013_2019_2025.py`: the same rules applied to the 2013, 2019 and 2025 maps.
2. `compare_mgmt_vs_forestmap_labels.py`: management records compared with the forest maps to form consensus labels.
3. `build_training_label_rasters.py`: `Chuncheon_labels_AE_10m.tif` (10 m) and `_S2_20m.tif` (20 m).
4. `build_forest_mask.py`: forest-land mask from the 2019 or 2025 map.

**Colab: `colab/08_최종변화지도.ipynb`**

- 08-0: Mount Drive and write `step05_s2_tsfm.py`.
- 08-1–08-3: Upload the labels. The AE grid origin was offset by one pixel, so the labels were cropped to match.
- 08-4–08-6: Compare S2-TSFM (54 bands) and AE with the same labels and the same 512-px hold-out (`outputs/model_reports/compare_*_report.json`).

| Features | Count | Balanced acc | Macro F1 | Change AUROC |
|---|---|---|---|---|
| AE | 202 | 0.693 | 0.578 | 0.991 |
| AE+S2 | 250 | 0.720 | 0.585 | 0.992 |
| S2 indices (matched to 10 m) | 48 | 0.683 | 0.518 | 0.976 |

Remaining cells of the same notebook:

| Cells | Content | Result |
|---|---|---|
| 08-7 | Comparison table | — |
| 08-8–08-9 | Diagnose and remove zero-vector pixels outside Chuncheon that had been classified as harvest | Harvest area 104,385 → 4,664 ha; the fix is at line 71 of `step08_ae_change_classifier.py` |
| 08-10–08-11 | Apply the forest mask | — |
| 08-12 | Compare minimum mapping units of 0.1 / 0.2 / 0.5 ha | — |
| 08-13 | `step08_final_change_map.py`: AE+S2, forest mask, MMU 0.5 ha for harvest and gain, 0.2 ha for conversion | `../final_change_2019_2025/` (58.2 min); hold-out macro-F1 0.606, harvest F1 0.84 |
| 08-14–08-15 | `step08_conv_label_experiment.py`: validate the map with notice labels, then retrain with augmented labels | Conversion F1 0.163 → 0.189 (`outputs/model_reports/conv_label_experiment.json`) |

---

## 10. Hotspots and compartment priorities (steps 9–10, Sections 3.4, 3.5, 4.4, 4.5, Table 8, Figure 8)

**Hotspots**: harvest/regeneration and conversion polygons in national forests were aggregated to 500 m cells by area. Gi* was computed with a 1,500 m binary band and BH-FDR q = 0.05.
- National-forest change totals 500.09 ha (1.82% of the 2019 forest): harvest 479 patches, 370.05 ha; conversion 420 patches, 130.04 ha.
- Of 1,935 cells, 166 (8.6%) are hotspots containing 267.7 ha (53.5%). There are no coldspots.

`scripts/08_hotspot_sensitivity.py` results:

| Cell / band | Hotspot cells | Area (ha) |
|---|---|---|
| 250 / 750 m | 556 | 389.8 |
| 500 / 1,000 m | 131 | 328.6 |
| **500 / 1,500 m** | **166** | **267.7** |
| 500 / 2,000 m | 203 | 260.3 |
| 1,000 / 3,000 m | 43 | 174.8 |

**Priorities**
- The 112 compartments were ranked by a weighted sum of min–max-normalised criteria: conversion 0.4, harvest 0.3, change fraction 0.2, growing stock 0.1.
- For the sensitivity analysis, each weight was multiplied by a U(0.8, 1.2) factor, 500 times.
- Results
  - Ranking: 033, 002, 060, 061, 065, 007, 096, 066, 059, 039
  - The top 7 stayed in the top 10 in all 500 trials; 066 in 98%, 059 in 96% and 039 in 64%.
- `scripts/09_priority_vs_area.py` compared this ranking with an area-only ranking.
  - Spearman ρ = 0.834, with 4 compartments common to both top 10s.
  - The compartments found only by the composite ranking (002, 061, 096, 066, 007, 059) are all conversion-dominated.

---

## 11. GeoAI agent demonstration (step 11, Sections 3.6 and 4.6, Tables 4 and 9)

Run `scripts/05_run_agent.py`. The backends are `fcd/agent_gemini.py` (default gemini-3.8-flash) and `fcd/agent.py` (Claude). Numeric grounding is done by `fcd/verify.py`.

- Call order: `set_analysis_spec` → `query_change_map` → `query_forest_status` → `hotspot_analysis` → `rank_priority_units` → `make_decision_card`
- `outputs/cards/card_20260926_214125.md`: 97% of 29 numbers matched. The self-computed "53.5" was unmatched.
- `outputs/cards/card_20260926_214131.md`: regenerated card; **all 28 numbers matched (R = 1.00)**.
- Compartment 060 (Table 9)

| Source | Area |
|---|---|
| Forest-map query | 49.6 ha |
| Instant embedding scan | 49.2 ha |
| Classifier | 69.4 ha (nominal 54.8–84.1 ha, agreement 0.92) |

---

## 12. Integrated decision-support service (steps 12 and 14, Sections 3.7, 4.6, 4.7, Table 10, Figure 9)

The service is in `../forest_decision_agent/`. Build order:

| # | Code | Content |
|---|---|---|
| 1 | `build_db.py` | 1,136 national-forest stands and 516 conversion parcels as units; AE and S2 change signals, management records, disasters, terrain and roads; five-way judgement |
| 2 | `build_trajectory.py` | Yearly trajectories per stand, change pathway and interpretation sentences, vectors for similar-embedding search |
| 3 | `prep/build_ae_class_samples.py` → `colab/12_AE분류모델_서비스용.ipynb` → `build_ae_classes.py` | Train and apply the three AE classifiers (`models/ae_class_models/`; table below) |
| 4 | `prep/build_yearly_labels.py` → `save_yearly_validation_set.py` → `validate_yearly_ae_s2.py` → `build_yearly_history.py` | Yearly forest-type history and re-survey candidates |
| 5 | `python -m forest_decision_agent.web` | FastAPI + Leaflet web app (:8800). Claude mode (claude-opus-5-5) and rule-based mode; Korean/English UI (`translate.py`) |

Results of `colab/12_AE분류모델_서비스용.ipynb` (`models/ae_class_models/report.json`):

| Task | Features | Macro-F1 | Note |
|---|---|---|---|
| LC land cover | 256 | 0.622 | grassland F1 0.11 |
| FT forest type | 256 | 0.646 | single-year 0.642 |
| CH change type (no conversion) | 204 | 0.913 | 0.913 with endpoints only |

**Report checks**: `scripts/10_report_checks.py` → `outputs/results/E8_report_checks.csv`

| Mode | Reports | Numeric tokens | Matched | Ranking rows |
|---|---|---|---|---|
| Claude agent | 2 | 250 | 100% | 20/20 |
| Rule-based | 7 | 385 | 100% | 50/50 |
| Scripted demo | 5 | 338 | 100% | 40/40 |
| **Total** | **14** | **973** | **100%** | **110/110** |

**Re-run check (2026-11-01)**: re-running `scripts/10` now checks all 32 runs in `forest_decision_agent/outputs/`, including those after 31 October. The result is saved as `E8_report_checks_rerun_20261101.csv`.
- Restricted to the paper's 14 reports (run on 2026-10-30), the token total is 973, the same as the original.
- `E8_report_checks.csv` is kept as the original result used in the paper.

---

## 13. Timing vs dated events (step 13, Sections 3.8 and 4.3, Figure 6a–b)

Run `scripts/07_event_timing.py`. Stand time series come from the Section 12 database, and final-map years from the Section 9 map.

| Events | n | AE peak | S2 max NDVI drop | Final-map year |
|---|---|---|---|---|
| Harvest/regeneration records 2020–2024 (exact / ±1 yr) | 31 stands | 29.0% / **77.4%** | 35.5% / 71.0% | 30.0% / 73.3% (n=30) |
| ├ final felling | 15 | 13.3% / 66.7% (mostly one year later) | | |
| └ site preparation · artificial regeneration | 16 | 43.8% / 87.5% | | |
| Conversion notices (peak ≤ notice / 0–2 yr before) | 44 parcels | 90.9% / 59.1% | 93.2% / 61.4% | |

---

## 14. Inje County transfer test (step 15, Sections 3.8 and 4.8, Tables 11–12, Figure 6c)

1. `scripts/11_inje_prepare.py` applies the same rules to the Inje (51810) maps of 2019 and 2025.
   - Samples: 1,000 no change / 500 harvest / 300 conversion / 300 gain. Outputs go to `work/inje/`.
   - `inje_change_polygons.parquet` (296 MB) was not copied; this step regenerates it.
2. Upload the samples to the GEE asset `projects/ee-choish7838/assets/inje_transfer`.
3. Run `scripts/12_inje_transfer.py`. The original run used `colab/15_인제군_전이시험.ipynb` on 2026-10-31.
   - Features for both counties are extracted with the same code.
   - The Chuncheon models are applied to Inje without retraining and compared with an Inje-only 5-fold block CV.
   - The result is `outputs/results/E10_inje_transfer.json`.
4. `scripts/13_inje_visual_check.py` builds a blind 100-point sheet with 25 points per predicted class (`outputs/inje_visual_check/`). **It has not yet been interpreted.**

| Features | Chuncheon hold-out* | Inje transfer F1 | P | R | Threshold 0.2 | Inje-only CV |
|---|---|---|---|---|---|---|
| AE | 0.933 | 0.778 | 0.955 | 0.656 | 0.825 | 0.792 ± 0.119 |
| S2 | 0.764 | 0.755 | 0.828 | 0.694 | — | 0.727 ± 0.062 |
| TERR | 0.487 | 0.378 | 0.572 | 0.282 | — | 0.474 ± 0.059 |
| **AE+S2** | 0.936 | **0.823** | 0.960 | 0.721 | **0.861** | 0.847 ± 0.082 |
| AE+S2+TERR | 0.934 | 0.760 | 0.969 | 0.625 | 0.800 | 0.843 ± 0.089 |

\* Recomputed from re-extracted features. S2 is 0.764 rather than the 0.772 in Table 5 because the GEE composites were regenerated.

- AE+S2+TERR miss rates: harvest 4.6%, conversion 65.0%, gain 64.7%.
- AE distance medians (Chuncheon / Inje): harvest 0.211 / 0.206, conversion 0.042 / 0.039, gain 0.080 / 0.083, no change 0.031 / 0.025.
- Transferred T2 macro-F1 is 0.776, against 0.785 for the re-extracted Chuncheon hold-out and 0.686 ± 0.130 for the Inje-only CV.
- In 33 Inje stands with records, the AE peak fell within one year in 78.8%. In 18 of 22 final-felling stands it came one year after the record.

---

## 15. Notes

1. **Error in `run.log`**: `outputs/results/run.log` (the original run record) ends with `StopIteration` in the E7 stage.
   - Cause: the management-plan stand path in `fcd/config.py` (`../data/북부지방산림청/…`) did not exist at the time.
   - E7 was then produced separately by `scripts/09`, which set the path explicitly.
   - `fcd/config.py` now points to `data/춘천(9)*/*.shp`, so the per-script overrides in 06, 08 and 09 were removed.
   - Check: re-running `scripts/08` reproduced the paper result (`E9_hotspot_sensitivity.json`) exactly.
2. **Raw-data path**: `DATA` in `fcd/config.py` defaults to `forest-change-agent/data/`.
   - Override it with the `FCD_DATA_DIR` environment variable.
   - `prep/` and `scripts/11` and `13` also read from `data/`.
   - Every raw-data path referenced by the scripts was checked to exist.
3. **Notebook origins**: the `colab/` notebooks re-split the root `scratchpad (4)–(7).ipynb` by experiment step.

   | Notebook | Source |
   |---|---|
   | 01 | (4) cells 0–10, 16–18 |
   | 05 | (4) cells 11–15 |
   | 08 | (4) cells 19–26 + all of (5) |
   | 12 | (6) |
   | 15 | (7) |

   - Cell outputs are the original run records, so their text still shows the old names (Mini-GFM, `mini_gfm_ts.py`).
   - Cell 27 of (4) was a comparison table that was never executed, so it was dropped; an executed version of the same table is cell 0 of (5).
   - `scratchpad (3).ipynb` has the same code as (4) with fewer outputs, so it was excluded.
   - `scripts/gee_export_sentinel_chuncheon.ipynb` is an early version of the first part of notebook 01, so it was excluded.
4. **Module versions**: `colab/step*.py` are the latest repository versions, with only file and import names changed to step numbers. The `%%writefile` cells in the notebooks contain the same code.
   - Compared with the original run, `step08_ae_change_classifier.py` includes the zero-vector fix from cell 08-9.
   - `step08_conv_label_experiment.py` adds one `s2_flat` line and a usage comment.
5. **Non-public data**: `outputs/results/E5b_harvest_timing.csv` is a stand-level table derived from the non-public management records. It is excluded from the public release (`../make_release.py`).
6. **Release scope**: `../make_release.py` includes `forest-change-agent/scripts/*.py`, so the new scripts 06–13 are released.
   - `prep/`, `colab/`, `models/`, `outputs/results/` and `data/` are not in the INCLUDE list.
   - `make_release.py` releases the models from `models/` (`ae_class_models/*_local.joblib`, `s2_tsfm_chuncheon/`).
   - `data/` holds non-public management-plan data and must not be released.
   - `../release/` predates the renaming; rebuild it with `python make_release.py`.
7. **Overwriting**: `scripts/06–13` write to `outputs/results/` and `work/inje/`.
   - The files there now are copies of the original paper results (originals in `../paper/revision/results/` and `../paper/revision/inje/`).
   - Re-running overwrites these copies. The originals remain.
8. **Scope of the S2-TSFM renaming (2026-11-01)**
   - Renamed:
     - folders, sources and imports
     - display names in code and documents, including the service UI
     - the E1 table row `S2_연간+S2-TSFM 임베딩거리`
     - the model key in the decision DB `build_meta.json`
   - Not renamed:
     - past run records: `../forest_decision_agent/outputs/*`, `run.log`, `../compare_runs/*.csv`, root `scratchpad*.ipynb`
     - internal keys in model reports (`지수+GFM`, `GFM emb_*`)
     - evidence-DB column names (`MG_변화면적_ha`, etc.). Renaming them requires rebuilding the DB.
   - Files as they were before the renaming were backed up to `backup_before_tsfm_rename.zip` in the session scratch folder.
9. **Models (2026-11-01)**: `models/` is the only location of the trained models. The root folders `../ae_class_models/` and `../s2_tsfm_chuncheon/` were removed after their contents were confirmed identical.
   - `scripts/06`, the decision-support service (`build_db`, `build_ae_classes`, `build_yearly_history`, `tools`) and `../make_release.py` read the models from `models/`.
   - The root `../scripts/` and `../paper/revision/` scripts were repointed to `models/` and `data/` as well. The root copies of the raw data and the partial copy `../revision/` were also removed.
   - Save retrained models to `models/`.
10. **Forest roads (2026-11-01)**: `build_db.py` previously read roads from a file outside the project (`C:/src/ai_test/공간데이터/춘천_산림공간정보.gpkg`, 1.09 GB). The two layers it uses (`임도` with 119 features, `도로_1000` with 30,451 features) were extracted to `data/춘천_임도_도로.gpkg` (9.9 MB), and the service now reads that file.

## Appendix: manuscript numbers vs result files

| Manuscript | Result file (relative to this folder) | Status |
|---|---|---|
| Tables 3, 5, 7; Figure 4 | `outputs/metrics.json`, `ablation.csv`, `outputs/results/run.log` | match |
| Table 6, fold composition, shared polygons | `outputs/results/E1_fair_comparison.csv`, `E3_fold_composition.csv`, `run.log` | match |
| 3.1 distances of excluded categories | `run.log` E6 | match |
| S2-TSFM validation MAE 0.075 | `models/s2_tsfm_chuncheon/history.json` (0.0747) | match (rounded) |
| 4.3 timing, Figure 6 | `outputs/results/timing_summary.json`, `E10_inje_transfer.json` | match |
| 4.4 hotspots and sensitivity | `outputs/results/E9_hotspot_sensitivity.json` | match |
| 4.5 ranking comparison | `outputs/results/E7_ranking.json` | match |
| 4.6 card with 28 tokens | `outputs/cards/card_20260926_214131.md` | match |
| Table 10 | `outputs/results/E8_report_checks.csv` | match |
| Table 11, 4.8 | `outputs/results/E10_inje_transfer.json` | match |
| Table 9 (49.2 / 69.4 ha) | demonstration card | not checked against a separate result file |
