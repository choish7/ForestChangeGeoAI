# Data sources and availability

## Included in this repository (derived, redistributable)

| File | Content | Source of values |
|---|---|---|
| `forest-change-agent/work/samples.parquet` | 6,027 Chuncheon sample points: transition label, 2019 forest attributes, 5 km block, train/hold-out split, lon/lat | Derived from public forest type maps |
| `forest-change-agent/work/features.parquet` | Raw GEE values at the sample points: AlphaEarth 2017–2025 (64 bands × 9 years), Sentinel-2 NDVI/NBR 2019/2025, SRTM elevation/slope | Google Earth Engine public collections |
| `paper/revision/inje/inje_samples.parquet` | 2,100 Inje County transfer-test points | Derived from public forest type maps |
| `paper/revision/inje/visual_check/` | 100-point blind visual-interpretation sheet, answer key, location map, KML, Sentinel-2 chips | Model output + GEE |
| `paper/revision/results/` | Experiment results E1–E10 (tables, JSON, figures) | Scripts in `paper/revision/` |
| Model files (`*.joblib`, `best_model.pt`) | Trained classifiers | This study |
| `final_change_2019_2025/final_change_class.tif`, `final_change_year.tif`, `final_change_patches.gpkg` | Chuncheon change map 2019–2025 | This study |

## Not included — obtain separately

| Data | Where to get it | Used for |
|---|---|---|
| Forest type maps 1:5,000 (Chuncheon 51110, Inje 51810; 2013, 2019, 2024, 2025) | Korea Forest Service, Forest Geospatial Information Service (산림공간정보서비스) | Labels, sampling, hotspots |
| National-forest management plans (7th–9th) and stand polygons | Northern Regional Forest Service on request — **not publicly distributed** | Compartment ranking, dated harvest records (E5b, E10b), decision-support database |
| Conservation forest-land release notices (산지전용 고시) and cadastral map | Gangwon-do public notices, national cadastral map (연속지적도) | E5c, conversion parcels in the web service |
| Wildfire and landslide reports | Korea Forest Service open data | Disaster layers |
| Administrative boundaries (읍면동) | National spatial data infrastructure (LSMD_ADM_SECT_UMD) | Maps, area summaries |
| Large rasters (AlphaEarth int8 stacks, Sentinel-2 54-band composites, probability maps) | Re-export with `scripts/gee_export_*.py` | Decision-support database, maps |

Set `FCD_DATA_DIR` to the folder holding the forest maps (see `forest-change-agent/fcd/config.py` for the expected file names). Per-stand tables derived from the management plan (for example the E5b timing table and the decision-support database) are excluded for the same reason as the plan itself; the paper reports their aggregate results.
