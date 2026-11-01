# 전처리 결과 (배포용)

`C:\src\ForestChangeGeoAI\output\`에 있는 전처리 결과 가운데 배포할 수 있는 파일만 모았다(2026-11-01).

- 다시 만들기: `python forest-change-agent/prep/build_release_preprocessed.py` (이 README는 그대로 남는다)
- 파일별 원본, 처리 내용, SHA-256: `MANIFEST.csv`
- 좌표계: 벡터는 EPSG:5179이고 `chuncheon_boundary.geojson`만 EPSG:4326이다. 래스터는 EPSG:32652(10 m / 20 m)다.

## 배포 원칙

산불·산사태 신고 정보는 내부 자료라 넣지 않았다.

북부지방산림청 경영계획부(임소반도·산림사업실적)는 비공개 자료다(논문 Data Availability, `docs/DATA.md`).
- 이 자료에서 나온 레이어·열·밴드는 **지웠다**.
  - 열: `국유림_임반`, `국유림_소반`, `국유림_실적_*` 등
  - 레이어: `실적_검증`
  - 밴드: `MGMT`, `MGMT_YEAR`
- 주로 이 자료로 만든 파일은 **넣지 않았다**(아래 "제외한 파일" 참고).
- 그 밖의 레이어와 값은 원본과 같다.

## 파일

| 폴더 | 파일 | 내용 | 생성 코드 (`prep/`) | 실험 순서 |
|---|---|---|---|---|
| `01_경계` | `춘천시경계.shp`, `춘천시_읍면동경계.shp`, `chuncheon_boundary.geojson` | 춘천시·읍면동 경계 (LSMD 읍면동 병합). geojson은 GEE 내보내기용(4326) | `clip_thinning_to_chuncheon.py` | 1·2 |
| `02_임상도_변화라벨` | `춘천시_산림변화라벨_2013_2019_2025.gpkg` | 임상도 3시기 중첩 조각 62,743개. P1(2013→2019)·P2(2019→2025) 변화유형·확신도·연도·궤적을 담는다. 이 밖에 `도화범위차이`, `구간별_요약`, `궤적_요약` 레이어가 있다 | `build_change_labels_2013_2019_2025.py` | 8 |
| | `산림변화라벨_구간별_요약.csv`, `산림변화라벨_궤적_요약.csv` | 위 라벨의 면적 요약 | 〃 | 8 |
| | `ae_class_samples.npz` | AE 분류모델 3종(LC·FT·CH)용 표본 60,698개: 과업, 연도, 라벨, 조각, AE 격자 행·열, 검증블록 | `build_ae_class_samples.py` | 12 |
| `03_래스터_마스크_라벨` | `Chuncheon_forestmask_AE_10m.tif`, `_S2_20m.tif` | 산림 마스크 (2019 또는 2025 임지 = 1) | `build_forest_mask.py` | 8 |
| | `Chuncheon_S2_2019_2025_labels_FC.tif` | 42밴드 S2 격자에 맞춘 임상도 변화 라벨 (`FC_TYPE`, `FC_YEAR`). S2-TSFM 평가용 | `make_s2_label_raster.py` | 5 |
| `04_산지전용` | `춘천시_산지전용_보전산지해제.gpkg`·`.csv` | 보전산지 해제 고시(2017~2024)를 연속지적도와 결합한 필지 549건 | `build_mountain_conversion_dataset.py` | 2 |
| | `산지전용_라벨.gpkg` | 전용 라벨 필지 516건: 전용확정, 후보, 전용이전, 고시연도, S2 지수 변화 | `build_conversion_labels.py` | 2·13 |
| | `Chuncheon_convlabels_AE_10m.tif`, `_S2_20m.tif` | 전용 라벨 래스터 (`CONV_LABEL`, `CONV_YEAR_MAX`) | 〃 | 8 |
| | `산지전용_라벨_테스트.csv`·`.json` | 라벨 점검: 지수별 AUROC·AP, 필지 탐지율 | 〃 | 2 |
| | `춘천시_법정동코드.csv` | 춘천시 법정동코드 (51110 체계, PNU 결합용) | `build_mountain_conversion_dataset.py` | 2 |
| `05_숲가꾸기` | `춘천시_산림경영활동_숲가꾸기공간정보.gpkg` | 산림청 숲가꾸기공간정보(2017 배포) 가운데 춘천시 사업지 482개와 경계 | `clip_thinning_to_chuncheon.py` | 2 |
| | `춘천시_숲가꾸기_임상도_사업실적_결합.gpkg` | 숲가꾸기 사업지에 임상도 2019·2025 속성과 변화를 결합한 것. 국유림 실적 열과 임소반 레이어는 뺐다(파일 이름은 원본 그대로) | `combine_thinning_forest_type.py` | 2 |

## 제외한 파일

| 구분 | 파일 (`output/`) | 이유 |
|---|---|---|
| 내부 자료 | `춘천시_산불_산사태.gpkg`, `춘천시_산불.csv`, `춘천시_산사태.csv` | 산불·산사태 신고 정보는 내부 자료다 |
| 비공개 자료 유래 | `춘천_임소반_산림사업실적.gpkg`, `미결합실적.csv`, `결합현황.csv` | 경영계획부 사업실적과 임소반도 그 자체 |
| | `춘천_경영실적_임상도_변화라벨_비교.gpkg`, `경영실적_임상도_비교_*.csv`, `산림변화라벨_실적_검증.csv` | 소반 단위로 실적과 임상도를 비교한 표 |
| | `Chuncheon_labels_AE_10m.tif`, `Chuncheon_labels_S2_20m.tif` | 실적과 임상도의 합의 라벨(학습 라벨) |
| | `춘천시_연도별_산림사업.gpkg`, `춘천시_연도별_산림사업_건수.csv` | 국유림 사업실적을 연도별로 정리한 것 |
| | `연도별라벨/검증용/*` | 등급 C가 국유림 실적으로 정해지고, 국유림소반 열이 있음 |
| 위성 래스터 (11 GB) | `Chuncheon_AE_*_64Bands_int8.tif`, `Chuncheon_AE_consec_dist_2017_2025.tif`, `Chuncheon_S2_*Bands.tif` | 공개 자료이고 `colab/01_위성래스터_내보내기.ipynb`로 다시 만들 수 있다 |
| 모델 산출·분석 결과 | `ae_class_maps/`, `의사결정DB/`, `연도별라벨/검증결과/` | 전처리가 아니라 모델 예측과 평가 결과 |
| 원자료 복사본 | `춘천시_숲가꾸기_임상도_사업실적_결합.gpkg`의 `임상도2019`·`임상도2025` 레이어 | 임상도 원본과 같다 (`data/`) |
| 보조 파일 | `*.png`, `*.qgz`, `산지전용/maps/`, `*.aux.xml`, `*.log`, `Chuncheon_S2_2019_2025_labels.b64.txt`, 숲가꾸기 `.shp` 사본, `output/Chuncheon_convlabels_AE_10m.tif`(`산지전용/`에 같은 파일 있음) | 지도 그림, QGIS 프로젝트, 통계 캐시, 실행 기록, 전송용 인코딩, 중복 |

비공개 자료까지 포함한 내부 공유본이 필요하면, 자료 제공 기관의 허가 범위를 확인한 뒤 별도 폴더로 만든다.
