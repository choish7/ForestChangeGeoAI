# 원자료 (행정자료·라벨 생성용)

이 폴더에는 지도 전이 라벨, 행정자료 라벨, 인제 전이 시험 라벨을 만드는 데 필요한 원자료를 모았다.

- 루트(`C:\src\ForestChangeGeoAI\`)에 있던 파일을 **원래 이름 그대로** 옮겼다(2026-11-01). 이곳이 유일한 사본이다.
  - 루트의 중복 사본은 지웠다. 해시가 같은 것을 확인한 뒤 지웠다.
  - 루트 `scripts/`와 `paper/revision/` 스크립트도 이 폴더를 읽는다.
- `fcd/config.py`의 `DATA`가 이 폴더를 가리킨다. 환경변수 `FCD_DATA_DIR`로 바꿀 수 있다.
- `prep/*.py`, `scripts/11`, `scripts/13`도 이 폴더에서 읽는다.

> ⚠ **비공개 자료**: 경영계획부 xls와 임소반도는 북부지방산림청이 제공한 비공개 자료다. 공개 배포(`../../make_release.py`)나 저장소에 올리지 않는다.

## 춘천시 (51110)

| 자료 | 출처 | 쓰는 코드 (실험 순서) |
|---|---|---|
| `임상도 2019 51110/`, `임상도 2025 51110/` (1:5,000 임상도, 51110.shp) | 산림청 산림공간정보서비스 | `fcd/config.FOREST_MAP` → `scripts/01` (3), `prep/build_forest_mask.py` (8), `prep/build_change_labels_2013_2019_2025.py` (8), `prep/combine_thinning_forest_type.py` (2), `prep/build_ae_class_samples.py` (12), `prep/build_mountain_conversion_dataset.py` (2) |
| `임상도 2013년 51110/` | 산림청 | `prep/build_change_labels_2013_2019_2025.py` (P1 2013→2019), `prep/build_yearly_labels.py` (12) |
| `임상도 2024 51110/` | 산림청 | `prep/build_yearly_labels.py` (12, 연도별 임상 이력) |
| `춘천(9)_20260921193101230/` (9차 경영계획 임소반도) | 북부지방산림청 (비공개) | `fcd/config.MGMT_PLAN_*` → 임반 112개, 소반 1,136개 (9·10·12 순서), `prep/join_management_results.py` (2) |
| `(조회용)경영계획부_춘천_7차·8차·9차_*.xls` | 북부지방산림청 (비공개) | `prep/join_management_results.py` (2): 산림사업실적. 수확·갱신 실적은 시점 검증(13)에, 합의 라벨은 8에 쓴다 |
| `춘천시 산지전용 고시/` (보전산지 해제 고시 xls 2017~2024 + 고시 이미지) | 춘천시 누리집 고시공고 | `prep/build_mountain_conversion_dataset.py` → `build_conversion_labels.py` (2): 전용 필지 516개, 전용 확정 44필지 |
| `LSMD_CONT_LDREG_강원_춘천시 연속지적도/` | 국가공간정보포털 | `prep/build_mountain_conversion_dataset.py` (2): 고시 필지 PNU를 폴리곤에 결합 |
| `LSMD_ADM_SECT_UMD_강원특별자치도/` (읍면동 경계) | 국가공간정보포털 | `prep/clip_thinning_to_chuncheon.py` (2), `scripts/13_inje_visual_check.py` (15) |
| `산림청_산림경영활동_숲가꾸기공간정보_20171231/` | 산림청 공공데이터 | `prep/clip_thinning_to_chuncheon.py` → `combine_thinning_forest_type.py` → `build_yearly_projects.py` (2) |
| `춘천_임도_도로.gpkg` (레이어 `임도` 119건, `도로_1000` 30,451건) | `C:/src/ai_test/공간데이터/춘천_산림공간정보.gpkg`에서 두 레이어만 추출 (2026-11-01) | `../forest_decision_agent/build_db.py`: 소반별 임도·도로 거리(접근성) |
| `산불정보.csv`, `산사태신고정보.csv` | 산림청 내부 자료 (배포 제외) | `prep/build_fire_landslide.py` (2): 서비스의 재해 레이어 |

## 인제군 (51810) — `타지역 테스트(강원도 인제군)/`

| 자료 | 쓰는 코드 |
|---|---|
| `임상도 51810 (2019 인제)/`, `임상도 51810(2025 인제)/` | `scripts/11_inje_prepare.py` (15): 전이 시험 라벨·표본 |
| `인제(9)_20260921임소반도/`, `(조회용)경영계획부_인제_9차_*.xls` | `scripts/11_inje_prepare.py` (15): 사업실적 시점 검증 표본 (비공개) |

인제 2013·2024 임상도는 어떤 스크립트도 쓰지 않아 복사하지 않았다. 원본은 루트의 `타지역 테스트(강원도 인제군)/`에 있다.

## 여기에 없는 입력

- **위성 래스터**(AlphaEarth, Sentinel-2): `colab/01_위성래스터_내보내기.ipynb`로 GEE에서 내보내 `../../output/`에 둔다.
- **파생 중간 산출물**: `../../output/`에 있다(예: `춘천_임소반_산림사업실적.gpkg`, 라벨 래스터, 산림 마스크).
- **법정동코드**: `prep/build_mountain_conversion_dataset.py`가 `../../output/산지전용/춘천시_법정동코드.csv`를 읽는다. 이 파일이 없으면 PublicDataReader에서 내려받아 만든다.

## 내려받는 곳 / Where to download

공개 저장소(GitHub)에는 원자료를 넣지 않는다. 공개 자료는 아래에서 내려받아 이 폴더에 **위 표의 이름 그대로** 둔다.
The public repository does not include raw data. Download the public datasets below and place them in this folder under the names listed above.

| 자료 / Dataset | 내려받는 곳 / Source | 비고 / Notes |
|---|---|---|
| 임상도 1:5,000 (춘천 51110, 인제 51810; 2013·2019·2024·2025) / Forest type maps | 산림청 산림공간정보서비스 (https://map.forest.go.kr) | 시군구 코드별 shapefile. 폴더 이름 예: `임상도 2019 51110/51110.shp`, `타지역 테스트(강원도 인제군)/임상도 51810 (2019 인제)/` |
| 연속지적도 `LSMD_CONT_LDREG` / Cadastral map | 국가공간정보포털 (https://www.nsdi.go.kr) 또는 브이월드 (https://www.vworld.kr) | 강원 춘천시 |
| 읍면동 경계 `LSMD_ADM_SECT_UMD` / Township boundaries | 국가공간정보포털 (https://www.nsdi.go.kr) | 강원특별자치도 |
| 숲가꾸기공간정보 / Forest-thinning sites | 공공데이터포털 (https://www.data.go.kr), 검색어 "산림청_산림경영활동_숲가꾸기공간정보" | 2017년 배포본 |
| 보전산지 해제(산지전용) 고시 / Conversion notices | 춘천시청 고시·공고 (https://www.chuncheon.go.kr) | 고시 제2017-463, 2018-106, 2019-445, 2020-529, 2021-526, 2022-369, 2023-579, 2024-366, 2024-649호 첨부파일 |
| 법정동코드 / Legal-dong codes | 행정표준코드관리시스템 (https://www.code.go.kr) | 자동 생성됨 (PublicDataReader) |
| AlphaEarth 연간 임베딩 / AlphaEarth annual embeddings | Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | `colab/01_위성래스터_내보내기.ipynb`, `scripts/03_extract_gee_features.py` |
| Sentinel-2 L2A | Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED` | 〃 |
| 지형 / Terrain | Google Earth Engine `USGS/SRTMGL1_003`, `COPERNICUS/DEM/GLO30` | 〃 |
| 토지피복 / Land cover | Google Earth Engine `ESA/WorldCover/v200` | 즉석 임베딩 스캔의 산림 마스크 |

**배포하지 않는 자료 / Not distributed**

| 자료 / Dataset | 이유 / Reason |
|---|---|
| 국유림 경영계획부(7~9차)·임소반도 (춘천, 인제) / National-forest management plans and stands | 북부지방산림청 제공 비공개 자료. 기관에 요청해야 한다 / Non-public; request from the Northern Regional Forest Service |
| 산불·산사태 신고정보 / Wildfire and landslide reports | 내부 자료 / Internal records |
| `춘천_임도_도로.gpkg` (임도·도로) / Forest roads and roads | 내부 실습 공간자료에서 추출 / Extracted from an internal dataset |
