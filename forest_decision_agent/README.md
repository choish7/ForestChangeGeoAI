# 산림변화 의사결정 지원 웹서비스 (춘천시, 2019→2025)

> **English:** see [English summary](#english-summary) at the end of this document.

AlphaEarth 위성 임베딩과 Sentinel-2 지수로 관측한 산림변화를 국유림 경영실적, 임상도, 산지전용 고시, 산불·산사태, 임도 접근성과 함께 보고 자연어로 묻고 답하는 서비스입니다.
답은 **지도, 표, 근거가 추적되는 보고서**로 받습니다.

- **역할 나눔:** 면적·비율·순위·판정은 모두 코드(도구)가 계산합니다. LLM(Claude)은 질의 해석, 도구 선택, 서술만 맡습니다.
- **수치 검증:** 보고서의 숫자가 도구 출력에 없으면 저장 단계에서 '미확인 수치'로 표시합니다.
- **해석 범위:** 위성 변화는 관측된 사실일 뿐 원인을 뜻하지 않습니다. 위법 여부와 법령 적용은 판단하지 않습니다.
- **연계:** 이 서비스는 `../forest-change-agent/`의 원자료(`data/`), 자료 구축 코드(`prep/`), Colab 노트북(`colab/`), 학습 모델(`models/`)을 씁니다. 논문 전체 절차는 `../forest-change-agent/EXPERIMENTS.md` 12번 단계에 있습니다.

---

## 0. 전체 실행 순서 (forest-change-agent 연계)

명령은 모두 프로젝트 루트(`C:\src\ForestChangeGeoAI`)에서 실행합니다. `FCA` = `forest-change-agent/`.
중간 산출물은 공용 작업 폴더 `output/`과 `final_change_2019_2025/`에 쌓입니다.

```
[A] 원자료        FCA/data/ (임상도·경영계획부·소반도·산지전용 고시·지적도·경계·숲가꾸기·산불·산사태·임도/도로)
      │
[B] 위성 래스터    FCA/colab/01 (Colab+GEE) ─────────────────────────────► output/Chuncheon_AE_*, Chuncheon_S2_*
      │
[C] 자료 구축      FCA/prep/*.py (로컬) ─────────────────────────────────► output/ (실적 결합, 변화라벨, 학습 라벨, 마스크, 전용, 재해)
      │
[D] 모델 학습      FCA/colab/05 → FCA/models/s2_tsfm_chuncheon/
      │            FCA/colab/08 → final_change_2019_2025/
      │            FCA/prep/build_ae_class_samples.py → FCA/colab/12 → FCA/models/ae_class_models/
      │
[E] 근거 DB        forest_decision_agent/build_*.py ────────────────────► output/의사결정DB/
      │
[F] 웹 서비스      python -m forest_decision_agent.web  (http://127.0.0.1:8800)
      │
[G] 논문 검증      FCA/scripts/07_event_timing.py, FCA/scripts/10_report_checks.py
```

| 단계 | 명령 (루트에서) | 산출 |
|---|---|---|
| **A** | 원자료를 `forest-change-agent/data/`에 둡니다 (`data/README.md`에 목록). 임도·도로는 `data/춘천_임도_도로.gpkg`입니다 | — |
| **B** | Colab에서 `forest-change-agent/colab/01_위성래스터_내보내기.ipynb`를 실행하고, Drive `GEE_Export/`의 래스터를 `output/`에 내려받습니다 | `output/Chuncheon_AE_{2017..2025}_64Bands_int8.tif`, `Chuncheon_AE_consec_dist_2017_2025.tif`, `Chuncheon_S2_2019_2025_42Bands.tif`, `Chuncheon_S2_2017_2025_54Bands.tif` |
| **C-1** | `python forest-change-agent/prep/join_management_results.py` | `output/춘천_임소반_산림사업실적.gpkg` |
| **C-2** | `python forest-change-agent/prep/clip_thinning_to_chuncheon.py` → `combine_thinning_forest_type.py` → `build_yearly_projects.py` | 숲가꾸기, 연도별 산림사업, 춘천시·읍면동 경계 |
| **C-3** | `python forest-change-agent/prep/build_change_labels_2013_2019_2025.py` → `compare_mgmt_vs_forestmap_labels.py` → `build_training_label_rasters.py` → `build_forest_mask.py` | 변화라벨, 실적–임상도 합의 라벨, `Chuncheon_labels_*`, `Chuncheon_forestmask_*` |
| **C-4** | `python forest-change-agent/prep/build_mountain_conversion_dataset.py` → `build_conversion_labels.py` | `output/산지전용/` (필지 516건, 전용 라벨 래스터) |
| **C-5** | `python forest-change-agent/prep/build_fire_landslide.py` | `output/춘천시_산불_산사태.gpkg` (내부 자료) |
| **D-1** | `python forest-change-agent/scripts/01_build_change_labels.py` → `python forest-change-agent/prep/make_s2_label_raster.py` → Colab `colab/05_S2-TSFM_사전학습.ipynb` | `forest-change-agent/models/s2_tsfm_chuncheon/` (Drive `GEE_Export/s2_tsfm_chuncheon`에서 내려받음) |
| **D-2** | Colab `colab/08_최종변화지도.ipynb` | `final_change_2019_2025/` (Drive에서 내려받음) |
| **D-3** | `python forest-change-agent/prep/build_ae_class_samples.py` → Colab `colab/12_AE분류모델_서비스용.ipynb` | `forest-change-agent/models/ae_class_models/` |
| **E** | 2장의 순서 1~5 | `output/의사결정DB/forest_decision.sqlite`, `.gpkg`, `build_meta.json` |
| **F** | `python -m forest_decision_agent.web` | 웹 화면, 질의마다 `forest_decision_agent/outputs/<시각>/` |
| **G** | `python forest-change-agent/scripts/07_event_timing.py`, `python forest-change-agent/scripts/10_report_checks.py` | 논문 4.3절 시점 검증(E5b·E5c), 4.6절 보고서 검사(E8) |

자료와 모델이 이미 있으면 B~D를 건너뛰고 **E → F**만 실행해도 됩니다. DB까지 있으면 **F**만 실행하면 됩니다.

---

## 1. 빠른 시작

**웹서비스는 `forest_decision_agent` 폴더만 있으면 실행됩니다.** 실행에 필요한 자료는 이 폴더의 `data/`에 들어 있습니다. 이 폴더의 **바로 위 폴더**에서 실행합니다(프로젝트 안에서는 `C:\src\ForestChangeGeoAI`).

```bash
python -m forest_decision_agent.web
```

브라우저에서 **http://127.0.0.1:8800** 을 엽니다. 서버를 끄려면 실행 창에서 `Ctrl+C`를 누릅니다.

**실행 자료 (`data/`, 약 8.6 GB)** — `sync_data.py`가 프로젝트의 원본에서 채웁니다.

| `data/` 안 | 원본 (프로젝트 루트 기준) | 쓰임 |
|---|---|---|
| `의사결정DB/forest_decision.sqlite`, `.gpkg`, `build_meta.json` | `output/의사결정DB/` | 소반·필지·재해 판정, 지도 레이어, 화면 상단 모델 성능 |
| `satellite/Chuncheon_AE_{2017..2025}_64Bands_int8.tif`, `Chuncheon_S2_2017_2025_54Bands.tif` | `output/` | 소반 시계열(임베딩 거리, NDVI·NDMI) |
| `final_change_2019_2025/` (`final_change_class·year·prob.tif`, `final_change_patches.gpkg`) | `final_change_2019_2025/` | 최종지도 예측, 변화 패치 레이어 |
| `s2_tsfm_chuncheon/` (`change_class.tif`, `change_prob.tif`) | `forest-change-agent/models/s2_tsfm_chuncheon/` | S2-TSFM 예측 |

- 20 MB가 넘는 래스터는 같은 드라이브 안에서 NTFS 하드링크로 둡니다. 디스크를 추가로 쓰지 않고, 폴더를 다른 곳으로 복사하면 실제 내용이 함께 복사됩니다.
- **근거 DB를 다시 만든 뒤**(2장)에는 `python forest_decision_agent/sync_data.py`를 실행해 `data/`를 갱신합니다. `--check`를 붙이면 누락만 확인합니다.
- 다른 위치의 자료를 쓰려면 환경변수 `FDA_DATA_DIR`에 그 폴더를 지정합니다.
- `data/`에는 비공개 경영계획부 자료와 내부 산불·산사태 자료가 들어 있습니다. 공개 배포(`make_release.py`)에서 제외되고, 외부로 공유하지 않습니다.

| 확인 항목 | 내용 |
|---|---|
| Python 패키지 | `fastapi`, `uvicorn`, `geopandas`, `pyogrio`, `rasterio`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `anthropic` |
| 실행 자료 | `data/`의 19개 파일. 확인: `python forest_decision_agent/sync_data.py --check` |
| Claude API 키 (선택) | 환경변수 `ANTHROPIC_API_KEY` 또는 `forest_decision_agent/claude_key.txt` 첫 줄. 없으면 규칙 기반 분석으로 동작합니다. **`claude_key.txt`는 배포·공유하지 않습니다.** |
| 인터넷 | 배경지도(OpenStreetMap, Esri 위성영상)와 지도 라이브러리(Leaflet)를 인터넷에서 불러옵니다 |

화면 위쪽 상태 표시줄에서 Claude 연결 여부와 변화탐지 모델 성능(최종지도 F1, S2-TSFM F1)을 확인할 수 있습니다.

---

## 2. 근거 DB 구축 (처음 한 번, 또는 자료가 바뀌었을 때)

0장의 A~D가 끝난 뒤 실행합니다. 웹서버가 DB 파일을 잡고 있으면 다시 만들 수 없으므로 **서버를 끄고** 실행합니다.

1번 `build_db.py`는 DB를 새로 만들므로 뒤 단계(2~5)가 추가한 테이블도 지워집니다. 1번을 다시 실행했으면 2~5도 다시 실행합니다.

| 순서 | 명령 | 하는 일 | 소요 |
|---|---|---|---|
| 1 | `python forest_decision_agent/build_db.py` | 소반 1,136곳과 산지전용 필지 516곳에 판정을 붙입니다. 쓰는 근거: 위성 변화신호(AlphaEarth·Sentinel-2·학습모형), 사업실적, 임상도 변화, 재해, 경사·임도거리 | 약 5분 |
| 2 | `python forest_decision_agent/build_trajectory.py` | 소반별로 계산합니다: 2017~2025 연간 추이(임베딩 거리, NDVI·NDMI·NDBI), 변화경로 판정, 임상도 2019·2025 우세 속성, 소반 평균 임베딩 | 약 10분 |
| 3 | `python forest_decision_agent/build_ae_classes.py` | AlphaEarth 분류모델 3종(산지 피복·임상·변화유형)을 소반·필지에 적용하고 집계 | 약 27분 |
| 4 | `python forest-change-agent/prep/build_yearly_labels.py` → `save_yearly_validation_set.py` → `validate_yearly_ae_s2.py` | 연도별 관측 라벨(2017~2025)과 검증 결과를 만듭니다. 라벨은 임상도 2019·2024·2025판과 갱신년도로 정하고, 검증은 AE와 S2를 비교합니다. 결과는 `output/연도별라벨/` | 약 15분 |
| 5 | `python forest_decision_agent/build_yearly_history.py` | 연도별 라벨로 AlphaEarth 연도별 임상 모델을 학습하고, 소반별 연도별 임상 추정, 임상도 조사 이력, 연도 신뢰도를 만듭니다 | 약 8분 |

**입력 자료 위치**

| 구분 | 위치 |
|---|---|
| 위성 원자료 | `output/Chuncheon_AE_*_64Bands_int8.tif`(2017~2025), `output/Chuncheon_AE_consec_dist_2017_2025.tif`, `output/Chuncheon_S2_2019_2025_42Bands.tif`, `output/Chuncheon_S2_2017_2025_54Bands.tif` |
| 학습모형 결과 | `final_change_2019_2025/`(AlphaEarth+S2 최종지도), `forest-change-agent/models/s2_tsfm_chuncheon/`(S2-TSFM), `forest-change-agent/models/ae_class_models/`(분류모델 3종, 연도별 임상 모델) |
| 행정·공간 자료 | `output/춘천_임소반_산림사업실적.gpkg`, `output/춘천_경영실적_임상도_변화라벨_비교.gpkg`, `output/Chuncheon_labels_*.tif`, `output/Chuncheon_forestmask_*.tif`, `output/산지전용/`, `output/춘천시_산불_산사태.gpkg`, `output/춘천시_읍면동경계.shp`, `output/ae_class_samples.npz`, `output/연도별라벨/` |
| 임도·도로 | `forest-change-agent/data/춘천_임도_도로.gpkg` (레이어 `임도`, `도로_1000`) |
| 공용 코드 | `forest-change-agent/prep/combine_thinning_forest_type.py`, `validate_yearly_ae_s2.py` (2번, 5번이 import) |

---

## 3. 화면 구성

```
┌────────────── 왼쪽 ──────────────┬──────── 가운데 ────────┬──────── 오른쪽 ────────┐
│ 질의 · 장소 · 속성 · 지역 · 보고서 │        지도            │      분석 보고서        │
│ (탭)                             │ (레이어·강조·결과목록) │ (MD·HTML·인쇄·PDF 저장) │
└──────────────────────────────────┴────────────────────────┴────────────────────────┘
```

### 왼쪽 탭

| 탭 | 사용법 |
|---|---|
| **질의** | ① 대상 지역을 고릅니다: 읍면동 선택, `지도 범위`(현재 화면), `선택 소반`(지도에서 Shift+클릭으로 여러 개 선택), `전체`. ② 질의를 쓰거나 예시 칩을 누릅니다. ③ 모드를 고르고 `질의 실행`을 누릅니다. 진행 로그가 아래에 나옵니다. |
| **장소** | 읍면동·리·지번·소반ID(예: `060-000`)·고시번호로 검색합니다. 결과를 누르면 지도로 이동하고 속성을 엽니다. |
| **속성** | 지도에서 소반·산지전용 필지·재해 지점을 누르면 속성과 관련 정보(2019~2025 사업실적, 가까운 재해, 1 km 안 산지전용 필지, AlphaEarth 분류모델 결과)를 보여 줍니다. |
| **지역** | LLM 없이 대상 지역 요약(판정별 현황, 사업실적, 변화연도, 산지전용, 재해, 점검 우선순위)을 즉시 계산합니다. `이 지역 보고서 저장`을 누르면 보고서로 남깁니다. |
| **보고서** | 저장된 보고서 목록입니다. 누르면 오른쪽에 다시 열립니다. |

**질의 모드**

| 모드 | 동작 |
|---|---|
| 자동 | API 키가 있으면 Claude 에이전트, 없으면 규칙 기반 |
| Claude 에이전트 | Claude가 질의를 해석해 필요한 도구를 골라 실행하고 보고서를 씁니다 (API 사용량 필요) |
| 규칙 기반 | 질의 키워드에 따라 정해진 도구를 실행하고 템플릿 보고서를 만듭니다 (LLM 불필요, 비용 없음) |

### 속성 창의 버튼 (소반)

| 버튼 | 내용 |
|---|---|
| 시계열·모델 근거 | 2017~2025 NDVI(선)와 AlphaEarth 연간 임베딩 거리(막대) 그래프, 최종지도·S2-TSFM 예측(유형·연도·확률) |
| 변화 해석 | 변화경로(예: 교란 후 회복, 회복 지연, 지속 손실 의심)와 근거를 관측 / 기록 / 해석 / 추가 확인 / 불확실성으로 나눠 보여 줍니다 |
| 유사 소반 | 이 소반과 임베딩이 비슷한 소반(비교조사·시범지 후보)과 경사·표고·수종 차이 |
| 대상 지역으로 선택 / 이 소반 질의 | 이 소반을 질의 대상으로 지정하거나 바로 질의 문장을 만듭니다 |

### 가운데 지도

| 레이어 | 표시 |
|---|---|
| 국유림 소반(판정) | 빨강 미기록 변화(점검 필요), 주황 기타실적·변화관측, 파랑 기록일치, 보라 실적있으나 변화 미관측, 연두 안정 |
| 산지전용 고시 필지 | 진파랑 전용확정, 하늘 전용후보, 회색 전용이전 |
| 산불·산사태 | 2019~2025 신고 지점 |
| 모델 변화 패치 | AlphaEarth+S2 최종지도의 벌채·갱신(주황)·산지전용(보라)·산림증가(초록) 구역 |
| 배경 | 일반지도 / 위성영상 (오른쪽 위 레이어 단추) |

**결과 강조:** 보고서를 열면 결과 소반이 자동으로 강조됩니다.
- 나머지 소반은 흐리게 처리하고, 결과 소반에는 노란 테두리, 색 경계, 깜빡이는 순위 번호를 표시합니다.
- 지도가 여러 장이면 묶음마다 색이 다릅니다(빨강 1, 2…, 보라 B1, B2…).
- 왼쪽 위 목록에서 항목을 누르면 그 소반으로 확대하고, `해제`를 누르면 강조를 끕니다.

### 오른쪽 보고서
`지도에 표시`(결과 강조 다시 보기), `MD`(원문), `HTML`(지도 이미지 포함 단일 파일), `인쇄·PDF` 버튼이 있습니다.

---

## 4. 소반 판정 기준

**변화관측:** 학습모형 최종지도(AlphaEarth+S2, 10 m, 산림 마스크·최소면적 적용)의 변화면적이 0.5 ha 이상인 소반입니다.

| 판정 | 조건 |
|---|---|
| 기록일치(수확실적·변화관측) | 2020~2025 수확·갱신 실적 + 변화관측 |
| 미기록 변화(점검 필요) | 실적 없음 + 변화관측 |
| 기타실적·변화관측(확인) | 솎아베기·피해 등 실적 + 변화관측 |
| 실적있으나 변화 미관측(이행 확인) | 수확·갱신 실적 + 변화 없음 |
| 안정 | 그 밖 |

**신호일치수(0~3):** AlphaEarth 임베딩 거리, S2-TSFM, 최종지도 가운데 몇 개가 변화로 봤는지 나타냅니다. 1이면 신호 하나뿐이라 불확실합니다.

**변화경로 (소반 연간 추이 기반, 잠정 기준)**

| 경로 | 관측 패턴 | 추가 확인 |
|---|---|---|
| 교란 후 회복 | 임베딩 급변 + NDVI 0.08 이상 감소 후 절반 이상 회복 | 벌채·산불·조림 기록 |
| 교란 후 회복 지연 | 감소 후 회복률 0.5 미만 | 조림·복구 시행, 활착 상태 |
| 지속 손실·전환 의심 | 학습모형 전용 판정, 또는 NDBI 크게 증가 + 2025 NDVI 낮음 | 고해상도 영상, 인허가 |
| 최근 교란(회복 판단 이름) | 최대 변화가 2024년 이후 | 다음 해 영상 |
| 수분 스트레스 등 식생 이상 의심 | NDVI 감소는 작고 NDMI 감소가 큼 | 기상, 병해충, 관측 품질 |
| 식생 증가·성장 | NDVI 상승 | 조림 기록, 초지·관목 구분 |
| 임베딩 변화·지수 안정 | 임베딩만 변하고 지수는 안정 | 원영상, 관측 조건 |
| 반복 교란 (별도 표시) | 떨어진 두 시기 이상에서 고변화 | 집중 감시 |

임베딩 개별 차원에는 물리적 의미를 붙이지 않습니다. 변화의 의미는 임베딩(시기·크기), 지수 추이(방향·지속성), 임상도·실적·재해(맥락)가 같은 해석을 지지하는지로 설명합니다.

---

## 5. 분석 도구 (에이전트가 호출)

| 도구 | 답하는 질문 |
|---|---|
| `describe_data` | 어떤 자료·컬럼·기준·한계가 있나 |
| `run_sql` | 근거 DB 읽기 전용 조회 (SELECT만 허용) |
| `area_summary` | 지역(읍면동·지도 범위·소반 목록)의 판정·실적·변화연도·산지전용·재해 요약 |
| `rank_inspection` | 현장점검 우선순위 (위성변화 0.4, 기록 부재 0.3, 재해이력 0.15, 접근성 0.15 가중합 + 가중치 ±30% 200회 순위안정성) |
| `unit_timeseries` | 소반·필지의 원자료 시계열(NDVI·NDMI, 연간 임베딩 거리)과 모델 예측 |
| `explain_change` | 소반 변화의 의미 설명 (관측·기록·해석·추가확인·불확실성) |
| `find_trajectory` | 회복 지연, 반복 교란, 지속 손실 등 경과별 소반 찾기 |
| `find_similar_units` | 기준 소반과 임베딩이 비슷한 소반 (비교·검증 후보) |
| `map_update_candidates` | 임상도와 위성의 불일치: A 갱신 검토, B 판독 수정 가능, C 임상 재판독, D1 제지 확대, D2 수확기록 없는 미립목지화 |
| `change_zones` | 변화 구역(패치)과 소반 중첩 면적표, 경계 확인 필요 구역 |
| `ae_forest_status` | AlphaEarth 분류모델로 본 지역의 피복·임상(2019→2025)·변화유형 면적과 임상도 일치율 |
| `survey_history` | 소반의 임상도 조사 이력(조사연도, 연도별 관측·유지추정·변화구간)과 AlphaEarth 연도별 임상 추정(2017~2025) |
| `resurvey_priority` | 임상도 재조사 우선순위: 마지막 조사 이후 위성 변화(임상도 미반영), 연도별 임상 우세 전환, 조사 경과연수 |
| `embedding_relations` | 임베딩 차원과 NDVI·NDMI·NDBI의 상관 (탐색용, 차원 이름 붙이기 근거 아님) |
| `evaluate_foundation_model` | AlphaEarth와 Sentinel-2·학습모형의 변화탐지 성능 비교 |
| `make_map`, `write_report`, `session_stats` | 지도, 보고서·근거기록·수치 추적률, 처리 시간·자료 결합 통계 |

### AlphaEarth 분류모델 3종

라벨은 임상도 속성만으로 만들었고, 2017~2025 임베딩으로 학습했습니다. 성능은 독립 공간 블록 검증 결과입니다.

| 모델 | 클래스 | macro-F1 | 쓰임 |
|---|---|---|---|
| 산지 피복 | 산림(입목지)·나지(미립목지)·초지(관목덤불)·시설지(제지) | 0.64 | 제지 확대(D1), 수확기록 없는 미립목지화(D2) |
| 임상 | 침엽수림·활엽수림·혼효림 | 0.65 | 2025 임상도와 우세임상 불일치 → 재판독 후보(C) |
| 변화유형 | 무변화·벌채·갱신·산림증가 (산지전용 제외) | 0.91 | 변화 해석 근거, 지역 현황 |

- **초지:** F1 0.11이라 판단에 쓰지 않습니다.
- **시설지:** 임도·묘지·송전탑 같은 산지 안의 산림 외 용도(제지)이며, 시가지 건물이 아닙니다.

---

### 임상도 연도별 라벨과 연도 신뢰도
- 임상도 2019·2024·2025판 갱신년도(조사연도)로 조각×연도(2017~2025) 상태 라벨을 만들었습니다(관측 / 유지추정 / 변화구간, 신뢰 A·B·C).
- 이 라벨로 학습한 연도별 임상 모델의 성능(학습 블록 학습 → 검증 블록 평가)은 임상 3클래스 macro-F1 0.58~0.68입니다. 2019·2025 라벨만 쓴 모델보다 연도별로 0.01~0.05 높습니다. 다년 특징이라 갑작스러운 벌채의 약 30%는 늦게 반영됩니다.
- 같은 라벨의 안정 조각 검증에서 AlphaEarth 연속연도 신호는 2022·2023년 오탐률(5.6%·7.4%)이 높았습니다. 그래서 그 해 변화연도이고 신호일치수가 1 이하인 소반은 점검 점수를 30% 깎고 '연도신뢰도낮음'으로 표시합니다.

## 6. 질의 예시

| 목적 | 질의 |
|---|---|
| 실적–위성 불일치 점검 | 춘천 국유림의 2020~2024년 산림변화 중 경영실적과 위성분석 결과가 일치하지 않는 소반을 찾아줘. 사업기록이 없는데 변화가 탐지된 곳과, 사업기록은 있지만 변화 신호가 약한 곳을 구분하고 현장 점검 후보 10곳을 지도와 표로 작성해줘. |
| 복구지 점검 | 2019~2025년 국유림에서 큰 변화 후 식생 회복이 더딘 소반을 찾고, 벌채·조림 이력과 비교해 현장 점검 소반을 추천해줘. |
| 비교조사지 선정 | 회복이 양호한 벌채지와 임베딩이 비슷한 소반을 찾아 경사·수종 차이와 함께 비교조사 후보를 정리해줘. |
| 지도 갱신 | 위성 변화는 있는데 2025 임상도에 반영되지 않은 임상도 갱신 검토 소반과 변화 구역 면적표를 정리해줘. |
| 임상도 재조사 | 임상도 조사 이력과 연도별 임상 추정을 바탕으로 동면 국유림의 임상도 재조사 우선순위를 정리해줘. |
| 분류모델 현황 | AlphaEarth 분류모델로 본 북산면 국유림의 2019→2025 피복·임상 변화와, 임상 재판독 후보·산지 내 시설·나지 확대 의심 소반을 정리해줘. |
| 이상 징후 | 반복 교란 소반과 수분 스트레스 등 식생 이상 의심 소반을 찾아 변화의 의미를 설명해줘. |
| 산지전용 | 산지전용 고시 필지 중 위성에서 변화가 확인되는 곳은? |
| 모델 가치 | AlphaEarth가 Sentinel-2보다 변화 탐지에 도움이 되는가? |

**규칙 기반 모드의 키워드 연결:** 질의에 아래 낱말이 들어가면 해당 분석이 보고서에 추가됩니다.

| 키워드 | 추가되는 분석 |
|---|---|
| 회복·복구·조림 | 회복 지연 소반 |
| 반복 | 반복 교란 소반 |
| 지속·전환 | 지속 손실 의심 소반 |
| 수분·병해충 | 식생 이상 의심 소반 |
| 유사·비교조사 | 유사 소반 |
| 갱신 | 임상도 갱신 후보 |
| 구역·면적표·경계 | 변화 구역 |
| 임상·피복·분류 | AlphaEarth 분류모델 현황과 재판독·제지 확대 후보 |
| 재조사·조사 이력·연도별 임상 | 임상도 재조사 우선순위 |
| 산지전용·고시 | 산지전용 필지 |
| AlphaEarth·모델·비교 | 성능 비교 |

---

## 7. 결과 파일

질의마다 `forest_decision_agent/outputs/<YYYYMMDD_HHMMSS>/` 폴더가 생깁니다.

| 파일 | 내용 |
|---|---|
| `report.md` | 보고서 본문 + 근거 기록(수치 추적률, 자료 출처, 도구 호출 순서) |
| `*.png` | 지도 (왼쪽: 춘천시 전체 위치와 순위 번호, 오른쪽: 소반별 확대) |
| `highlight.json` | 웹 지도 강조 정보 |
| `evidence.json` | 모든 도구 호출의 입력·출력 |
| `query.json`, `run_stats.json` | 질의·지역·모드, 소요 시간·토큰 (Claude 모드) |

명령줄에서 웹 없이 실행할 수도 있습니다.
```bash
python -m forest_decision_agent.agent --demo                    # API 키 없이 고정 흐름 시연
python -m forest_decision_agent.agent "질의 문장"                 # Claude 에이전트
```

---

## 8. 문제 해결

| 증상 | 조치 |
|---|---|
| 화면이 응답하지 않음 | 메모리가 부족하면 시스템이 서버를 끌 수 있습니다. `python -m forest_decision_agent.web`으로 다시 실행합니다. |
| 질의 실행 오류 "API 키가 없습니다" / "credit balance is too low" | 모드를 `규칙 기반`으로 바꾸거나, `claude_key.txt`를 확인하거나 API 크레딧을 충전합니다. |
| DB 다시 만들 때 `PermissionError` | 웹서버가 DB를 쓰고 있습니다. 서버를 끄고 다시 실행합니다. |
| `FileNotFoundError: ...models/...` | 학습 모델이 `forest-change-agent/models/`에 없습니다. 0장의 D 단계로 만들거나 Drive에서 내려받습니다. |
| `ModuleNotFoundError: combine_thinning_forest_type` / `validate_yearly_ae_s2` | `forest-change-agent/prep/`가 있는지 확인합니다. 루트에서 실행해야 합니다. |
| 분류모델 결과가 속성 창에 없음 | `build_ae_classes.py`를 실행합니다. |
| 변화 해석·유사 소반 오류 | `build_trajectory.py`를 실행합니다. |
| 포트 8800 사용 중 | 기존 서버를 끕니다 (`netstat -ano | findstr :8800`으로 PID 확인 후 `taskkill /PID <번호> /F`). |

---

## 9. 배포 구성

공개 배포본은 루트의 `make_release.py`가 만듭니다 → `release/ForestChangeGeoAI/`. 배포 저장소의 `.git`은 그대로 둡니다.

| 구분 | 파일 |
|---|---|
| **포함** | `*.py` (`agent`, `build_db`, `build_trajectory`, `build_ae_classes`, `build_yearly_history`, `tools`, `translate`, `web`, `__init__`), `static/index.html`, `README.md`, `reference_results.json`, `docs/capture_screens.py`, `docs/capture_en.py` |
| **함께 포함 (연계)** | `forest-change-agent/prep/*.py`, `colab/`, `models/`(S2-TSFM, AE 분류모델 `*_local`), `EXPERIMENTS.md`, `data/README.md` |
| **제외: 비밀값** | `claude_key.txt` (API 키). 배포 스크립트가 파일 이름과 내용(`sk-ant-…`)을 모두 검사합니다 |
| **제외: 실행 기록** | `outputs/` (질의 보고서·근거 기록. 소반 단위 실적과 재해 정보가 들어 있음) |
| **제외: 화면 캡처** | `docs/screens/`, `docs/screens_hi/`. 지도에 산불·산사태 지점(내부 자료)과 실적 기반 소반 판정이 보이고, 옛 이름(Mini-GFM)도 남아 있습니다. 공개하려면 해당 레이어를 끄고 다시 캡처합니다 (`docs/capture_screens.py`) |
| **제외: 기타** | `static/index_backup_ko.html`(백업), `docs/capture_fix.py`·`capture_rest.py`(일회성 보정), `docs/*.docx`(기능 설명서, 캡처 포함), `__pycache__/` |
| **제외: 자료** | 실행 자료(`forest_decision_agent/data/`), 근거 DB(`output/의사결정DB/`), 원자료(`forest-change-agent/data/`) — 비공개 경영계획부와 내부 산불·산사태 자료가 들어 있음 |

배포본만으로는 근거 DB가 없어 서비스가 바로 돌지 않습니다. 자료 제공 기관의 허가를 받은 자료로 0장 A~E를 실행해야 합니다.

---

## 10. 해석 시 유의

- **원인 판단 아님:** 위성 변화는 관측된 사실이며 원인이 아닙니다. 실적 없는 변화도 위법으로 단정하지 않고 '기록과 일치하지 않는 변화 후보'로 표현합니다.
- **실적 범위:** 사업실적은 국유림 소반 단위로만 있습니다. 소반 일부에서 한 사업도 소반 전체에 연결되고, 사유림 변화는 산지전용 고시로만 대조할 수 있습니다.
- **연도 정확도:** 변화연도는 연간 자료에서 임베딩이 가장 크게 바뀐 해입니다. 실제 발생 시점과 1년 안팎 차이가 날 수 있습니다(국유림 수확 실적 기준 1년 이내 일치 77%).
- **임상도와 실제 변화는 다를 수 있음:** 임상도가 그대로라는 사실만으로 무변화라고 보지 않습니다(갱신 지연 가능). 반대로 임상도가 바뀌었어도 판독 수정일 수 있습니다.
- **유사 지역의 의미:** 유사 지역 검색은 관측된 지표 특성이 비슷하다는 뜻일 뿐입니다. 기후·토양·생태 기능이 같다는 보장은 없습니다.
- **판정 기준값:** 판정 기준값은 춘천 자료로 정한 잠정값입니다. 다른 지역에 쓰려면 라벨을 다시 만들고 검증해야 합니다.

---

## English summary

### Forest-change decision-support web service (Chuncheon, 2019→2025)

This service lets users ask questions in natural language about forest change in Chuncheon. Forest change is observed with AlphaEarth satellite embeddings and Sentinel-2 indices, and it is read together with:
- national-forest management records
- forest type maps
- forest-conversion notices
- wildfire and landslide reports
- forest-road access

Answers come as **maps, tables and reports whose numbers can be traced to tool outputs**. The interface is available in Korean and English. The service implements Sections 3.7 and 4.7 of the paper; the full procedure is in `../forest-change-agent/EXPERIMENTS.md` (step 12).

**Design principles**

- **Division of work**: all areas, fractions, ranks and judgements are computed by deterministic tools. The LLM (Claude) only interprets the query, selects tools and writes the narrative.
- **Numeric grounding**: any number in a report that does not match a tool output is flagged as "unverified" when the report is saved.
- **Scope**: satellite change is an observation, not a cause. The service does not judge legality or permit compliance.

### 1. Quick start (this folder alone is enough)

The web service runs with only the `forest_decision_agent` folder. Run it from the **parent folder** of `forest_decision_agent` (in this project, `C:\src\ForestChangeGeoAI`):

```bash
python -m forest_decision_agent.web
```

Open **http://127.0.0.1:8800**; stop the server with `Ctrl+C`.

| Requirement | Details |
|---|---|
| Python packages | `fastapi`, `uvicorn`, `geopandas`, `pyogrio`, `rasterio`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `anthropic` |
| Runtime data | 19 files in `data/`. Check with `python forest_decision_agent/sync_data.py --check` |
| Claude API key (optional) | `ANTHROPIC_API_KEY` or the first line of `forest_decision_agent/claude_key.txt`. Without a key the service runs in rule-based mode. **Never share or publish `claude_key.txt`.** |
| Internet | Base maps (OpenStreetMap, Esri imagery) and Leaflet are loaded online |

**Runtime data (`data/`, about 8.6 GB)** — filled from the project sources by `sync_data.py`:

| In `data/` | Source (relative to project root) | Used for |
|---|---|---|
| `의사결정DB/forest_decision.sqlite`, `.gpkg`, `build_meta.json` | `output/의사결정DB/` | Stand, parcel and disaster judgements; map layers; model scores in the status bar |
| `satellite/Chuncheon_AE_{2017..2025}_64Bands_int8.tif`, `Chuncheon_S2_2017_2025_54Bands.tif` | `output/` | Per-stand time series (embedding distance, NDVI, NDMI) |
| `final_change_2019_2025/` (class, year, probability rasters; change patches) | `final_change_2019_2025/` | Final-map predictions and change-patch layer |
| `s2_tsfm_chuncheon/` (`change_class.tif`, `change_prob.tif`) | `forest-change-agent/models/s2_tsfm_chuncheon/` | S2-TSFM predictions |

- Rasters over 20 MB are NTFS hard links within the same drive: they use no extra disk space, and copying the folder elsewhere copies their full content.
- After rebuilding the evidence DB (Section 3), run `python forest_decision_agent/sync_data.py` to refresh `data/`.
- Set `FDA_DATA_DIR` to use data in another location.
- `data/` contains non-public management-plan data and internal fire/landslide records. It is excluded from the public release and must not be shared.

### 2. Full pipeline (linked to `forest-change-agent`)

Run all commands from the project root; `FCA` = `forest-change-agent/`. Derived files are stored in the shared folders `output/` and `final_change_2019_2025/`.

| Step | Command | Output |
|---|---|---|
| **A** Raw data | Place inputs in `FCA/data/` (list in `data/README.md`); roads are in `data/춘천_임도_도로.gpkg` | — |
| **B** Satellite rasters | Run `FCA/colab/01_위성래스터_내보내기.ipynb` in Colab and download from Drive `GEE_Export/` to `output/` | AlphaEarth 2017–2025, consecutive-year distances, Sentinel-2 42/54-band stacks |
| **C-1** | `python FCA/prep/join_management_results.py` | Management records joined to stands |
| **C-2** | `clip_thinning_to_chuncheon.py` → `combine_thinning_forest_type.py` → `build_yearly_projects.py` (in `FCA/prep/`) | Thinning sites, yearly projects, boundaries |
| **C-3** | `build_change_labels_2013_2019_2025.py` → `compare_mgmt_vs_forestmap_labels.py` → `build_training_label_rasters.py` → `build_forest_mask.py` | Change labels, consensus labels, label rasters, forest masks |
| **C-4** | `build_mountain_conversion_dataset.py` → `build_conversion_labels.py` | Conversion parcels (516) and label rasters |
| **C-5** | `build_fire_landslide.py` | Fire/landslide points (internal data) |
| **D-1** | `FCA/scripts/01_build_change_labels.py` → `FCA/prep/make_s2_label_raster.py` → Colab `colab/05_S2-TSFM_사전학습.ipynb` | `FCA/models/s2_tsfm_chuncheon/` |
| **D-2** | Colab `colab/08_최종변화지도.ipynb` | `final_change_2019_2025/` |
| **D-3** | `FCA/prep/build_ae_class_samples.py` → Colab `colab/12_AE분류모델_서비스용.ipynb` | `FCA/models/ae_class_models/` |
| **E** Evidence DB | Section 3 below, then `sync_data.py` | `output/의사결정DB/` → `forest_decision_agent/data/` |
| **F** Web service | `python -m forest_decision_agent.web` | Web UI; one folder per query in `forest_decision_agent/outputs/<timestamp>/` |
| **G** Paper checks | `FCA/scripts/07_event_timing.py`, `FCA/scripts/10_report_checks.py` | Timing checks (Section 4.3), report checks (Section 4.6) |

If the data and models already exist, run only **E → F**. If `data/` is already filled, run only **F**.

### 3. Building the evidence DB

Stop the web server first, because a running server keeps the DB files open. `build_db.py` recreates the DB from scratch, so after re-running step 1 you must also re-run steps 2–5.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python forest_decision_agent/build_db.py` | Judges 1,136 stands and 516 conversion parcels from satellite change signals, management records, map change, disasters, slope and road distance | ~5 min |
| 2 | `python forest_decision_agent/build_trajectory.py` | Yearly trajectories per stand (2017–2025), change pathways, dominant map attributes, mean embeddings | ~10 min |
| 3 | `python forest_decision_agent/build_ae_classes.py` | Applies the three AlphaEarth classifiers (land cover, forest type, change type) | ~27 min |
| 4 | `FCA/prep/build_yearly_labels.py` → `save_yearly_validation_set.py` → `validate_yearly_ae_s2.py` | Yearly observation labels from the 2019/2024/2025 maps and their validation | ~15 min |
| 5 | `python forest_decision_agent/build_yearly_history.py` | Yearly forest-type model, per-stand yearly estimates, map survey history, year reliability | ~8 min |

Then run `python forest_decision_agent/sync_data.py`.

### 4. Using the interface

- **Layout**: left tabs (Query, Place, Attributes, Area, Reports), a central map and a report panel on the right (MD, HTML, print/PDF).
- **Query modes**:
  - *Auto*: Claude if a key is available, otherwise rule-based.
  - *Claude agent*: Claude selects and runs tools, then writes the report.
  - *Rule-based*: keyword-driven tools with a template report; no LLM and no cost.
- **Stand buttons**:
  - *Time series*: NDVI and AlphaEarth distances, with model predictions.
  - *Change explanation*: observation / records / interpretation / checks / uncertainty.
  - *Similar stands*: stands with similar embeddings.
- **Map layers**: stand judgements, conversion parcels, fire/landslide reports, model change patches, base map or satellite imagery.

### 5. Stand judgements

A stand counts as **changed** when the final AE+S2 map shows ≥ 0.5 ha of change.

| Judgement | Rule |
|---|---|
| Records match | Harvest/regeneration record 2020–2025 + change |
| Unrecorded change (inspect) | No record + change |
| Other records with change | Thinning, damage or other records + change |
| Records without observed change | Harvest/regeneration record + no change |
| Stable | Otherwise |

- **Signal agreement (0–3)** counts how many of three sources flag change: AE embedding distance, S2-TSFM and the final map.
- **Change pathways** use provisional thresholds from Chuncheon: recovery after disturbance, delayed recovery, suspected persistent loss, recent disturbance, suspected moisture stress, vegetation gain, embedding-only change, repeated disturbance.

### 6. Main tools

| Tool | Purpose |
|---|---|
| `describe_data`, `run_sql` | Data dictionary; read-only SQL (SELECT only) |
| `area_summary`, `rank_inspection` | Area summary; inspection priority by weighted sum (satellite change 0.4, missing record 0.3, disasters 0.15, access 0.15) with ±30% weight stability over 200 trials |
| `unit_timeseries`, `explain_change`, `find_trajectory`, `find_similar_units` | Time series, change explanation, pathway search, similar stands |
| `map_update_candidates`, `change_zones`, `ae_forest_status`, `survey_history`, `resurvey_priority` | Forest-map update and re-survey candidates, change zones, AlphaEarth classifier status |
| `evaluate_foundation_model` | AlphaEarth vs Sentinel-2 vs trained models |
| `make_map`, `write_report`, `session_stats` | Maps, reports with numeric-grounding rate, run statistics |

AlphaEarth classifiers (spatial-block validation, macro-F1):

| Classifier | Macro-F1 | Note |
|---|---|---|
| Land cover | 0.64 | Grassland F1 is 0.11, so it is not used for decisions |
| Forest type | 0.65 | |
| Change type (excluding conversion) | 0.91 | |

### 7. Outputs

Each query creates `forest_decision_agent/outputs/<YYYYMMDD_HHMMSS>/`:

| File | Content |
|---|---|
| `report.md` | Report and evidence record |
| `*.png` | Maps |
| `highlight.json` | Map highlights for the web UI |
| `evidence.json` | Inputs and outputs of all tool calls |
| `query.json`, `run_stats.json` | Query and run statistics |

Command line without the web UI: `python -m forest_decision_agent.agent --demo` (no API key), or `python -m forest_decision_agent.agent "<query>"`.

### 8. Release contents

`make_release.py` builds the public release in `release/ForestChangeGeoAI/` and keeps that folder's `.git`.

- **Included**: service code (`*.py`), `static/index.html`, `README.md`, `reference_results.json`, screenshot scripts, and the linked `forest-change-agent` code and models.
- **Excluded**:
  - `claude_key.txt`, which is also caught by a content check for `sk-ant-…`.
  - `outputs/` (run records).
  - `data/` and the evidence DB.
  - Screenshots, because they show internal fire/landslide points and record-based judgements.
  - Backups.
- The release cannot run on its own. It needs the evidence DB, built from data used with the provider's permission (steps A–E).

### 9. Interpretation caveats

- **Not a cause**: satellite change is an observation. Change without a record is reported as "change not matching records", never as illegal.
- **Record coverage**: records exist only for national-forest stands. A project in part of a stand is linked to the whole stand, and change in private forests can only be compared with conversion notices.
- **Year accuracy**: the change year is the year of the largest embedding change. It can differ from the event by about one year (77% within one year for harvest records).
- **Forest-map caveats**: an unchanged map does not prove no change, because updates can be delayed. A changed map may only reflect reinterpretation.
- **Similar stands**: they are similar in observed surface characteristics only, not necessarily in climate, soil or ecological function.
- **Thresholds**: thresholds are provisional values set on Chuncheon data. Other regions need new labels and validation.
