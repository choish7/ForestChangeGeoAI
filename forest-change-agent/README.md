# 산림변화 탐지 데이터셋 + GeoAI 에이전트 (샘플 구현)

`개발명세서_산림변화_GeoAI.md`의 4.2(데이터셋·탐지모형), 4.3(공간분석), 4.4.2(Geo-LLM 에이전트)를
**현재 보유 자료(춘천시 51110)** 로 구현한 샘플 코드입니다.

```
data/ (보유 원자료)                              GEE
 ├ 임상도 2019·2025 (51110)  ──┐                 AlphaEarth 연간 임베딩 2017~2025 (64-d, 10 m)
 ├ 북부청 경영계획부 소반도 ────┤                 Sentinel-2 L2A 성장기 NDVI·NBR (농림위성 대체)
 └ (숲가꾸기·조림 사업정보)     │                 SRTM 표고·경사
                               ▼                        │
 01 임상도 중첩 → 변화 라벨 폴리곤 ──► 02 층화 표본점 ──► 03 GEE 특징 추출 ──► 04 학습·검증
       (work/change_polygons)          (공간블록 분할)     (work/features)      (outputs/)
                               │                                                  │
                               └────────────► 05 GeoAI 에이전트 (Claude tool use) ◄┘
                                              질의해석→DB조회→RS분석→GIS→MCDA→의사결정카드
```

## 실행

```bash
pip install -r requirements.txt
python -c "import ee; ee.Authenticate()"          # GEE 최초 1회 (프로젝트: GEE_PROJECT, 기본 ee-choish7838)

python scripts/01_build_change_labels.py          # 임상도 중첩·라벨 (~3분)
python scripts/02_sample_points.py                # 표본 6,027점
python scripts/03_extract_gee_features.py         # AlphaEarth 9년×64 + S2 + DEM (~2분)
python scripts/03_extract_gee_features.py --export-rasters   # (선택) 전체 래스터 Drive export
python scripts/04_train_evaluate.py               # 학습·ablation·공간검증 (~10분)

python scripts/05_run_agent.py --demo             # LLM 없이 도구 파이프라인 확인
export GEMINI_API_KEY=...                         # Gemini (기본, 모델 GEMINI_MODEL=gemini-3.8-flash)
# export ANTHROPIC_API_KEY=... ; --provider claude  # Claude 사용 시
python scripts/05_run_agent.py "북산면 국유림에서 2019~2025년 벌채·산지전용 현황과 현장점검 우선 임반을 알려줘"
python scripts/05_run_agent.py -i                 # 대화형
```

## 1. 데이터셋 구축 (4.2)

| 단계 | 내용 |
|---|---|
| 라벨 | 임상도 2019×2025 중첩(EPSG:5179) → 조각별 속성비교 규칙(`fcd/labels.py`) |
| 변화유형 | `harvest` 산림→미립목지 또는 영급 3↑→1 초기화 / `conversion` 산림→제지·비산림 / `gain` 비산림→산림 / `no_change` 동일 임상·영급 정상증가 / `type_shift`·`ambiguous` 학습 제외 |
| 발생연도 | 2025 임상도 `갱신년도` (도면 갱신시점, 약한 라벨) |
| 표본 | 내부버퍼 −15 m 안 층화추출, 10 m 셀 중복 제거, 5 km 공간블록 중 20% 독립 홀드아웃 |
| 특징 | AE: 2019→2025 차이벡터 64 + 코사인거리 + 연속연도 거리 8개 요약 / S2: NDVI·NBR 두 시기와 차이 / TERR: 표고·경사 / ATTR: 2019 영급·밀도·수종군 |

라벨 면적 (춘천시, ≥0.1 ha 조각): 무변화 76,495 ha · 벌채·갱신 1,593 ha · 산지전용 278 ha · 산림증가 487 ha · 임상변화(검토) 2,207 ha

## 2. 검증 결과 (`outputs/metrics.json`, `outputs/ablation.csv`)

**T1 변화 여부**: F1. 학습 블록 공간 CV(5-fold)와 독립 공간 홀드아웃(블록 12개, 1,084점)

| 특징 | CV F1 | 공간 홀드아웃 F1 |
|---|---|---|
| AE (AlphaEarth) | 0.859 ± 0.093 | 0.933 |
| S2 | 0.773 ± 0.036 | 0.772 |
| TERR | 0.524 ± 0.068 | 0.513 |
| AE+S2 | 0.866 ± 0.085 | 0.936 |
| **AE+S2+TERR (최종)** | **0.866 ± 0.085** | **0.936** (정밀도 0.943, 재현율 0.930) |

- **T1b 산림속성 결합**(2019 산림 표본): 속성(ATTR)을 더하면 공간 홀드아웃 F1이 0.896에서 0.905로 조금 오릅니다.
- **T2 변화 유형**(macro-F1): 공간 홀드아웃 0.777입니다. 클래스별 F1은 벌채 0.88, 증가 0.83, 전용 0.62로, 전용이 가장 어렵습니다.
- **T3 발생연도**(3,008건): AlphaEarth 연속연도 거리가 최대인 연도를 도면 갱신년도와 비교했습니다.
  - 정확히 일치 10.7%, ±1년 이내 26.4%
  - 최대 거리 연도가 갱신년도와 같거나 앞선 경우 91.7%, 0~2년 앞선 경우 56.2%
  - 라벨의 70%가 일괄 갱신 연도인 2025년이어서 **실제 발생연도는 검증할 수 없습니다**. 사업·허가 자료로 연도 라벨을 확보해야 합니다.
- **즉석 스캔 임계값**: 코사인거리 0.0665입니다. 학습셋에서 산림 손실 F1이 최대가 되는 값으로 정했습니다.

> AlphaEarth 단독으로도 S2 분광지수보다 F1이 약 0.16 높습니다. S2·지형을 더해도 향상 폭은 작습니다(+0.003).

## 3. GeoAI 에이전트 (4.4.2)

| 단계 | 도구 (`fcd/tools.py`) | 검증·출력 |
|---|---|---|
| 질의 해석 | `set_analysis_spec` | 지역명 → region_id 해석, 기간·유형 검증, 모호하면 되묻기 |
| 데이터검색 | `query_change_map`, `query_forest_status`, `query_mgmt_plan`, `list_regions` | 출처·기준시점·면적합 일치 검사 |
| 원격탐사 분석 | `scan_alphaearth_change`(임의 연도쌍, GEE), `run_change_model`(학습모형 추론, 95% CI, 임상도와 일치도) | 모형 적용범위·산림마스크 경고 |
| GIS 공간분석 | `hotspot_analysis` | 격자집계 + Getis-Ord Gi*, BH-FDR 5% |
| 대안 평가 | `rank_priority_units` | 임반별 가중합, 가중치 ±20% 몬테카를로 순위안정성 |
| 결과 제공 | `make_decision_card` | Markdown 카드 + folium 지도, **문장 속 수치 근거검증**, 분석이력 표, "초안-담당자 검토" 상태 |

- LLM은 도구를 고르고, 결과를 해석하고, 서술만 합니다. 면적·순위 같은 수치는 모두 GIS·통계 도구가 계산합니다. `verify.grounding`이 카드 문장 속 수치를 도구 출력과 대조합니다.
- LLM 백엔드는 두 가지이고 `--provider`로 고릅니다.
  - Gemini(`fcd/agent_gemini.py`): 기본 모델 `gemini-3.8-flash`, 환경변수 `GEMINI_MODEL`로 변경
  - Claude(`fcd/agent.py`): 기본 모델 `claude-opus-5`, 환경변수 `FOREST_AGENT_MODEL`로 변경. 서버측 거절 대체(`fallbacks="default"`)를 켜 두었습니다.
  - 두 백엔드는 도구·시스템 프롬프트·검증 로직을 공유합니다. API 키는 환경변수로만 읽습니다.
- 60임반 교차확인 예시에서 세 방법의 결과는 아래와 같습니다.
  - 임상도 비교 DB: 49.6 ha
  - AlphaEarth 즉석 스캔: 49.2 ha
  - 학습모형 추정: 69.4 ha (95% CI 54.8~84.1), 임상도 라벨과 일치도 0.92

## 한계 및 다음 단계

- **농림위성**: 아직 쓰지 못해 Sentinel-2로 대신했습니다. 자료를 확보하면 `gee.s2_summer`를 농림위성 합성으로 바꾸면 됩니다.
- **읍면 단위**: 행정경계 파일이 없어 국유림 소반을 합쳐 만든 근사 구역입니다. `data/`에 읍면동 경계를 넣고 `data.build_regions`를 고치면 됩니다.
- **조림·숲가꾸기 사업정보 CSV**: 지번(PNU)만 있어 공간 연계를 하지 않았습니다. 연속지적도가 있으면 연도 라벨과 경영이력 특징으로 추가할 수 있습니다.
- **통합 임베딩**: 명세서의 딥러닝 융합모형(PyTorch, 다중과업·대조학습)은 이 GBM 베이스라인 다음 단계입니다. `work/features.parquet`를 그대로 입력으로 쓸 수 있습니다.
- **법령·지침 규칙검사**: 근거 자료가 없어 구현하지 않았습니다. 에이전트는 이런 판단을 '담당자 확인 필요'로 표시하도록 되어 있습니다.
