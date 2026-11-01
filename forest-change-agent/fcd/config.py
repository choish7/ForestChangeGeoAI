"""경로·기간·파라미터 공통 설정.

환경변수로 덮어쓸 수 있는 값:
  GEE_PROJECT          GEE 클라우드 프로젝트 ID (기본 ee-choish7838)
  FOREST_AGENT_MODEL   에이전트 LLM 모델 ID (기본 claude-opus-5)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # forest-change-agent/
DATA = Path(os.environ.get("FCD_DATA_DIR", ROOT / "data"))          # 원자료 폴더 (data/README.md 참고)
WORK = ROOT / "work"                                 # 중간 산출물
OUT = ROOT / "outputs"                               # 모형·지표·의사결정 카드
for d in (WORK, OUT):
    d.mkdir(exist_ok=True)

# ---- 원자료 (data/ 에 원래 이름 그대로 둠)
FOREST_MAP = {
    2019: (DATA / "임상도 2019 51110" / "51110.shp").as_posix(),
    2025: (DATA / "임상도 2025 51110" / "51110.shp").as_posix(),
}
FOREST_FUNCTION = DATA / "산림기능구분도 51110" / "51110.shp"   # 미사용 (자료 없음)
MGMT_PLAN_DIR = DATA
MGMT_PLAN_GLOB = "춘천(9)*/*.shp"                     # 춘천국유림관리소 경영계획부(9차) 소반도

# ---- 중간 산출물
CHANGE_POLYGONS = WORK / "change_polygons.parquet"   # 임상도 2019×2025 중첩 + 변화 라벨
SAMPLES = WORK / "samples.parquet"                   # 층화 표본점 + 공간블록/분할
FEATURES = WORK / "features.parquet"                 # GEE 추출 특징 (AlphaEarth·S2·DEM)
REGIONS = WORK / "regions.parquet"                   # 에이전트용 분석단위(시 전체·읍면·임반)
MODEL = OUT / "change_model.joblib"

# ---- 좌표계 / 기간
CRS_WORK = "EPSG:5179"        # 임상도 원본 좌표계 (면적 계산)
CRS_GEE = "EPSG:4326"
YEAR_FROM, YEAR_TO = 2019, 2025                      # 라벨(임상도) 비교 연도
AE_YEARS = list(range(2017, 2026))                   # AlphaEarth 연간 임베딩 가용 연도
AE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AE_BANDS = [f"A{i:02d}" for i in range(64)]

# ---- 데이터셋 파라미터
MIN_PIECE_HA = 0.1            # 중첩 조각(sliver) 제거 기준
INNER_BUFFER_M = 15           # 경계 오차 회피용 내부 버퍼 (AlphaEarth 10 m 1.5셀)
BLOCK_M = 5000                # 공간 교차검증 블록 크기
SAMPLES_PER_CLASS = {"no_change": 3000, "harvest": 1500, "conversion": 800, "gain": 800}
TEST_BLOCK_FRAC = 0.2         # 공간 홀드아웃 블록 비율
SEED = 42

GEE_PROJECT = os.environ.get("GEE_PROJECT", "ee-choish7838")
AGENT_MODEL = os.environ.get("FOREST_AGENT_MODEL", "claude-opus-5")

# ---- LLM 백엔드: gemini | claude  (키는 환경변수 GEMINI_API_KEY / ANTHROPIC_API_KEY)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini" if os.environ.get("GEMINI_API_KEY") else "claude")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODELS = [m for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash").split(",") if m]
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "4"))   # 모델당 재시도 (429/5xx)
