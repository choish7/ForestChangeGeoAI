"""의사결정 지원 도구: 수치·순위·지도는 모두 여기서 계산하고, LLM 은 도구 선택·해석·서술만 맡는다.

도구
  describe_data         보유 자료·컬럼·출처·기준시점·한계
  run_sql               근거 DB(SQLite) 읽기 전용 조회 (Text-to-SQL)
  rank_inspection       현장점검 우선순위 (가중치 + 민감도 분석)
  evaluate_foundation_model  AlphaEarth vs Sentinel-2 변화신호 비교 (행정기록·임상도·산지전용 고시 기준)
  make_map              소반·필지 지도 PNG
  write_report          보고서(md) + 근거기록, 보고서 수치의 도구출력 추적률 자동 점검
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# 서비스 자료는 이 폴더의 data/ 에서 읽는다 (sync_data.py 로 채움). 다른 위치를 쓰려면 환경변수 FDA_DATA_DIR.
DATA = Path(__import__("os").environ.get("FDA_DATA_DIR", Path(__file__).resolve().parent / "data"))
DB_DIR = DATA / "의사결정DB"
DB = DB_DIR / "forest_decision.sqlite"
GPKG = DB_DIR / "forest_decision.gpkg"
RUNS = Path(__file__).resolve().parent / "outputs"
REF = Path(__file__).resolve().parent / "reference_results.json"

COLUMNS = {
    "소반": {
        "단위ID": "국유림 임반/소반 (예 067-000/001-000)", "읍면동": "", "소재지": "", "면적_ha": "",
        "실적구분_2020_2025": "2020~2025 사업실적 구분: 수확·갱신 / 솎아베기 / 피해·기타 / 실적없음(NULL 포함)",
        "실적연도": "", "실적종별": "", "실적건수": "",
        "합의라벨": "경영실적과 임상도(2019→2025) 비교 결과", "임상도_변화비율": "임상도 2019→2025 변화 면적 비율",
        "AE_변화비율": "AlphaEarth 변화 화소 비율 (오탐률 5% 임계값)", "AE_변화면적_ha": "", "AE_거리평균": "연속연도 임베딩 거리 최댓값의 평균",
        "AE_변화연도": "AE 변화 화소의 최빈 변화연도(그 해 임베딩이 직전 해와 가장 달라짐)",
        "S2_변화비율": "Sentinel-2 연간 NDVI 최대감소 변화 화소 비율 (오탐률 5%)", "S2_변화면적_ha": "", "S2_dNDVI": "NDVI 2019−2025 (양수=감소)",
        "경사_도": "", "표고_m": "", "산불_500m_2019_2025": "", "산사태_500m_2019_2025": "",
        "임도거리_m": "소반 대표점→가장 가까운 임도", "도로거리_m": "소반 대표점→가장 가까운 도로(1:1,000)",
        "FM_벌채·갱신_ha": "학습 모델 최종지도(AlphaEarth+S2, 10 m, 산림 마스크·최소면적) 벌채·갱신 면적", "FM_산지전용_ha": "", "FM_산림증가_ha": "",
        "FM_변화면적_ha": "최종지도 변화 합계", "FM_변화확률": "최종지도 변화확률(1−무변화확률) 평균", "FM_주변화유형": "", "FM_변화연도": "최종지도 변화 화소의 최빈 변화연도",
        "MG_벌채·갱신_ha": "S2-TSFM(Sentinel-2 시계열 파운데이션 모델, 20 m, 산림 마스크) 벌채·갱신 면적", "MG_산지전용_ha": "", "MG_산림증가_ha": "",
        "MG_변화면적_ha": "", "MG_변화확률": "", "MG_주변화유형": "",
        "신호일치수": "AE 임베딩거리·S2-TSFM·최종지도 중 변화로 본 신호 수 0~3 (클수록 확실)", "판정_AE거리": "AE 임베딩거리만으로 본 변화/안정",
        "판정": "변화관측 = 최종지도 변화면적 ≥ 0.5 ha. 기록일치(수확실적·변화관측) / 미기록 변화(점검 필요) / 기타실적·변화관측(확인) / 실적있으나 변화 미관측(이행 확인) / 안정",
    },
    "산지전용": {
        "레코드ID": "", "고시번호": "보전산지 해제 고시", "고시연도": "", "주소": "", "읍면동": "", "지목": "", "전용용도": "",
        "해제면적_m2": "", "라벨": "전용확정(2020~24 고시·임상도 확인) / 전용후보 / 전용이전(2017~19 고시)",
        "임지비율_2019": "", "임지비율_2025": "", "AE_변화비율": "", "S2_변화비율": "", "AE_변화연도": "", "AE_화소수": "필지에 걸친 10 m 화소 수",
        "FM_변화면적_ha": "", "FM_산지전용_ha": "", "FM_주변화유형": "", "FM_변화연도": "", "MG_변화면적_ha": "", "MG_주변화유형": "",
    },
    "사업실적": {"단위ID": "", "연도": "2019~2025", "사업구분": "조림/숲가꾸기/임목생산/시설/소득/병해충방제및산림피해복구",
             "종별": "", "면적_ha": "", "사업비_천원": ""},
    "재해": {"유형": "산불/산사태", "연도": "2019~2025", "단위ID": "가장 가까운 소반", "소반거리_m": "", "x": "EPSG:5186", "y": ""},
    "소반_연별": {"단위ID": "", "연도": "2017~2025", "AE_연속거리": "전년→당해 임베딩 코사인 거리(변화 화소 또는 소반 평균)",
              "AE_고변화비율": "임계값을 넘는 화소 비율", "NDVI": "", "NDMI": "", "NDBI": "", "NDWI": "", "집계범위": "변화 화소 / 소반 전체"},
    "소반_경로": {"단위ID": "", "변화경로": "교란 후 회복 / 교란 후 회복 지연 / 지속 손실·전환 의심 / 최근 교란(회복 판단 이름) / 수분 스트레스 등 식생 이상 의심 / "
                                  "식생 증가·성장 (변화 신호 동반) / 임베딩 변화·지수 안정 / 식생 증가·성장 추정 / 안정",
              "변화구간": "임베딩 변화가 가장 큰 연간 구간", "NDVI_변화전": "", "NDVI_최저": "", "NDVI_2025": "", "회복률": "(2025−최저)/(변화전−최저)",
              "NDMI_감소": "", "NDBI_증가": "", "반복교란": "고변화 연도가 떨어져 2회 이상", "고변화연도": "",
              "임상_2019": "임상도 우세 속성", "수종_2019": "", "영급_2019": "", "임상_2025": "", "수종_2025": "", "영급_2025": "", "임상도_속성변화": "",
              "관측": "근거 문장", "기록": "", "해석": "", "추가확인": "", "불확실성": ""},
    "소반_AE분류": {"단위ID": "", "LC_{2019|2025}_{산림|나지|초지|시설지}_ha/_비율": "AlphaEarth 산지 피복 분류", "FT_{2019|2025}_{침엽수림|활엽수림|혼효림}_ha/_비율": "임상 분류",
                "AE_{무변화|벌채·갱신|산림증가}_ha": "변화유형 분류 2019→2025", "AE_우세임상_2019/2025": "", "AE_산림비율_변화": "", "AE_시설지나지_증가": "",
                "LC_2025_평균확률": "", "FT_2025_평균확률": "", "CH_2019_2025_평균확률": ""},
    "소반_임상도이력": {"단위ID": "", "조사연도": "임상도 2019·2024·2025판 갱신년도", "마지막조사_면적가중": "", "마지막조사_최근": "",
                  "연도별라벨유형": "연도:관측/유지추정/변화구간/미상", "변화구간": "조각별 전이(구간)", "입목변화_ha": "입목지↔무립목지 변화 면적",
                  "위성변화연도": "최종지도 또는 AE", "조사후_경과연수(2025기준)": "", "임상도_미반영변화": "위성 변화가 마지막 조사 이후",
                  "재조사_우선": "", "AE연도별_우세임상": "연도별 임상 추정의 우세 임상 전환"},
    "소반_연도별임상": {"단위ID": "", "연도": "2017~2025", "침엽수림_비율": "연도별 임상도 라벨로 학습한 AlphaEarth 모델 추정(조각 면적 가중)",
                  "활엽수림_비율": "", "혼효림_비율": "", "무립목지_비율": "", "표본조각": "", "우세": ""},
    "소반_임베딩": {"단위ID": "", "연도": "2019·2025", "A00~A63": "소반 평균 단위벡터(유사 지역 검색용)"},
    "임베딩_지수_관계": {"차원": "", "지수": "NDVI·NDMI·NDBI", "구분": "2025 수준 / 2019→2025 변화", "rho": "Spearman", "표본": ""},
}
SOURCES = {
    "AlphaEarth": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2017~2025 64차원, 연속연도 거리 (10 m)",
    "Sentinel-2": "COPERNICUS/S2_SR_HARMONIZED 연간 지수 2017~2025 (20 m, 54밴드)",
    "최종지도 모델": "AlphaEarth+S2 지수 HistGradientBoosting, 산림 마스크·최소면적 (final_change_2019_2025)",
    "S2-TSFM 모델": "Sentinel-2 시계열 마스크 오토인코더 + 분류기 (s2_tsfm_chuncheon)",
    "사업실적": "국유림 경영계획부 7·8·9차 산림사업실적 (춘천국유림관리소, 임소반 결합)",
    "임상도": "임상도 2019·2025 (51110)",
    "산지전용": "춘천시 보전산지 해제 고시 2017~2024 세부내역 + 연속지적도",
    "재해": "산림청 산불·산사태 신고정보 (춘천시 범위)",
    "접근성": "임도망도, 1:1,000 수치지형도 도로",
}
LIMITS = [
    "변화 신호는 '관측된 변화'이며 원인을 뜻하지 않는다. 실적·재해와의 관계는 공간적 동시발생일 뿐 인과가 아니다.",
    "AE_변화연도는 임베딩이 가장 크게 바뀐 해로, 실제 발생일과 1년 이상 다를 수 있다.",
    "학습 모델의 산지전용 정확도는 낮다(최종지도 F1 0.20, S2-TSFM 0.13). 산지전용 유형은 후보로만 본다.",
    "신호일치수 1 은 한 신호만 변화로 본 경우로 오탐 가능성이 크다.",
    "사업실적은 국유림 소반만 있다(사유림 사업 없음). 실적 위치는 소반 단위라 소반 일부 사업도 소반 전체로 연결된다.",
    "임계값(오탐률 5%)은 검증된 무변화 산림화소 기준이다. 큰 소반은 무변화여도 변화 화소가 일부 생긴다.",
    "산지전용 필지는 대부분 0.1 ha 미만이라 10 m 화소 몇 개로 판단된다(필지 경계 화소 포함).",
]


@dataclass
class Session:
    query: str
    run_dir: Path = None
    log: list = field(default_factory=list)
    highlight: list = field(default_factory=list)      # 웹 지도 강조용: make_map 호출마다 한 묶음
    t0: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.run_dir is None:
            self.run_dir = RUNS / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)


def _ok(data, warnings=None, sources=None):
    return {"data": data, "meta": {"warnings": warnings or [], "sources": sources or []}}


def _records(df, n=200):
    df = df.head(n).replace({np.nan: None})
    return json.loads(df.to_json(orient="records", force_ascii=False))


def describe_data(s: Session):
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        counts = {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in COLUMNS}
    meta = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    return _ok({"tables": {t: {"rows": counts[t], "columns": c} for t, c in COLUMNS.items()},
                "기준": {"기간": "2019→2025", **meta}, "출처": SOURCES, "한계": LIMITS}, sources=list(SOURCES.values()))


SQL_BAD = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I)


def run_sql(s: Session, sql: str, purpose: str = ""):
    q = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", q, re.I) or SQL_BAD.search(q) or ";" in q:
        raise ValueError("SELECT/WITH 단일 조회문만 허용됩니다.")
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        df = pd.read_sql(q, con)
    warn = [f"결과 {len(df)}행 중 200행만 반환"] if len(df) > 200 else []
    return _ok({"sql": q, "purpose": purpose, "n_rows": len(df), "rows": _records(df)}, warn, ["forest_decision.sqlite"])


DEFAULT_W = {"위성변화": 0.4, "기록없음": 0.3, "재해이력": 0.15, "접근용이": 0.15}


def _scores(u, w):
    comp = pd.DataFrame({
        "위성변화": (u["FM_변화면적_ha"].fillna(0) / u["면적_ha"]).clip(0, 1),
        "기록없음": u["실적구분_2020_2025"].isna() | u["실적구분_2020_2025"].eq("실적없음"),
        "재해이력": ((u["산불_500m_2019_2025"] + u["산사태_500m_2019_2025"]).clip(0, 3) / 3),
        "접근용이": 1 - u[["임도거리_m", "도로거리_m"]].min(axis=1).clip(0, 2000) / 2000,
    }).astype(float)
    tot = sum(w.values())
    return comp, sum(comp[k] * v / tot for k, v in w.items())


_GEO = {}


def _layer(name):
    """GPKG 레이어 캐시 (EPSG:5186)."""
    import geopandas as gpd
    if name not in _GEO:
        _GEO[name] = gpd.read_file(GPKG, layer=name)
    return _GEO[name]


def resolve_area(emd: str | None = None, unit_ids: list | None = None, bbox: list | None = None):
    """지역 지정 → (소반 단위ID 목록, 지역 도형 EPSG:5186, 설명). bbox 는 [서, 남, 동, 북] 경위도."""
    import geopandas as gpd
    from shapely.geometry import box
    su = _layer("소반")
    if unit_ids:
        sel = su[su["단위ID"].isin(unit_ids)]
        return list(sel["단위ID"]), sel.union_all(), f"선택 소반 {len(sel)}개"
    if emd:
        b = _layer("읍면동경계")
        g = b[b["EMD_NM"].eq(emd)]
        if g.empty:
            raise ValueError(f"읍면동 '{emd}' 없음. 가능: {sorted(b['EMD_NM'])}")
        return list(su.loc[su["읍면동"].eq(emd), "단위ID"]), g.union_all(), emd
    if bbox:
        geom = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(5186).iloc[0]
        return list(su.loc[su.intersects(geom), "단위ID"]), geom, "지도 범위 " + ", ".join(f"{v:.4f}" for v in bbox)
    return list(su["단위ID"]), None, "춘천시 전체"


def area_summary(s: Session, emd: str | None = None, unit_ids: list | None = None, bbox: list | None = None):
    ids, geom, label = resolve_area(emd, unit_ids, bbox)
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        u = pd.read_sql('select * from "소반"', con)
        r = pd.read_sql('select * from "사업실적"', con)
    u = u[u["단위ID"].isin(ids)]
    by_j = u.groupby("판정").agg(소반=("단위ID", "size"), 면적_ha=("면적_ha", "sum"), AE_변화면적_ha=("AE_변화면적_ha", "sum"),
                               S2_변화면적_ha=("S2_변화면적_ha", "sum")).round(1).reset_index()
    rec = r[r["단위ID"].isin(ids)].groupby("사업구분").agg(건수=("연도", "size"), 면적_ha=("면적_ha", "sum")).round(1).reset_index()
    yr = u["AE_변화연도"].dropna().astype(int).value_counts().sort_index()
    out = {"지역": label, "국유림_소반": len(u), "국유림_면적_ha": round(float(u["면적_ha"].sum()), 1),
           "AE_변화면적_ha": round(float(u["AE_변화면적_ha"].sum()), 1), "S2_변화면적_ha": round(float(u["S2_변화면적_ha"].sum()), 1),
           "판정별": _records(by_j), "사업실적_2019_2025": _records(rec),
           "AE_변화연도별_소반수": {int(k): int(v) for k, v in yr.items()},
           "모델_변화면적_ha": {m: {c: round(float(u[f"{m}_{c}_ha"].sum()), 1) for c in ("벌채·갱신", "산지전용", "산림증가")}
                            for m in ("FM", "MG")},
           "FM_변화연도별_소반수": {int(k): int(v) for k, v in u["FM_변화연도"].dropna().astype(int).value_counts().sort_index().items()},
           "신호일치수별_소반": {int(k): int(v) for k, v in u["신호일치수"].value_counts().sort_index().items()}}
    if geom is not None:   # 사유림 포함 자료: 산지전용 필지·재해는 지역 도형으로
        pc, dz = _layer("산지전용"), _layer("재해")
        pc = pc[pc.intersects(geom)]
        dz = dz[dz.within(geom)]
    else:
        pc, dz = _layer("산지전용"), _layer("재해")
    out["산지전용_필지"] = _records(pc.groupby("라벨").agg(필지=("레코드ID", "size"), 해제면적_m2=("해제면적_m2", "sum"),
                                                     AE_변화비율_중앙값=("AE_변화비율", "median")).round(3).reset_index())
    out["재해_2019_2025"] = {k: int(v) for k, v in dz["유형"].value_counts().items()}
    warn = [] if len(u) else ["지역 안에 국유림 소반이 없음 (사업실적·판정은 국유림만)"]
    return _ok(out, warn, ["forest_decision.sqlite", "forest_decision.gpkg"])


def rank_inspection(s: Session, top_n: int = 10, judgements: list | None = None, emd: str | None = None,
                    weights: dict | None = None, unit_ids: list | None = None):
    w = {**DEFAULT_W, **(weights or {})}
    if set(w) - set(DEFAULT_W):
        raise ValueError(f"가중치 키는 {list(DEFAULT_W)} 만 가능")
    judgements = judgements or ["미기록 변화(점검 필요)", "기타실적·변화관측(확인)"]
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        u = pd.read_sql('select * from "소반"', con)
    u = u[u["판정"].isin(judgements)]
    if emd:
        u = u[u["읍면동"].eq(emd)]
    if unit_ids:
        u = u[u["단위ID"].isin(unit_ids)]
    if u.empty:
        return _ok({"ranking": [], "n_candidates": 0}, ["조건에 맞는 소반 없음"])
    hi, _ = _year_reliability()
    yr = u["FM_변화연도"].astype("float")
    u = u.assign(연도신뢰도낮음=(yr.isin(hi) & u["신호일치수"].fillna(0).le(1)))
    comp, score = _scores(u, w)
    score = score * np.where(u["연도신뢰도낮음"], 0.7, 1.0)          # 오탐률 높은 해의 단독 신호는 30% 감점
    u = u.assign(점수=score.round(3), **{f"요인_{k}": comp[k].round(3) for k in comp})
    top = u.sort_values("점수", ascending=False).head(top_n)
    rng = np.random.default_rng(0)            # 민감도: 가중치 ±30% 무작위 200회
    hits = pd.Series(0, index=top.index)
    for _ in range(200):
        wp = {k: v * rng.uniform(0.7, 1.3) for k, v in w.items()}
        hits += _scores(u, wp)[1].nlargest(top_n).index.to_series().reindex(top.index).notna().astype(int)
    top = top.assign(순위안정성=(hits / 200).round(2))
    cols = ["단위ID", "읍면동", "소재지", "면적_ha", "판정", "점수", "순위안정성", "신호일치수", "FM_주변화유형", "FM_변화면적_ha", "FM_변화연도",
            "FM_변화확률", "연도신뢰도낮음", "MG_변화면적_ha", "AE_변화비율", "AE_변화면적_ha", "AE_변화연도",
            "S2_변화비율", "실적구분_2020_2025", "실적종별", "산불_500m_2019_2025", "산사태_500m_2019_2025", "임도거리_m", "도로거리_m",
            "경사_도"] + [c for c in top if c.startswith("요인_")]
    return _ok({"n_candidates": len(u), "weights": w, "judgements": judgements, "읍면동": emd,
                "ranking": _records(top[cols]), "민감도": "가중치 각각 ±30% 무작위 200회 중 상위 N 유지 비율"},
               ["점수는 점검 순서 제안이며 위법·원인 판단이 아님",
                f"변화연도가 AlphaEarth 오탐률이 높은 해({hi})이고 신호일치수 1 이하인 소반은 점수 30% 감점(연도신뢰도낮음)"], ["forest_decision.sqlite 소반"])


def evaluate_foundation_model(s: Session):
    from sklearn.metrics import average_precision_score, roc_auc_score
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        u = pd.read_sql('select * from "소반"', con)
        p = pd.read_sql('select * from "산지전용"', con)
    meta = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    pos = u["합의라벨"].isin(["확정변화(수확실적·임상도변화)", "수확외 실적·임상도변화", "임상도만 변화(미기록 변화 후보)"])
    neg = u["합의라벨"].eq("확정무변화(실적없음·변화없음)")
    e = u[pos | neg]
    y = pos[pos | neg].astype(int)
    unit = {}
    e = e.assign(FM_비율=e["FM_변화면적_ha"] / e["면적_ha"], MG_비율=e["MG_변화면적_ha"] / e["면적_ha"])
    for name, col, sign in [("AlphaEarth 임베딩거리 변화비율", "AE_변화비율", 1), ("Sentinel-2 NDVI 변화비율", "S2_변화비율", 1),
                            ("Sentinel-2 ΔNDVI", "S2_dNDVI", 1), ("모델: S2-TSFM 변화비율", "MG_비율", 1),
                            ("모델: 최종지도(AE+S2) 변화비율", "FM_비율", 1), ("모델: 최종지도 변화확률", "FM_변화확률", 1)]:
        v = (e[col] * sign).fillna(0)
        unit[name] = {"AUROC": round(float(roc_auc_score(y, v)), 3), "AP": round(float(average_precision_score(y, v)), 3)}
    rec = u[u["실적구분_2020_2025"].eq("수확·갱신")]
    parcels = {}
    for lab in ("전용확정", "전용후보", "전용이전"):
        q = p[p["라벨"].eq(lab) & p["AE_화소수"].gt(0)]
        parcels[lab] = {"필지": len(q), "AE_변화비율≥0.5": round(float((q["AE_변화비율"] >= 0.5).mean()), 3),
                        "S2_변화비율≥0.5": round(float((q["S2_변화비율"] >= 0.5).mean()), 3),
                        "최종지도_변화있음": round(float((q["FM_변화면적_ha"] > 0).mean()), 3),
                        "S2-TSFM_변화있음": round(float((q["MG_변화면적_ha"] > 0).mean()), 3)}
    ref = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else {}
    return _ok({
        "화소_검증(오탐률 5%에서 변화화소 탐지율)": {"AlphaEarth": meta["보정"]["AE"], "Sentinel-2": meta["보정"]["S2"]},
        "소반_검증(임상도 변화 확인 소반 vs 확정무변화 소반)": {"양성": int(y.sum()), "음성": int((1 - y).sum()), **unit},
        "수확·갱신 실적 소반의 변화관측 비율": {"소반": len(rec), "AE 임베딩거리": round(float(((rec["AE_변화면적_ha"] >= 0.5) & (rec["AE_변화비율"] >= 0.15)).mean()), 3),
                                          "최종지도": round(float((rec["FM_변화면적_ha"] >= 0.5).mean()), 3),
                                          "S2-TSFM": round(float(((rec["MG_변화면적_ha"] >= 0.5) & (rec["MG_변화면적_ha"] / rec["면적_ha"] >= 0.15)).mean()), 3)},
        "학습모델_홀드아웃(모델 보고서)": meta.get("모델", {}),
        "산지전용_고시필지(필지 화소 절반 이상이 변화)": parcels,
        "학습모델_비교(기존 실험)": ref,
    }, ["소반 검증의 양성은 임상도로 확인된 변화(59개 내외)로 표본이 작다",
        "학습 모델(S2-TSFM·최종지도)은 이 합의 라벨로 학습했으므로 소반 검증 값이 낙관적이다. 모델 간 비교는 '학습모델_홀드아웃' 값을 쓸 것. "
        "AE 임베딩거리·S2 지수는 라벨 없이 계산한 신호라 소반 검증이 공정하다","산지전용 전용이전 필지도 AE 변화가 높게 나올 수 있다(필지 경계 화소·주변 개발)"],
        ["build_meta.json", "forest_decision.sqlite", str(REF.name)])


AE_FILE = DATA / "satellite" / "Chuncheon_AE_{}_64Bands_int8.tif"
S2_FILE = DATA / "satellite" / "Chuncheon_S2_2017_2025_54Bands.tif"
FM_DIR, TSFM_DIR = DATA / "final_change_2019_2025", DATA / "s2_tsfm_chuncheon"
CLS_NAMES = {0: "무변화", 1: "벌채·갱신", 2: "산지전용", 3: "산림증가", 255: "산림 밖(마스크)"}


def _window(path, geom, bands=None):
    """도형 범위 창만 읽고, 도형 안 화소 마스크를 함께 돌려준다 (작은 도형은 닿는 화소 모두)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    with rasterio.open(path) as ds:
        g = gpd.GeoSeries([geom], crs=5186).to_crs(ds.crs).iloc[0]
        win = from_bounds(*g.buffer(ds.res[0]).bounds, transform=ds.transform).round_offsets().round_lengths()
        arr = ds.read(bands, window=win)
        small = g.area < 10 * ds.res[0] ** 2
        inside = ~geometry_mask([g], out_shape=arr.shape[-2:], transform=ds.window_transform(win), all_touched=small)
        return arr, inside


def unit_timeseries(s: Session, unit_id: str | None = None, record_id: str | None = None):
    """소반(또는 산지전용 필지)의 원자료 시계열과 모델 예측을 원본 래스터에서 그 자리에서 읽는다."""
    import rasterio
    if unit_id:
        g = _layer("소반")
        row = g[g["단위ID"].eq(unit_id)]
    elif record_id:
        g = _layer("산지전용")
        row = g[g["레코드ID"].eq(record_id)]
    else:
        raise ValueError("unit_id 또는 record_id 필요")
    if row.empty:
        raise ValueError("해당 소반/필지 없음")
    geom = row.geometry.iloc[0]
    years = list(range(2017, 2026))
    # Sentinel-2 54밴드: 연별 NDVI·NDMI 평균
    with rasterio.open(S2_FILE) as ds:
        idx = {d: i + 1 for i, d in enumerate(ds.descriptions)}
    a, m = _window(S2_FILE, geom, [idx[f"{k}_{y}"] for y in years for k in ("NDVI", "NDMI")])
    a = np.where(a == -9999, np.nan, a)
    mean = lambda v: round(float(np.nanmean(v)), 3) if np.isfinite(v).any() else None
    s2 = {y: {"NDVI": mean(a[2 * i][m]), "NDMI": mean(a[2 * i + 1][m])} for i, y in enumerate(years)}
    # AlphaEarth 64차원 int8: 연속연도 코사인 거리 평균
    embs = {}
    for y in years:
        e, me = _window(Path(str(AE_FILE).format(y)), geom)
        v = e.reshape(64, -1).T.astype("float32")[me.ravel()]
        ok = (v != -128).all(1) & (v != 0).any(1)
        v = np.where(ok[:, None], v / 127.0, np.nan)
        embs[y] = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos = lambda p, q: 1 - np.sum(p * q, axis=1)
    ae = {f"{y0}→{y1}": mean(cos(embs[y0], embs[y1])) for y0, y1 in zip(years[:-1], years[1:])}
    # 학습 모델 예측
    fc, mf = _window(FM_DIR / "final_change_class.tif", geom)
    fy, _ = _window(FM_DIR / "final_change_year.tif", geom)
    fp, _ = _window(FM_DIR / "final_change_prob.tif", geom)
    mc, mm = _window(TSFM_DIR / "change_class.tif", geom)
    mp, _ = _window(TSFM_DIR / "change_prob.tif", geom)
    counts = lambda v: {CLS_NAMES[int(k)]: int(c) for k, c in zip(*np.unique(v, return_counts=True))}
    prob = lambda p, mk: {n: mean(np.where(p[k] == -9999, np.nan, p[k])[mk]) for k, n in enumerate(["무변화", "벌채·갱신", "산지전용", "산림증가"])}
    fyv = fy[0][mf]
    out = {"대상": unit_id or record_id, "면적_ha": round(geom.area / 1e4, 3),
           "Sentinel-2_연별_평균": s2,
           "AlphaEarth_연속연도_임베딩거리": ae, "AlphaEarth_2019→2025_거리": mean(cos(embs[2019], embs[2025])),
           "AlphaEarth_최대변화구간": max((k for k in ae if ae[k] is not None), key=lambda k: ae[k], default=None),
           "최종지도(AE+S2)": {"화소_구성(10 m)": counts(fc[0][mf]),
                           "변화연도_화소": {int(k): int(c) for k, c in zip(*np.unique(fyv[fyv > 0], return_counts=True))},
                           "평균확률": prob(fp, mf)},
           "S2-TSFM": {"화소_구성(20 m, 마스크 전)": counts(mc[0][mm]), "평균확률": prob(mp, mm)}}
    return _ok(out, ["NDVI·임베딩 거리는 대상 전체 평균이라 일부 구역의 변화는 희석된다"],
               [S2_FILE.name, "Chuncheon_AE_<연도>_64Bands_int8.tif", FM_DIR.name, TSFM_DIR.name])


def make_map(s: Session, title: str, unit_ids: list | None = None, parcel_ids: list | None = None, filename: str = "map.png"):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    emd, su = _layer("읍면동경계"), _layer("소반")
    sel = su[su["단위ID"].isin(unit_ids or [])].set_index("단위ID").reindex(unit_ids or []).dropna(subset=["geometry"])
    pc = _layer("산지전용")
    psel = pc[pc["레코드ID"].isin(parcel_ids or [])]
    # 왼쪽: 춘천시 전체 위치(번호 원) / 오른쪽: 소반마다 확대 썸네일(노란 후광 + 빨간 경계, 주변 소반은 회색)
    n = len(sel)
    ncol = 2 if n > 4 else 1
    nrow = max(1, -(-min(n, 12) // ncol))
    fig = plt.figure(figsize=(15, max(7.5, 2.1 * nrow)))
    gs = fig.add_gridspec(nrow, 1 + ncol, width_ratios=[2.6] + [1] * ncol, wspace=0.05, hspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    su.plot(ax=ax, color="#e9f1e9", edgecolor="#c9d6c9", lw=0.2)
    emd.boundary.plot(ax=ax, color="#8a8a8a", lw=0.6)
    for _, r in emd.iterrows():
        c = r.geometry.representative_point()
        ax.text(c.x, c.y, r["EMD_NM"], fontsize=6, color="#777", ha="center")
    if len(psel):
        psel.centroid.plot(ax=ax, color="#1f5fbf", markersize=40, marker="s", edgecolor="white", lw=0.8, zorder=4)
    placed = []
    for i, (uid, r) in enumerate(sel.iterrows(), 1):
        c = r.geometry.representative_point()
        x, y = c.x, c.y
        for px, py in placed:          # 겹치는 번호는 살짝 비켜 놓고 지시선
            if abs(px - x) < 1500 and abs(py - y) < 1500:
                y += 1800
        if (x, y) != (c.x, c.y):
            ax.plot([c.x, x], [c.y, y], color="#ff3b1f", lw=1, zorder=4)
        placed.append((x, y))
        ax.scatter(x, y, s=300, color="#ff3b1f", edgecolor="white", lw=1.5, zorder=5)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8.5, weight="bold", color="white", zorder=6)
    ax.set_title("위치 (춘천시 전체, 번호 = 순위)", fontsize=11)
    ax.set_axis_off()
    for i, (uid, r) in enumerate(list(sel.iterrows())[:12]):
        t = fig.add_subplot(gs[i // ncol, 1 + i % ncol])
        x0, y0, x1, y1 = r.geometry.bounds
        pad = max(x1 - x0, y1 - y0) * 0.6 + 150
        win = su.cx[x0 - pad:x1 + pad, y0 - pad:y1 + pad]
        win.plot(ax=t, color="#eef3ee", edgecolor="#9aa89a", lw=0.4)
        gpd.GeoSeries([r.geometry]).boundary.plot(ax=t, color="#ffe600", lw=6, alpha=0.9)
        gpd.GeoSeries([r.geometry]).plot(ax=t, color="#ff3b1f", alpha=0.45, edgecolor="#b30000", lw=1.8)
        t.set_xlim(x0 - pad, x1 + pad)
        t.set_ylim(y0 - pad, y1 + pad)
        t.set_aspect("equal")
        t.set_xticks([]); t.set_yticks([])
        lab = f"{i + 1}. {uid}"
        if "읍면동" in r and isinstance(r["읍면동"], str):
            lab += f" ({r['읍면동']})"
        t.set_title(lab, fontsize=8.5, weight="bold", color="#b30000")
    if len(psel) and not n:
        x0, y0, x1, y1 = psel.total_bounds
        ax.set_xlim(x0 - 2000, x1 + 2000); ax.set_ylim(y0 - 2000, y1 + 2000)
    fig.suptitle(title, fontsize=14, weight="bold")
    shown = len(sel) + len(psel)
    out = s.run_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    s.highlight.append({"title": title, "unit_ids": list(sel.index), "parcel_ids": list(psel["레코드ID"]), "image": filename})
    return _ok({"path": str(out), "표시개체": shown})


PATHS = ["교란 후 회복", "교란 후 회복 지연", "지속 손실·전환 의심", "최근 교란(회복 판단 이름)", "수분 스트레스 등 식생 이상 의심",
         "식생 증가·성장 (변화 신호 동반)", "임베딩 변화·지수 안정", "식생 증가·성장 추정", "안정"]


def _q(sql, params=()):
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        return pd.read_sql(sql, con, params=params)


def explain_change(s: Session, unit_id: str):
    """임베딩 차원 해독 없이, 변화 시기(임베딩)·방향(지수)·지속성(연간 추이)·산림 맥락(임상도·실적·재해)을 엮어 설명."""
    p = _q('select * from "소반_경로" where "단위ID"=?', (unit_id,))
    if p.empty:
        raise ValueError("해당 소반 없음 (build_trajectory.py 실행 필요)")
    yr = _q('select "연도","AE_연속거리","AE_고변화비율","NDVI","NDMI","NDBI" from "소반_연별" where "단위ID"=? order by "연도"', (unit_id,))
    r = p.iloc[0]
    cls = ""
    try:
        a = _q('select * from "소반_AE분류" where "단위ID"=?', (unit_id,)).iloc[0]
        cls = (f"AlphaEarth 분류모델: 산림 비율 {a['LC_2019_산림_비율']}→{a['LC_2025_산림_비율']}, 2025 나지 {a['LC_2025_나지_비율']}·시설지(제지) {a['LC_2025_시설지_비율']}; "
               f"우세 임상 {a['AE_우세임상_2019']}→{a['AE_우세임상_2025']}(임상도 {r['임상_2019']}→{r['임상_2025']}); "
               f"변화유형 벌채·갱신 {a['AE_벌채·갱신_ha']} ha, 산림증가 {a['AE_산림증가_ha']} ha.")
    except Exception:
        pass
    hist = ""
    try:
        hh = _q('select * from "소반_임상도이력" where "단위ID"=?', (unit_id,)).iloc[0]
        hist = (f" 임상도 조사연도 {hh['조사연도']}(마지막 {hh['마지막조사_최근']}년)"
                + (f", 입목 변화 구간: {hh['변화구간']}" if isinstance(hh["변화구간"], str) else "")
                + (f"; AE 연도별 임상 추정: {hh['AE연도별_우세임상']}" if isinstance(hh["AE연도별_우세임상"], str) else "") + ".")
    except Exception:
        pass
    hi, rate = _year_reliability()
    unc_y = (f"; 변화구간 끝 연도 {r['정점연도']}년은 AlphaEarth 연도별 오탐률이 높은 해(안정 조각 {rate.get(str(int(r['정점연도'])))})"
             if pd.notna(r["정점연도"]) and int(r["정점연도"]) in hi else "")
    return _ok({"소반": unit_id, "변화경로": r["변화경로"], "변화구간": r["변화구간"], "반복교란": bool(r["반복교란"]),
                "근거": {"관측": r["관측"] + (" " + cls if cls else ""), "기록": r["기록"] + hist, "해석": r["해석"] + " (‘부합’ 수준의 설명이며 원인 확정이 아님)",
                       "추가확인": r["추가확인"], "불확실성": r["불확실성"] + unc_y},
                "연간추이": _records(yr.round(3)),
                "임상도": {"2019": [r["임상_2019"], r["수종_2019"], r["영급_2019"], r["밀도_2019"]],
                        "2025": [r["임상_2025"], r["수종_2025"], r["영급_2025"], r["밀도_2025"]], "속성변화": bool(r["임상도_속성변화"])}},
               ["임베딩 개별 차원에 물리적 의미를 부여하지 않음 — 거리(변화 크기)만 사용",
                "임상도가 그대로라는 사실만으로 무변화로 보지 않음(갱신 지연 가능), 임상도 변경도 판독 수정일 수 있음"],
               ["소반_경로", "소반_연별"])


def find_trajectory(s: Session, path: str | None = None, repeated: bool | None = None, emd: str | None = None,
                    unit_ids: list | None = None, top_n: int = 20):
    """변화경로(회복·회복지연·지속손실·수분이상 등)나 반복교란으로 소반 찾기."""
    if path and path not in PATHS:
        raise ValueError(f"path 는 {PATHS} 중 하나")
    t = _q('select p.*, u."읍면동", u."면적_ha", u."판정", u."FM_변화면적_ha", u."임도거리_m", u."경사_도" '
           'from "소반_경로" p join "소반" u using("단위ID")')
    if path:
        t = t[t["변화경로"].eq(path)]
    if repeated is not None:
        t = t[t["반복교란"].astype(bool) == repeated]
    if emd:
        t = t[t["읍면동"].eq(emd)]
    if unit_ids:
        t = t[t["단위ID"].isin(unit_ids)]
    t = t.sort_values(["NDVI_감소", "FM_변화면적_ha"], ascending=False)
    cnt = t["변화경로"].value_counts().to_dict()
    cols = ["단위ID", "읍면동", "면적_ha", "변화경로", "변화구간", "NDVI_변화전", "NDVI_최저", "NDVI_2025", "회복률", "NDMI_감소", "NDBI_증가",
            "반복교란", "고변화연도", "판정", "FM_변화면적_ha", "임상_2019", "수종_2019", "영급_2019", "임도거리_m", "경사_도"]
    return _ok({"조건": {"path": path, "repeated": repeated, "emd": emd}, "해당소반": len(t), "경로별": cnt,
                "목록": _records(t[cols].head(top_n)), "기준": json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8")).get("경로_기준")},
               ["경로 판정은 잠정 기준값에 따른 근거 정리이며 현장 확인 전 변화 원인을 뜻하지 않음"], ["소반_경로"])


def find_similar_units(s: Session, reference_ids: list, year: int = 2025, top_n: int = 10, same_forest_type: bool = False,
                       exclude_changed: bool = False):
    """기준 소반(예: 회복 양호지)의 평균 임베딩과 코사인 유사도가 높은 소반을 찾고 지형·임상·이력 차이를 함께 돌려준다."""
    E = _q('select * from "소반_임베딩" where "연도"=?', (year,)).set_index("단위ID").drop(columns="연도")
    ref = [i for i in reference_ids if i in E.index]
    if not ref:
        raise ValueError("기준 소반이 없음")
    v = E.loc[ref].mean().to_numpy()
    v = v / np.linalg.norm(v)
    sim = pd.Series(E.fillna(0).to_numpy() @ v, index=E.index, name="유사도")
    t = _q('select u."단위ID", u."읍면동", u."면적_ha", u."경사_도", u."표고_m", u."임도거리_m", u."판정", u."실적구분_2020_2025", '
           'p."변화경로", p."임상_2025", p."수종_2025", p."영급_2025" from "소반" u join "소반_경로" p using("단위ID")').set_index("단위ID")
    t = t.join(sim).drop(index=ref, errors="ignore")
    rb = _q('select u."단위ID", u."경사_도", u."표고_m", p."임상_2025", p."수종_2025", p."변화경로" from "소반" u join "소반_경로" p using("단위ID")')
    rb = rb[rb["단위ID"].isin(ref)]
    if same_forest_type and rb["임상_2025"].notna().any():
        t = t[t["임상_2025"].isin(rb["임상_2025"].dropna().unique())]
    if exclude_changed:
        t = t[t["판정"].eq("안정")]
    t = t.sort_values("유사도", ascending=False).head(top_n).reset_index()
    t["경사차"] = (t["경사_도"] - rb["경사_도"].mean()).round(1)
    t["표고차"] = (t["표고_m"] - rb["표고_m"].mean()).round(0)
    return _ok({"기준소반": _records(rb.round(2)), "연도": year, "후보": _records(t.round(3))},
               ["임베딩 유사도는 관측된 지표 특성의 유사성이며 기후·토양·생태 기능이 같다는 뜻이 아님 — 비교·검증할 후보로만 사용",
                "경사·표고·임상 차이를 함께 보고 현장 조건을 재검토할 것"], ["소반_임베딩", "소반"])


def map_update_candidates(s: Session, emd: str | None = None, top_n: int = 20):
    """임상도와 위성 관측의 불일치: (A) 위성 변화 있는데 2025 임상도는 그대로 → 갱신 검토, (B) 임상도는 바뀌었는데 위성 신호 없음 → 판독 수정 가능."""
    t = _q('select u.*, p."변화경로", p."임상_2019", p."수종_2019", p."임상_2025", p."수종_2025", p."임상도_속성변화" '
           'from "소반" u join "소반_경로" p using("단위ID")')
    if emd:
        t = t[t["읍면동"].eq(emd)]
    obs = t["FM_변화면적_ha"].ge(0.5) & t["신호일치수"].ge(2)
    a = t[obs & t["임상도_변화비율"].fillna(0).lt(0.05) & ~t["임상도_속성변화"].astype(bool)]
    b = t[t["임상도_변화비율"].fillna(0).ge(0.2) & t["신호일치수"].eq(0)]
    cols = ["단위ID", "읍면동", "면적_ha", "변화경로", "FM_주변화유형", "FM_변화면적_ha", "FM_변화연도", "신호일치수", "임상도_변화비율",
            "임상_2019", "수종_2019", "임상_2025", "수종_2025", "실적구분_2020_2025"]
    c = d1 = d2 = pd.DataFrame()
    try:
        ae = _q('select * from "소반_AE분류"')
        tt = t.merge(ae, on="단위ID")
        prop = tt[[f"FT_2025_{l}_비율" for l in AE_FT]].max(1)
        c = tt[tt["AE_우세임상_2025"].notna() & tt["임상_2025"].isin(AE_FT) & (tt["AE_우세임상_2025"] != tt["임상_2025"])
               & (prop >= 0.6) & (tt["FT_2025_평균확률"] >= 0.6)]
        fac = tt["LC_2025_시설지_비율"] - tt["LC_2019_시설지_비율"]
        bare = tt["LC_2025_나지_비율"] - tt["LC_2019_나지_비율"]
        tt = tt.assign(시설지_증가=fac.round(3), 나지_증가=bare.round(3))
        d1 = tt[(fac >= 0.05) & (tt["면적_ha"] >= 1) & (tt["LC_2025_산림_비율"] <= tt["LC_2019_산림_비율"])]   # 제지 확대(산림 비율이 늘지 않은 곳만: 미립목지 회복지 제외)
        d2 = tt[(bare >= 0.10) & (tt["면적_ha"] >= 1) & ~tt["실적구분_2020_2025"].eq("수확·갱신")]   # 수확 기록 없는 미립목지화
    except Exception:
        pass
    ccols = ["단위ID", "읍면동", "면적_ha", "임상_2025", "AE_우세임상_2025", "FT_2025_평균확률", "임상도_변화비율"]
    dcols = ["단위ID", "읍면동", "면적_ha", "LC_2019_산림_비율", "LC_2025_산림_비율", "LC_2025_시설지_비율", "LC_2025_나지_비율", "시설지_증가", "나지_증가",
             "실적구분_2020_2025", "변화경로"]
    return _ok({"A_임상도_갱신검토(위성 변화·임상도 유지)": {"소반": len(a), "목록": _records(a.sort_values("FM_변화면적_ha", ascending=False)[cols].head(top_n))},
                "B_판독수정_가능(임상도 변경·위성 무변화)": {"소반": len(b), "목록": _records(b.sort_values("임상도_변화비율", ascending=False)[cols].head(top_n))},
                "C_임상_재판독(AE 분류 우세임상 ≠ 2025 임상도)": {"소반": len(c), "목록": _records(c.sort_values("FT_2025_평균확률", ascending=False)[ccols].head(top_n)) if len(c) else []},
                "D1_산림외용도(제지)_확대(AE 분류 시설지 +5%p 이상)": {"소반": len(d1), "목록": _records(d1.sort_values("시설지_증가", ascending=False)[dcols].head(top_n)) if len(d1) else []},
                "D2_수확기록없는_미립목지화(AE 분류 나지 +10%p 이상)": {"소반": len(d2), "목록": _records(d2.sort_values("나지_증가", ascending=False)[dcols].head(top_n)) if len(d2) else []}},
               ["A는 임상도 갱신 지연, B는 판독 기준·경계 수정 또는 미세한 변화일 수 있음 — 어느 쪽도 오류로 단정하지 않음",
                "C·D는 AlphaEarth 분류모델 결과(임상 구분 F1 0.65, 시설지는 제지 의미) — 판독·현장 확인 대상 후보"], ["소반", "소반_경로", "소반_AE분류"])


def change_zones(s: Session, change_class: str | None = None, emd: str | None = None, unit_ids: list | None = None,
                 min_ha: float = 0.5, top_n: int = 30):
    """학습모형 변화 화소를 연속 구역(패치)으로 묶은 결과를 소반과 중첩해 점검 구역·면적표를 만든다."""
    import geopandas as gpd
    if "패치" not in _GEO:
        p = gpd.read_file(FM_DIR / "final_change_patches.gpkg").to_crs(5186)
        _GEO["패치"] = p
    p = _GEO["패치"]
    if change_class:
        p = p[p["class_ko"].eq(change_class)]
    p = p[p["area_ha"] >= min_ha]
    su = _layer("소반")
    if emd or unit_ids:
        ids, geom, _ = resolve_area(emd=emd, unit_ids=unit_ids)
        p = p[p.intersects(geom)]
    x = gpd.overlay(p[["patch_id", "class_ko", "area_ha", "change_year", "mean_prob", "geometry"]], su[["단위ID", "geometry"]],
                    how="intersection", keep_geom_type=True)
    x["중첩_ha"] = x.area / 1e4
    per = x.groupby("patch_id").agg(소반수=("단위ID", "nunique"), 국유림내_ha=("중첩_ha", "sum"), 소반=("단위ID", lambda v: ",".join(sorted(set(v)))))
    z = p.set_index("patch_id").join(per).fillna({"소반수": 0, "국유림내_ha": 0})
    z["국유림외_ha"] = (z["area_ha"] - z["국유림내_ha"]).clip(lower=0)
    z["경계확인"] = np.where(z["소반수"] >= 2, "여러 소반에 걸침", np.where(z["국유림외_ha"] > 0.05, "국유림 경계 밖으로 이어짐", ""))
    z = z.reset_index().sort_values("area_ha", ascending=False)
    by_unit = x.groupby("단위ID").agg(패치수=("patch_id", "nunique"), 변화구역_ha=("중첩_ha", "sum")).sort_values("변화구역_ha", ascending=False)
    return _ok({"패치수": len(z), "면적_ha": round(float(z["area_ha"].sum()), 1), "유형별_ha": z.groupby("class_ko")["area_ha"].sum().round(1).to_dict(),
                "경계확인_패치": int((z["경계확인"] != "").sum()),
                "구역": _records(z[["patch_id", "class_ko", "area_ha", "change_year", "mean_prob", "소반수", "국유림내_ha", "국유림외_ha", "소반", "경계확인"]].round(3).head(top_n)),
                "소반별": _records(by_unit.round(2).reset_index().head(top_n))},
               ["10 m 격자는 분석 해상도이며 경계 정확도가 10 m 라는 뜻이 아님", f"{min_ha} ha 미만 패치는 제외(최소면적)"],
               ["final_change_patches.gpkg", "소반"])


def embedding_relations(s: Session, index: str = "NDVI", kind: str = "2019→2025 변화", top_n: int = 8):
    """임베딩 차원과 지수의 Spearman 상관(탐색). '관련됨'까지만 말하고 차원에 이름을 붙이지 않는다."""
    r = _q('select * from "임베딩_지수_관계" where "지수"=? and "구분"=?', (index, kind))
    if r.empty:
        raise ValueError("지수는 NDVI·NDMI·NDBI, 구분은 '2025 수준' 또는 '2019→2025 변화'")
    r = r.assign(절대=r["rho"].abs()).sort_values("절대", ascending=False)
    return _ok({"지수": index, "구분": kind, "표본": int(r["표본"].iloc[0]), "상위차원": _records(r[["차원", "rho"]].head(top_n)),
                "|rho|≥0.5_차원수": int((r["절대"] >= 0.5).sum())},
               ["상관은 춘천 산림 화소 3,000개 표본의 단조 관계이며, 'A17은 수분 변수'처럼 차원에 의미를 명명하는 근거가 아님",
                "Sentinel 지수와 임베딩은 관측정보를 공유할 수 있어, 일치가 독립적 변화 검증을 대신하지 못함"], ["임베딩_지수_관계"])


AE_LC, AE_FT, AE_CH = ["산림", "나지", "초지", "시설지"], ["침엽수림", "활엽수림", "혼효림"], ["무변화", "벌채·갱신", "산림증가"]
AE_CAVEAT = ["AlphaEarth 분류모델 라벨은 임상도 속성만 사용(산림=입목지, 나지=미립목지, 초지=관목덤불·초지, 시설지=제지·주거지)",
             "초지 F1 0.11 — 초지 결과는 판단 근거로 쓰지 않음", "시설지는 임도·묘지·송전탑 등 산지 안 산림 외 용도이며 시가지 건물이 아님",
             "임상 구분 macro-F1 0.65(혼효림 0.57) — 임상 불일치는 판독 대상 후보일 뿐 오류 확정이 아님",
             "변화유형 모델은 산지전용을 제외하고 무변화·벌채·갱신·산림증가만 구분(macro-F1 0.91)"]


def ae_forest_status(s: Session, emd: str | None = None, unit_ids: list | None = None, bbox: list | None = None):
    """AlphaEarth 분류모델 3종으로 본 지역의 산지 피복·임상(2019→2025)과 변화유형 면적, 임상도와의 일치."""
    ids, _, label = resolve_area(emd, unit_ids, bbox)
    a = _q('select a.*, p."임상_2019", p."임상_2025", u."면적_ha" from "소반_AE분류" a join "소반_경로" p using("단위ID") join "소반" u using("단위ID")')
    a = a[a["단위ID"].isin(ids)]
    if a.empty:
        return _ok({"지역": label, "소반": 0}, ["지역 안에 국유림 소반이 없음"])
    lc = {y: {l: round(float(a[f"LC_{y}_{l}_ha"].sum()), 1) for l in AE_LC} for y in (2019, 2025)}
    ft = {y: {l: round(float(a[f"FT_{y}_{l}_ha"].sum()), 1) for l in AE_FT} for y in (2019, 2025)}
    ch = {l: round(float(a[f"AE_{l}_ha"].sum()), 1) for l in AE_CH[1:]}
    agree = {}
    for y in (2019, 2025):
        m = a[f"AE_우세임상_{y}"].notna() & a[f"임상_{y}"].isin(AE_FT)
        agree[y] = {"비교소반": int(m.sum()), "우세임상_일치율": round(float((a.loc[m, f"AE_우세임상_{y}"] == a.loc[m, f"임상_{y}"]).mean()), 3) if m.any() else None}
    return _ok({"지역": label, "소반": len(a), "피복_ha": lc, "임상_ha": ft, "변화유형_ha(2019→2025)": ch,
                "임상도와_우세임상_일치": agree,
                "평균확률": {k: round(float(a[f"{k}_평균확률"].mean()), 3) for k in ("LC_2025", "FT_2025", "CH_2019_2025")}},
               AE_CAVEAT, ["소반_AE분류", "ae_class_maps"])


def _year_reliability():
    m = json.loads((DB_DIR / "build_meta.json").read_text(encoding="utf-8")).get("연도신뢰도", {})
    return m.get("AE_고오탐연도", []), m.get("AE_오탐률", {})


def survey_history(s: Session, unit_id: str):
    """소반의 임상도 조사 이력(조사연도, 연도별 관측·유지·변화구간, 입목 변화 구간)과 연도별 임상도 라벨로 학습한 AlphaEarth 연도별 임상 추정."""
    h = _q('select * from "소반_임상도이력" where "단위ID"=?', (unit_id,))
    if h.empty:
        raise ValueError("해당 소반 없음 (build_yearly_history.py 실행 필요)")
    y = _q('select * from "소반_연도별임상" where "단위ID"=? order by "연도"', (unit_id,))
    r = h.iloc[0]
    hi, rate = _year_reliability()
    warn = ["연도별 임상 추정은 임상도 연도별 A등급 라벨로 학습한 AlphaEarth 모델(임상 3클래스 macro-F1 0.58~0.68, 블록 검증) 결과",
            "다년 특징을 써서 갑작스러운 벌채의 약 30%는 다음 해 이후 늦게 반영됨 — 급변 판단은 변화 모델(최종지도·변화유형) 우선",
            "소반 표본 조각이 적으면(표본조각 수 확인) 소반 전체를 대표하지 못함"]
    sy = r["위성변화연도"]
    if pd.notna(sy) and int(sy) in hi:
        warn.append(f"위성 변화연도 {int(sy)}년은 AlphaEarth 연도별 오탐률이 높은 해(안정 조각 {rate.get(str(int(sy)))}) — 다른 신호와 함께 확인 필요")
    return _ok({"소반": unit_id, "조사연도": r["조사연도"], "마지막조사": {"면적가중": r["마지막조사_면적가중"], "최근": r["마지막조사_최근"]},
                "조사후_경과연수": r["조사후_경과연수(2025기준)"], "연도별라벨유형": r["연도별라벨유형"], "임상도_변화구간": r["변화구간"],
                "입목변화_ha": r["입목변화_ha"], "위성변화연도": sy, "임상도_미반영변화": bool(r["임상도_미반영변화"]), "재조사_우선": r["재조사_우선"],
                "AE_연도별_임상추정": _records(y.round(3)), "AE_우세임상_전환": r["AE연도별_우세임상"]},
               warn + ["임상도 '유지추정' 구간은 갱신되지 않았다는 뜻이며 확인된 무변화가 아님", "임상 간 전이는 판독 수정일 가능성이 큼"],
               ["소반_임상도이력", "소반_연도별임상", "임상도 2019·2024·2025"])


def resurvey_priority(s: Session, emd: str | None = None, unit_ids: list | None = None, top_n: int = 15):
    """임상도 재조사 우선순위: 마지막 조사 이후 위성 변화(임상도 미반영) > AE 연도별 임상 우세 전환 > 조사 후 오래 경과."""
    t = _q('select h.*, u."읍면동", u."면적_ha", u."FM_변화면적_ha", u."FM_주변화유형", u."신호일치수", u."판정" '
           'from "소반_임상도이력" h join "소반" u using("단위ID")')
    if emd:
        t = t[t["읍면동"].eq(emd)]
    if unit_ids:
        t = t[t["단위ID"].isin(unit_ids)]
    hi, _ = _year_reliability()
    sw = t["AE연도별_우세임상"].fillna("").str.count("→")
    lowrel = t["위성변화연도"].isin(hi) & t["신호일치수"].fillna(0).le(1)
    t = t.assign(우세전환수=sw, 연도신뢰도낮음=lowrel,
                 점수=(t["임상도_미반영변화"].astype(int) * 3 + (sw > 0).astype(int) * 1.5
                     + t["조사후_경과연수(2025기준)"].fillna(0).clip(0, 10) / 10 + (t["FM_변화면적_ha"].fillna(0) / t["면적_ha"]).clip(0, 1)
                     - lowrel.astype(int)).round(2))
    t = t[t["점수"] > 0.5].sort_values("점수", ascending=False)
    cols = ["단위ID", "읍면동", "면적_ha", "점수", "재조사_우선", "마지막조사_최근", "조사후_경과연수(2025기준)", "위성변화연도", "FM_주변화유형",
            "FM_변화면적_ha", "신호일치수", "연도신뢰도낮음", "우세전환수", "AE연도별_우세임상", "변화구간"]
    return _ok({"후보": len(t), "목록": _records(t[cols].head(top_n)),
                "점수": "임상도 미반영 위성변화 3 + AE 연도별 우세임상 전환 1.5 + 조사 경과(10년=1) + 최종지도 변화비율 − 고오탐 연도 단독신호 1"},
               ["재조사 우선순위는 판독·현장 확인 순서 제안이며 임상도 오류 확정이 아님"], ["소반_임상도이력", "소반"])


NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def write_report(s: Session, title: str, body_markdown: str):
    tool_text = json.dumps([e["output"] for e in s.log], ensure_ascii=False, default=str)
    tool_nums = {n.replace(",", "") for n in NUM.findall(tool_text)}
    body_nums = [n.replace(",", "") for n in NUM.findall(re.sub(r"\d{3}-\d{3}/\d{3}-\d{3}", "", body_markdown))]
    body_nums = [n for n in body_nums if not re.fullmatch(r"(19|20)\d\d|\d", n)]   # 연도·한 자리 순번 제외
    def found(n):
        if n in tool_nums:
            return True
        try:
            f = float(n)
        except ValueError:
            return False
        return any(abs(float(t) - f) < 1e-9 or (abs(f) >= 1 and str(round(float(t), 1)) == n) or str(round(float(t) * 100, 1)).rstrip("0").rstrip(".") == n
                   for t in tool_nums if re.fullmatch(r"-?\d+\.?\d*", t))
    miss = sorted({n for n in body_nums if not found(n)})
    trace = round(1 - len(miss) / max(len(set(body_nums)), 1), 3)
    evidence = [{"step": i + 1, "tool": e["tool"], "input": e["input"], "seconds": e["seconds"]} for i, e in enumerate(s.log)]
    md = (f"# {title}\n\n> 질의: {s.query}\n\n{body_markdown}\n\n---\n## 근거 기록\n"
          f"- 수치 추적률: {trace} (보고서 수치 {len(set(body_nums))}개 중 도구 출력에서 확인되지 않은 값: {miss or '없음'})\n"
          f"- 자료 출처: " + "; ".join(f"{k} — {v}" for k, v in SOURCES.items()) + "\n"
          + "".join(f"- {e['step']}. `{e['tool']}` {json.dumps(e['input'], ensure_ascii=False)[:300]}\n" for e in evidence))
    out = s.run_dir / "report.md"
    out.write_text(md, encoding="utf-8")
    if s.highlight:
        (s.run_dir / "highlight.json").write_text(json.dumps({"groups": s.highlight}, ensure_ascii=False), encoding="utf-8")
    (s.run_dir / "evidence.json").write_text(json.dumps({"query": s.query, "log": s.log}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return _ok({"path": str(out), "수치_추적률": trace, "미확인_수치": miss},
               ["미확인 수치가 있으면 도구 출력값으로 고쳐 다시 작성할 것"] if miss else [])


def session_stats(s: Session):
    """GeoAI 업무 가치 검증용: 이 질의를 처리하는 데 든 시간·도구·자료 결합 현황."""
    used = [e["tool"] for e in s.log]
    sqls = [e["input"].get("sql", "") for e in s.log if e["tool"] == "run_sql"]
    tables = sorted({t for q in sqls for t in COLUMNS if f'"{t}"' in q})
    return _ok({"경과_초": round(time.time() - s.t0, 1), "도구_호출수": len(used), "도구별": {t: used.count(t) for t in dict.fromkeys(used)},
                "SQL_조회수": len(sqls), "SQL에서_결합한_테이블": tables,
                "자동화된_단계": ["자료 조회", "공간 결합(소반·재해·필지)", "기록-관측 대조 판정", "우선순위·민감도 계산", "지도 작성", "보고서·근거기록"],
                "결합된_자료원": list(SOURCES)},
               ["경과 시간은 이 세션의 도구·LLM 처리 시간이며, 수작업 소요시간 비교값은 아님"])


TOOLS = {"describe_data": describe_data, "run_sql": run_sql, "area_summary": area_summary, "rank_inspection": rank_inspection,
         "unit_timeseries": unit_timeseries, "explain_change": explain_change, "find_trajectory": find_trajectory,
         "find_similar_units": find_similar_units, "map_update_candidates": map_update_candidates, "change_zones": change_zones,
         "embedding_relations": embedding_relations, "ae_forest_status": ae_forest_status,
         "survey_history": survey_history, "resurvey_priority": resurvey_priority,
         "evaluate_foundation_model": evaluate_foundation_model, "make_map": make_map, "write_report": write_report,
         "session_stats": session_stats}

TOOL_SCHEMAS = [
    {"name": "describe_data", "description": "보유 자료의 테이블·컬럼 설명, 변화 임계값, 출처, 해석 한계를 돌려준다. SQL 작성 전에 먼저 호출.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_sql", "description": "근거 DB(SQLite: 소반, 산지전용, 사업실적, 재해)를 읽기 전용 SELECT 로 조회. 한국어 컬럼명은 큰따옴표로 감싼다. 최대 200행 반환.",
     "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}, "purpose": {"type": "string", "description": "이 조회로 확인하려는 것"}},
                      "required": ["sql"]}},
    {"name": "area_summary", "description": "지역(읍면동, 소반 목록, 또는 경위도 범위) 요약: 판정별 소반·면적, AE/S2 변화면적, 사업실적, 변화연도, 산지전용 필지, 재해. 지역을 지정하지 않으면 춘천시 전체.",
     "input_schema": {"type": "object", "properties": {
         "emd": {"type": "string", "description": "읍면동명 (예: 북산면)"},
         "unit_ids": {"type": "array", "items": {"type": "string"}, "description": "소반 단위ID 목록"},
         "bbox": {"type": "array", "items": {"type": "number"}, "description": "[서경도, 남위도, 동경도, 북위도]"}}}},
    {"name": "rank_inspection", "description": "현장점검 우선순위. 판정이 변화관측인 소반을 위성변화(최종지도 변화비율)·기록없음·재해이력·접근용이 가중합으로 정렬하고, 가중치 민감도(순위안정성)와 신호일치수를 함께 돌려준다.",
     "input_schema": {"type": "object", "properties": {
         "top_n": {"type": "integer", "default": 10},
         "unit_ids": {"type": "array", "items": {"type": "string"}, "description": "이 소반들로 한정 (지역 질의)"},
         "judgements": {"type": "array", "items": {"type": "string"}, "description": "대상 판정 (기본: 미기록 변화(점검 필요), 기타실적·변화관측(확인))"},
         "emd": {"type": "string", "description": "읍면동 한정 (예: 북산면)"},
         "weights": {"type": "object", "description": "위성변화·기록없음·재해이력·접근용이 가중치 (기본 0.4/0.3/0.15/0.15)"}}}},
    {"name": "unit_timeseries", "description": "소반(unit_id) 또는 산지전용 필지(record_id)의 원자료 시계열과 모델 예측: Sentinel-2 NDVI·NDMI 2017~2025, AlphaEarth 연속연도 임베딩 거리, 최종지도·S2-TSFM 유형·확률·변화연도. 후보의 변화 시점·근거 확인용 (호출당 수 초).",
     "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}, "record_id": {"type": "string"}}}},
    {"name": "survey_history", "description": "소반의 임상도 조사 이력(2019·2024·2025판 갱신년도, 연도별 관측·유지추정·변화구간, 입목 변화 구간)과 연도별 임상도 라벨로 학습한 AlphaEarth 연도별 임상 추정, 위성 변화가 마지막 조사 이후인지.",
     "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]}},
    {"name": "resurvey_priority", "description": "임상도 재조사 우선순위: 마지막 조사 이후 위성 변화(임상도 미반영), AE 연도별 임상 우세 전환, 조사 후 경과연수, 변화비율. 오탐률이 높은 해의 단독 신호는 감점.",
     "input_schema": {"type": "object", "properties": {"emd": {"type": "string"}, "unit_ids": {"type": "array", "items": {"type": "string"}}, "top_n": {"type": "integer"}}}},
    {"name": "ae_forest_status", "description": "AlphaEarth 분류모델 3종(산지 피복 산림·나지·초지·시설지, 임상 침엽·활엽·혼효, 변화유형 벌채·갱신·산림증가)으로 본 지역 구성(2019→2025)과 임상도 우세임상 일치율.",
     "input_schema": {"type": "object", "properties": {"emd": {"type": "string"}, "unit_ids": {"type": "array", "items": {"type": "string"}},
                                                        "bbox": {"type": "array", "items": {"type": "number"}}}}},
    {"name": "explain_change", "description": "소반 변화를 근거로 설명: 임베딩 변화 시기·크기, NDVI·NDMI·NDBI 연간 추이(방향·회복), 임상도 2019→2025 속성, 실적·재해를 엮어 관측/기록/해석/추가확인/불확실성으로 돌려준다. 임베딩 차원 의미는 쓰지 않는다.",
     "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]}},
    {"name": "find_trajectory", "description": "변화경로(교란 후 회복, 교란 후 회복 지연, 지속 손실·전환 의심, 수분 스트레스 등 식생 이상 의심 등)나 반복교란 여부로 소반을 찾는다. 회복 지연지·반복 교란지 점검에 사용.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "repeated": {"type": "boolean"}, "emd": {"type": "string"},
                                                        "unit_ids": {"type": "array", "items": {"type": "string"}}, "top_n": {"type": "integer"}}}},
    {"name": "find_similar_units", "description": "기준 소반(예: 회복 양호지)과 임베딩이 비슷한 소반을 찾고 경사·표고·임상·이력 차이를 함께 준다. 비교조사지·시범지 후보용.",
     "input_schema": {"type": "object", "properties": {"reference_ids": {"type": "array", "items": {"type": "string"}}, "year": {"type": "integer", "description": "2019 또는 2025"},
                                                        "top_n": {"type": "integer"}, "same_forest_type": {"type": "boolean"}, "exclude_changed": {"type": "boolean"}},
                      "required": ["reference_ids"]}},
    {"name": "map_update_candidates", "description": "임상도와 위성 관측의 불일치: (A) 위성 변화·임상도 유지 → 갱신 검토, (B) 임상도 변경·위성 무변화 → 판독 수정 가능, (C) AlphaEarth 임상 분류와 2025 임상도의 우세임상 불일치 → 재판독, (D1) AlphaEarth 피복 분류에서 제지(임도·묘지 등 산림 외 용도) 비율 증가, (D2) 수확 기록 없이 나지(미립목지) 비율 증가.",
     "input_schema": {"type": "object", "properties": {"emd": {"type": "string"}, "top_n": {"type": "integer"}}}},
    {"name": "change_zones", "description": "학습모형 변화 화소를 연속 구역(패치)으로 묶어 소반과 중첩: 구역별 면적·변화연도·걸친 소반·경계 확인 필요 여부와 소반별 변화구역 면적표.",
     "input_schema": {"type": "object", "properties": {"change_class": {"type": "string", "description": "벌채·갱신 / 산지전용 / 산림증가"}, "emd": {"type": "string"},
                                                        "unit_ids": {"type": "array", "items": {"type": "string"}}, "min_ha": {"type": "number"}, "top_n": {"type": "integer"}}}},
    {"name": "embedding_relations", "description": "임베딩 차원과 NDVI·NDMI·NDBI의 Spearman 상관(탐색용). 차원에 의미를 명명하는 근거로 쓰지 않는다.",
     "input_schema": {"type": "object", "properties": {"index": {"type": "string"}, "kind": {"type": "string"}, "top_n": {"type": "integer"}}}},
    {"name": "evaluate_foundation_model", "description": "AlphaEarth 임베딩, Sentinel-2 지수, 학습 모델(S2-TSFM, 최종지도)의 변화 탐지 성능을 같은 기준(화소·소반·산지전용 고시 필지·모델 홀드아웃)으로 비교.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "make_map", "description": "선택한 소반(빨강, 번호=목록 순서)과 산지전용 필지(파랑)를 춘천시 지도에 표시한 PNG 생성.",
     "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "unit_ids": {"type": "array", "items": {"type": "string"}},
                                                        "parcel_ids": {"type": "array", "items": {"type": "string"}}, "filename": {"type": "string"}},
                      "required": ["title"]}},
    {"name": "session_stats", "description": "GeoAI 업무상 가치 검증용: 지금까지의 처리 시간, 도구·SQL 호출 수, 결합한 테이블·자료원, 자동화된 단계를 돌려준다. 보고서 작성 직전에 호출.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "write_report", "description": "최종 보고서(markdown) 저장. 근거기록이 자동 첨부되고, 본문 수치가 도구 출력에 있는지 추적률을 돌려준다. 미확인 수치가 있으면 고쳐서 다시 호출.",
     "input_schema": {"type": "object", "properties": {"title": {"type": "string"}, "body_markdown": {"type": "string"}},
                      "required": ["title", "body_markdown"]}},
]


def call(s: Session, name: str, args: dict):
    t0 = time.time()
    out = TOOLS[name](s, **args)
    s.log.append({"tool": name, "input": args, "output": out["data"], "seconds": round(time.time() - t0, 2)})
    return out
