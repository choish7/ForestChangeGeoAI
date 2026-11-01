# [통합 사본] 원본: scripts/save_yearly_validation_set.py — 경로만 forest-change-agent 기준으로 바꿈 (계산 로직 동일)
"""임상도 연도별 관측 라벨(build_yearly_labels.py 결과)을 '검증 전용' 데이터셋으로 정리 → output/연도별라벨/검증용/

  조각ID별 고정 표본점 1개(조각을 10 m 안쪽으로 줄인 뒤 대표점) × 2017~2025 → AlphaEarth 10 m 격자 행·열
  같은 화소를 9개 연도에 쓰므로 연도별 임베딩·지수를 같은 위치에서 비교할 수 있다.
  블록: 분류모델 표본과 같은 500×500 화소(5 km) 블록·seed 42 의 검증 블록 표시(검증블록=1) — 임상도 2019·2025 로 학습한 모델은 검증블록만 사용
출력
  검증_표본점_연도별.parquet : 조각ID, 연도, 행, 열, x, y, 블록, 검증블록, 상태, 입목여부, 임상, 라벨유형, 등급, 면적_ha, 국유림소반
  검증_변화구간.parquet      : 변화구간 조각의 표본점 + 이전·이후 상태, 구간시작·끝, 구간길이, 양끝_조사연도, 시점평가용(≤3년·양끝 조사)
  검증_안정조각.parquet      : 2017~2025 중 5개 연도 이상 A등급(관측·유지) 같은 상태인 조각 — 연도별 오탐 평가용
  연도별관측라벨_2017_2025.gpkg, 연도별관측라벨_long.parquet, 변화구간.csv, 진단.json (원자료를 이 폴더로 이동)
  README_검증용.md
"""
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine, rowcol

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output" / "연도별라벨"
DST = SRC / "검증용"
TF = Affine(10, 0, 368350, 0, -10, 4216210)
H, W = 4549, 4647
BLOCK, SEED = 500, 42
rng = np.random.default_rng(SEED)
nb_c = -(-W // BLOCK)
HOLD = rng.random(-(-H // BLOCK) * nb_c) < 0.2          # build_ae_class_samples.py 와 같은 검증 블록


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in ("연도별관측라벨_2017_2025.gpkg", "연도별관측라벨_long.parquet", "변화구간.csv", "진단.json"):
        if (SRC / f).exists():
            shutil.move(str(SRC / f), str(DST / f))
    p = gpd.read_file(DST / "연도별관측라벨_2017_2025.gpkg", layer="조각", columns=["조각ID", "area_ha", "국유림소반", "geometry"])
    inner = p.geometry.buffer(-10)
    inner = inner.where(~inner.is_empty & inner.notna(), p.geometry)
    pt = gpd.GeoSeries(inner, crs=p.crs).representative_point().to_crs(32652)
    r, c = rowcol(TF, pt.x.values, pt.y.values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    blk = (np.clip(r, 0, H - 1) // BLOCK) * nb_c + np.clip(c, 0, W - 1) // BLOCK
    pts = pd.DataFrame({"조각ID": p["조각ID"], "행": r, "열": c, "x": pt.x.round(1), "y": pt.y.round(1), "블록": blk,
                        "검증블록": HOLD[blk].astype(int), "면적_ha": p["area_ha"].round(3), "국유림소반": p["국유림소반"]})[ok]
    pts = pts.drop_duplicates(["행", "열"])                                                     # 같은 화소에 두 조각이면 하나만
    lab = pd.read_parquet(DST / "연도별관측라벨_long.parquet").drop(columns=["area_ha"])
    v = pts.merge(lab, on="조각ID")
    v.to_parquet(DST / "검증_표본점_연도별.parquet", index=False)

    chg = pd.read_csv(DST / "변화구간.csv").merge(pts, on="조각ID")
    chg["시점평가용"] = (chg["구간길이"] <= 3) & chg["양끝_조사연도"]
    chg.to_parquet(DST / "검증_변화구간.parquet", index=False)

    a = v[v["등급"].eq("A") & v["라벨유형"].isin(["관측", "유지추정"])]
    st = a.groupby("조각ID").agg(A연도수=("연도", "nunique"), 상태수=("상태", "nunique"), 상태=("상태", "first"))
    stable = st[(st["A연도수"] >= 5) & (st["상태수"] == 1) & ~st.index.isin(chg["조각ID"])].reset_index().merge(pts, on="조각ID")
    stable.to_parquet(DST / "검증_안정조각.parquet", index=False)

    d = json.loads((DST / "진단.json").read_text(encoding="utf-8"))
    usable = v[v["등급"].eq("A")]
    tab = usable.pivot_table(index="연도", columns="임상", values="조각ID", aggfunc="count").fillna(0).astype(int)
    tab_h = usable[usable["검증블록"].eq(1)].pivot_table(index="연도", columns="임상", values="조각ID", aggfunc="count").fillna(0).astype(int)
    readme = f"""# 임상도 연도별 관측 라벨 — 검증 전용 데이터셋 (2017~2025)

이 폴더의 자료는 **모델 평가(검증)에만 사용합니다. 학습·임계값 조정·특징 선택에 쓰지 않습니다.**
생성: `scripts/build_yearly_labels.py` → `scripts/save_yearly_validation_set.py`

## 원자료와 원칙
- 임상도 2019판(구축 2015~2019), 2024판(2020~2024), 2025판(2025)을 교차 중첩한 조각 {len(p):,}개(≥0.1 ha). 2013판은 갱신년도가 없어 `조각.임상_2013(맥락)`으로만 둠.
- 갱신년도 = 그 폴리곤 속성을 조사·판독한 해. 그 해 상태는 **관측**, 다음 해부터 판 연도까지는 **유지추정**(갱신되지 않았다는 뜻이며 확인된 무변화가 아님).
- 상태 = 입목여부(입목지·무립목지·비산림) + 임상(침엽수림·활엽수림·혼효림). 영급·경급·밀도는 시간에 따라 변하므로 라벨에서 제외.

## 라벨유형·등급
| 라벨유형 | 뜻 |
|---|---|
| 관측 | 그 해가 어느 판의 갱신년도 (A: 갱신년도 있음, B: 갱신년도가 없어 판 연도로 대신한 약한 관측) |
| 유지추정 | 조사 후 같은 상태가 이어진다고 추정되는 해 |
| 변화구간 | 앞 조사와 뒤 조사의 상태가 달라 그 사이 어느 해에 변한 구간 (상태 미상) |
| 미상 | 첫 조사 이전 등 근거 없음 |
| 판간불일치 | 같은 해를 두 판이 다른 상태로 기록 (제외) |

| 등급 | 기준 | 쓰임 |
|---|---|---|
| A | 갱신년도 있는 두 조사가 같은 상태를 확인한 사이의 해(유지추정), 또는 갱신년도 관측 | **주 평가** |
| B | 한 판만 지지(마지막 조사 뒤 ~ 판 연도) 또는 약한 관측 | 보조 결과로 따로 제시 |
| C | 유지 구간 안에 벌채·조림 실적(국유림 소반) 또는 산지전용 고시 | 제외 |

## 파일
| 파일 | 내용 |
|---|---|
| `검증_표본점_연도별.parquet` | 조각마다 고정 표본점 1개(10 m 안쪽 대표점) × 2017~2025. AlphaEarth 10 m 격자 `행`·`열`, UTM52N `x`·`y`, 5 km `블록`, `검증블록` |
| `검증_변화구간.parquet` | 변화구간 {len(chg):,}건(이전·이후 상태, 구간시작·끝). `시점평가용`(구간 ≤3년, 양끝 조사연도) {int(chg['시점평가용'].sum()):,}건 |
| `검증_안정조각.parquet` | 5개 연도 이상 A등급으로 같은 상태가 확인된 조각 {len(stable):,}개 — 연도별 오탐 평가 |
| `연도별관측라벨_2017_2025.gpkg` | 레이어 `조각`(판별 상태·수종·영급·갱신년도, 국유림소반, 2013 맥락), `변화구간` |
| `연도별관측라벨_long.parquet` | 조각×연도 라벨 전체 |
| `진단.json` | 판 간 일치 진단, 연도별 라벨 면적, 변화 전이 |

## 사용 규칙
1. **독립성:** 임상도 2019·2025판 라벨로 학습한 모델(ae_class_models 등)은 같은 지도에서 라벨을 얻었으므로 `검증블록 == 1` 표본만 평가에 씁니다. 같은 조각의 여러 연도는 하나의 단위로 다룹니다(연도 간 상관).
2. **주 결과는 A등급**, B등급은 별도 표로 제시합니다.
3. **시점 평가:** `시점평가용` 변화구간에서 신호 연도가 (구간시작, 구간끝]에 드는지로 평가합니다(정의상 자동 만족되는 편향이 없음).
4. **임상 간 전이 주의:** 임상 간 전이(활엽↔침엽↔혼효)는 실제 수종 변화보다 판독 수정일 가능성이 큽니다. 입목지↔무립목지 전이와 따로 보고합니다.
5. **연도 편중:** 연도별 표본은 조사연도에 따라 크게 다릅니다(2017·2019·2023·2025에 집중). 연도별 비교에서는 표본 수를 함께 제시합니다.

## 진단 요약
- 판 사이 같은 갱신년도인데 상태가 다른 모순: 2019→2024 {d['2019→2024']['같은갱신년도인데_상태다름(모순)']:.1%}, 2024→2025 {d['2024→2025']['같은갱신년도인데_상태다름(모순)']:.1%}
- 갱신년도 역행(뒤 판이 더 이른 조사): 2019→2024 {d['2019→2024']['갱신년도_역행(뒤판이_더이른조사)']:.1%}
- 재조사로 같은 상태 확인(면적비): 2019→2024 {d['2019→2024']['재조사로_같은상태확인(뒤판 갱신 > 앞판 갱신)']:.1%}, 2019→2025 {d['2019→2025']['재조사로_같은상태확인(뒤판 갱신 > 앞판 갱신)']:.1%}
- 상태 변경 면적비: 2019→2024 {d['2019→2024']['상태변경_면적비']:.1%}, 2024→2025 {d['2024→2025']['상태변경_면적비']:.1%}
- 변화구간 전이 상위: {', '.join(f'{k} {v:,}' for k, v in list(d['변화구간']['전이'].items())[:6])}

## A등급 표본점 수 (연도 × 임상)
전체
{tab.to_markdown()}

검증블록만
{tab_h.to_markdown()}
"""
    (DST / "README_검증용.md").write_text(readme, encoding="utf-8")
    print(f"표본점 {len(pts):,} × 연도 → {len(v):,}행, A등급 {len(usable):,}행 (검증블록 {int(usable['검증블록'].sum()):,})")
    print(f"변화구간 {len(chg):,} (시점평가용 {int(chg['시점평가용'].sum()):,}), 안정조각 {len(stable):,}")
    print(tab.to_string())
    print("저장:", DST)


if __name__ == "__main__":
    main()
