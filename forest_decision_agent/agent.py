"""AlphaEarth 기반 산림변화 의사결정 지원 에이전트 (Claude API 도구 호출, 수동 루프) — 가능성 검토용.

  python -m forest_decision_agent.agent "사업실적이 없는데 변화가 집중된 소반은 어디인가?"
  python -m forest_decision_agent.agent --demo          # API 키 없이 고정 순서로 도구 실행
키: 환경변수 ANTHROPIC_API_KEY, 없으면 ../claude_key.txt 첫 줄 (forest_agent 와 같은 방식)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import tools as T

MODEL = os.environ.get("FOREST_AGENT_MODEL", "claude-opus-5-5")

SYSTEM = """당신은 산림청 정책담당자를 돕는 산림변화 의사결정 지원 에이전트입니다.
대상: 강원특별자치도 춘천시, 2019→2025. 근거: 학습된 변화탐지 모델 2종(AlphaEarth+S2 최종지도, Sentinel-2 Mini-GFM),
AlphaEarth 연간 임베딩·Sentinel-2 연간 지수 원자료(2017~2025),
국유림 경영계획부 사업실적(소반 단위), 임상도 2019·2025, 보전산지 해제(산지전용) 고시 필지, 산불·산사태 이력, 임도·도로 거리.

작업 방식
- 처음에 describe_data 로 자료와 한계를 확인한 뒤, 질문에 필요한 도구만 쓴다.
- 조회·집계는 run_sql, 점검 우선순위는 rank_inspection, 파운데이션 모델의 추가 가치는 evaluate_foundation_model,
  상위 후보 몇 곳의 변화 시점과 원자료 근거는 unit_timeseries, 변화의 의미 설명은 explain_change,
  회복 지연·반복 교란·지속 손실 등 경과별 소반은 find_trajectory, 비교조사지·시범지 후보는 find_similar_units,
  임상도 갱신 검토 대상은 map_update_candidates, 점검·복구 구역과 소반별 면적은 change_zones,
  AlphaEarth 분류모델(산지 피복·임상·변화유형)로 본 지역 현황은 ae_forest_status,
  임상도 조사 이력·연도별 임상 추정은 survey_history, 임상도 재조사 순서는 resurvey_priority,
  위치 제시는 make_map, 마지막에 write_report 로 보고서를 저장한다.
- write_report 가 미확인 수치를 돌려주면 도구 출력값으로 고쳐 한 번 더 저장한다.

서술 원칙
- 면적·비율·순위 등 모든 수치는 도구 출력값을 그대로 쓴다. 직접 계산하거나 반올림을 바꾸지 않는다.
- 세 가지를 구분해 쓴다: 관측(위성 신호), 기록(사업실적·고시·재해 이력), 해석(가능한 설명과 불확실성).
- 자료 간 관계를 변화 원인으로 단정하지 않는다. 위법 여부·법령 적용은 판단하지 않고 '담당자 확인 필요'로 둔다.
- 우선순위를 제시할 때는 가중치와 순위안정성을 함께 밝힌다.
- 변화의 의미는 임베딩 차원 이름이 아니라 '임베딩으로 변화를 발견 → 지수 추이로 방향·지속성 설명 → 임상도·실적·재해로 맥락 확인'의 순서로 설명한다.
  해석은 '○○와 부합한다' 수준으로 쓰고, 정확한 발생일·원인·위법 여부·임목축적 손실량은 연간 자료로 정할 수 없다고 밝힌다.
- 임상도가 그대로라는 사실만으로 무변화로 보지 않는다(갱신 지연 가능). 유사 지역은 정책을 복제할 대상이 아니라 비교·검증할 후보다.
- AlphaEarth 분류모델은 임상도 속성만으로 학습했다. 초지(F1 0.11)는 판단에 쓰지 말고, 시설지는 '제지(임도·묘지 등)'로 표현하며,
  임상 불일치는 '재판독 후보'로만 쓴다.
- 임상도 연도별 관측 라벨로 검증한 결과, AlphaEarth 연속연도 신호는 2022·2023년 안정 산림에서도 오탐률이 높았다(5.6%·7.4%).
  이 해의 AlphaEarth 단독 변화는 Sentinel-2·학습모형 신호와 함께 확인될 때만 강한 근거로 쓰고, 아니면 '연도 신뢰도 낮음'으로 밝힌다.
- 상위 몇 % 같은 상대 기준으로 고른 면적은 '검증된 변화면적'이 아니라 후보 면적임을 밝힌다.
- 변화 판정은 학습 모델 최종지도 기준이다. 신호일치수(AE 임베딩거리·Mini-GFM·최종지도 중 변화로 본 수)로 불확실성을 표시하고,
  모델의 변화유형(벌채·갱신/산지전용/산림증가)과 변화연도를 함께 쓴다. 산지전용 유형은 모델 정확도가 낮아 후보로만 쓴다.
- 기간 조건(예: 2020~2024)은 AE_변화연도와 사업실적 연도로 걸러 적용하고, 위성 신호 자체는 2019→2025 비교임을 밝힌다.
- 'AlphaEarth 의 분석상 추가 가치'는 evaluate_foundation_model 결과(같은 오탐률에서 Sentinel-2 와 비교)로,
  'GeoAI 의 업무상 추가 가치'는 session_stats(처리 시간, 결합 자료, 자동화 단계)와 write_report 의 수치 추적률로 근거를 든다.
- 지도는 make_map 에 우선 후보 소반을 순위 순서로 넘겨 번호가 표의 순위와 같게 한다.
- 보고서는 결론 3줄 → 근거 표 → 점검 권고 → 한계 순으로 짧게 쓴다. 답변은 한국어."""


def api_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    for f in (Path(__file__).resolve().parent / "claude_key.txt", T.ROOT / "claude_key.txt"):
        if not k and f.exists():
            k = f.read_text(encoding="utf-8").splitlines()[0].strip()
    return k or None


def area_text(area: dict | None) -> str:
    """웹에서 지정한 지역을 질의에 붙이는 문장."""
    if not area:
        return ""
    return ("\n\n[대상 지역] " + json.dumps(area, ensure_ascii=False)
            + "\n이 지역에 한정해 답할 것: area_summary 에 같은 지역 인자를 넘기고, 순위·SQL 도 그 지역(읍면동 또는 area_summary 결과의 소반)으로 제한.")


def run_claude(query: str, max_turns: int = 20, area: dict | None = None, log=print, session: T.Session | None = None) -> T.Session:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key())
    s = session or T.Session(query=query)
    messages = [{"role": "user", "content": query + area_text(area)}]
    usage = {"input": 0, "output": 0}
    t0 = time.time()
    for _ in range(max_turns):
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=T.TOOL_SCHEMAS, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # 안전 분류기 거절 시 서버측 대체 모델
        )
        usage["input"] += resp.usage.input_tokens
        usage["output"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                log(f"[에이전트] {b.text}")
        if resp.stop_reason == "refusal":
            log(f"[거절] {getattr(resp.stop_details, 'explanation', '')}")
            break
        if resp.stop_reason == "pause_turn":
            continue
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            break
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("응답이 max_tokens 에서 잘렸습니다.")
        results = []
        for tu in uses:
            log(f"[도구] {tu.name} {json.dumps(tu.input, ensure_ascii=False)[:160]}")
            try:
                if tu.name not in T.TOOLS or not isinstance(tu.input, dict):
                    raise ValueError(f"잘못된 도구 호출: {tu.name}")
                out = T.call(s, tu.name, tu.input)
                content, err = json.dumps(out, ensure_ascii=False, default=str), False
            except Exception as e:  # 도구 오류는 모델에 돌려주고 계속
                content, err = f"{type(e).__name__}: {e}", True
                log(f"   ✗ {content}")
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content, "is_error": err})
        messages.append({"role": "user", "content": results})
    (s.run_dir / "run_stats.json").write_text(json.dumps({"model": MODEL, "seconds": round(time.time() - t0, 1), "tokens": usage,
                                                         "tool_calls": len(s.log)}, ensure_ascii=False), encoding="utf-8")
    return s


def run_demo(query: str = "사업실적이 없는데 위성 변화가 집중된 소반을 찾고, 재해 이력과 접근성을 고려한 점검 우선순위 10곳과 "
                          "AlphaEarth 활용 가치를 정리해줘") -> T.Session:
    """LLM 없이 도구를 고정 순서로 실행하고 템플릿 보고서를 쓴다 (도구·근거기록 동작 확인용)."""
    s = T.Session(query=query)
    step = lambda n, **kw: T.call(s, n, kw)["data"]
    step("describe_data")
    j = step("run_sql", sql='select "판정", count(*) as 소반수, round(sum("면적_ha"),1) as 면적_ha, round(sum("AE_변화면적_ha"),1) as AE_변화면적_ha '
                            'from "소반" group by "판정" order by 소반수 desc', purpose="판정별 현황")
    emd = step("run_sql", sql='select "읍면동", count(*) as 미기록변화_소반, round(sum("AE_변화면적_ha"),1) as AE_변화면적_ha from "소반" '
                              'where "판정"=\'미기록 변화(점검 필요)\' group by "읍면동" order by AE_변화면적_ha desc limit 5', purpose="미기록 변화 읍면동")
    rk = step("rank_inspection", top_n=10)
    ev = step("evaluate_foundation_model")
    ids = [r["단위ID"] for r in rk["ranking"]]
    step("make_map", title="현장점검 우선순위 상위 10 소반 (2019→2025)", unit_ids=ids, filename="priority_map.png")
    jt = "\n".join(f"| {r['판정']} | {r['소반수']} | {r['면적_ha']} | {r['AE_변화면적_ha']} |" for r in j["rows"])
    et = "\n".join(f"| {r['읍면동']} | {r['미기록변화_소반']} | {r['AE_변화면적_ha']} |" for r in emd["rows"])
    rt = "\n".join(f"| {i} | {r['단위ID']} | {r['읍면동']} | {r['점수']} | {r['순위안정성']} | {r['AE_변화면적_ha']} | {int(r['AE_변화연도']) if r['AE_변화연도'] else '-'} | "
                   f"{r['산불_500m_2019_2025'] + r['산사태_500m_2019_2025']} | {int(r['임도거리_m'])} |" for i, r in enumerate(rk["ranking"], 1))
    px = ev["화소_검증(오탐률 5%에서 변화화소 탐지율)"]
    un = ev["소반_검증(임상도 변화 확인 소반 vs 확정무변화 소반)"]
    body = f"""## 결론
- 미기록 변화(점검 필요)의 AE 변화면적이 가장 큰 곳은 {emd["rows"][0]["읍면동"]}입니다.
- 1순위 점검 소반은 {ids[0]} (점수 {rk['ranking'][0]['점수']}, 순위안정성 {rk['ranking'][0]['순위안정성']})입니다.
- 같은 오탐률 5%에서 변화 화소 탐지율은 AlphaEarth {px['AlphaEarth']['변화화소_탐지율']}, Sentinel-2 {px['Sentinel-2']['변화화소_탐지율']}입니다.

## 판정별 현황
| 판정 | 소반 | 면적 ha | AE 변화면적 ha |
|---|---|---|---|
{jt}

## 미기록 변화 상위 읍면동
| 읍면동 | 소반 | AE 변화면적 ha |
|---|---|---|
{et}

## 점검 우선순위 (가중치 {rk['weights']})
| 순위 | 소반 | 읍면동 | 점수 | 순위안정성 | AE 변화면적 ha | 변화연도 | 재해(500 m) | 임도거리 m |
|---|---|---|---|---|---|---|---|---|
{rt}

## 파운데이션 모델 활용 가치
- 소반 검증 AUROC: AlphaEarth {un['AlphaEarth 임베딩거리 변화비율']['AUROC']}, Sentinel-2 {un['Sentinel-2 NDVI 변화비율']['AUROC']}

## 한계
- 위성 변화는 관측 사실이며 원인이 아닙니다. 위법 여부는 담당자 확인이 필요합니다.
- 데모 모드: 서술문은 템플릿입니다."""
    step("write_report", title="산림변화 현장점검 우선순위 (데모)", body_markdown=body)
    return s


# 영문 질의도 같은 규칙으로 고르도록 한글 키워드별 영문 표현
EN_KEYS = {"산지전용": ["conversion"], "전용": ["conversion"], "필지": ["parcel"], "고시": ["notice"],
           "AlphaEarth": ["alphaearth"], "파운데이션": ["foundation"], "모델": ["model"], "Sentinel": ["sentinel"], "가치": ["value", "help"],
           "비교": ["compare", "comparison"], "회복": ["recover"], "복구": ["restor"], "조림": ["planting", "plantation"], "반복": ["repeat"],
           "지속": ["persistent"], "비산림": ["non-forest"], "수분": ["moisture", "water stress"], "스트레스": ["stress"], "병해충": ["pest"],
           "고사": ["dieback", "mortality"], "유사": ["similar"], "비슷": ["resembl"], "비교조사": ["comparison candidate"], "시범": ["pilot"],
           "갱신": ["not yet reflected", "map update"], "지도 수정": ["map correction"], "구역": ["zone"], "면적표": ["area table"],
           "경계": ["boundar"], "임상": ["forest type", "forest-type"], "피복": ["land cover", "land-cover"], "분류": ["classif"],
           "시설": ["non-forest use", "facilit"], "나지": ["bare"], "침엽": ["conifer"], "활엽": ["broadleaf"], "혼효": ["mixed forest"],
           "재조사": ["re-survey", "resurvey"], "조사 이력": ["survey history"], "연도별 임상": ["yearly forest"], "차원": ["dimension"],
           "임베딩 의미": ["embedding meaning"], "상관": ["correlat"]}


def _has(query: str, *ks) -> bool:
    ql = query.lower()
    return any(k in query or any(e in ql for e in EN_KEYS.get(k, ())) for k in ks)


def run_offline(query: str, area: dict | None = None, log=print, session: T.Session | None = None) -> T.Session:
    """LLM 미연결 시 규칙 기반 분석: 지역 요약 → 점검 우선순위 → (요청 시) 산지전용·모델 비교 → 지도 → 템플릿 보고서."""
    s = session or T.Session(query=query)
    area = {k: v for k, v in (area or {}).items() if v}

    def step(n, **kw):
        log(f"[도구] {n} {json.dumps(kw, ensure_ascii=False)[:160]}")
        return T.call(s, n, kw)["data"]

    sm = step("area_summary", **area)
    ids = T.resolve_area(**area)[0] if area else None
    rk = step("rank_inspection", top_n=10, unit_ids=ids) if ids != [] else {"ranking": [], "weights": T.DEFAULT_W}
    want_conv = _has(query, "산지전용", "전용", "필지", "고시")
    want_eval = _has(query, "AlphaEarth", "알파어스", "파운데이션", "모델", "Sentinel", "센티넬", "가치", "비교")
    conv = None
    if want_conv:
        geo = T.resolve_area(**area)[1] if area else None
        pc = T._layer("산지전용")
        pc = pc[pc.intersects(geo)] if geo is not None else pc
        ids_p = [f"'{i}'" for i in pc["레코드ID"]] or ["''"]
        conv = step("run_sql", sql='select "레코드ID","고시번호","주소","지목","전용용도","해제면적_m2","라벨","AE_변화비율","S2_변화비율","AE_변화연도" '
                                    f'from "산지전용" where "레코드ID" in ({",".join(ids_p)}) order by "고시연도" desc, "AE_변화비율" desc limit 30',
                    purpose="지역 산지전용 고시 필지와 위성 변화")
    ev = step("evaluate_foundation_model") if want_eval else None
    top_ids = [r["단위ID"] for r in rk["ranking"]]
    step("make_map", title=f"{sm['지역']} 점검 우선순위 소반", unit_ids=top_ids,
         parcel_ids=[r["레코드ID"] for r in conv["rows"]] if conv else None, filename="map.png")

    jt = "\n".join(f"| {r['판정']} | {r['소반']} | {r['면적_ha']} | {r['AE_변화면적_ha']} | {r['S2_변화면적_ha']} |" for r in sm["판정별"])
    rt = "\n".join(f"| {i} | {r['단위ID']} | {r['읍면동']} | {r['판정']} | {r['점수']} | {r['순위안정성']} | {r['FM_주변화유형'] or '-'} | "
                   f"{r['FM_변화면적_ha']} | {int(r['FM_변화연도']) if r['FM_변화연도'] else '-'} | {r['신호일치수']} | {r['실적구분_2020_2025'] or '실적없음'} | "
                   f"{r['산불_500m_2019_2025'] + r['산사태_500m_2019_2025']} |" for i, r in enumerate(rk["ranking"], 1))
    ts = [step("unit_timeseries", unit_id=r["단위ID"]) for r in rk["ranking"][:3]]   # 상위 3곳 원자료 시계열
    tt = "\n".join(f"| {t['대상']} | {t['AlphaEarth_최대변화구간']} | {t['AlphaEarth_2019→2025_거리']} | "
                   f"{t['Sentinel-2_연별_평균'][2019]['NDVI']} → {t['Sentinel-2_연별_평균'][2025]['NDVI']} | "
                   f"{t['최종지도(AE+S2)']['변화연도_화소'] or '-'} |" for t in ts)
    pt = "\n".join(f"| {r['라벨']} | {r['필지']} | {r['해제면적_m2']} | {r['AE_변화비율_중앙값']} |" for r in sm["산지전용_필지"])
    body = [f"## 요약 ({sm['지역']})",
            f"- 국유림 소반 {sm['국유림_소반']}개, {sm['국유림_면적_ha']} ha. AlphaEarth 변화면적 {sm['AE_변화면적_ha']} ha, Sentinel-2 변화면적 {sm['S2_변화면적_ha']} ha.",
            f"- 학습 모델 변화면적(ha): 최종지도(AE+S2) {sm['모델_변화면적_ha']['FM']}, Mini-GFM(S2) {sm['모델_변화면적_ha']['MG']}.",
            f"- 신호일치수별 소반(AE 임베딩거리·Mini-GFM·최종지도 중 변화로 본 수): {sm['신호일치수별_소반']}.",
            f"- 재해(2019~2025): {sm['재해_2019_2025'] or '없음'}. 최종지도 변화연도별 소반수: {sm['FM_변화연도별_소반수'] or '없음'}.",
            "", "## 판정별 현황 (관측·기록 대조)", "| 판정 | 소반 | 면적 ha | AE 변화 ha | S2 변화 ha |", "|---|---|---|---|---|", jt]
    if rk["ranking"]:
        body += ["", f"## 현장점검 우선순위 (가중치 {rk['weights']})",
                 "| 순위 | 소반 | 읍면동 | 판정 | 점수 | 순위안정성 | 모델 유형 | 모델 변화 ha | 변화연도 | 신호일치 | 실적 | 재해(500 m) |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|", rt,
                 "", "### 상위 3곳 원자료 시계열 (AlphaEarth·Sentinel-2·최종지도)",
                 "| 소반 | AE 최대 변화 구간 | AE 거리 2019→2025 | NDVI 2019 → 2025 | 최종지도 변화연도(화소) |", "|---|---|---|---|---|", tt]
    if sm["산지전용_필지"]:
        body += ["", "## 산지전용 고시 필지", "| 라벨 | 필지 | 해제면적 m² | AE 변화비율 중앙값 |", "|---|---|---|---|", pt]
    if conv and conv["rows"]:
        body += ["", "| 레코드 | 주소 | 용도 | 라벨 | AE 변화비율 | S2 변화비율 |", "|---|---|---|---|---|---|"]
        body += [f"| {r['레코드ID']} | {r['주소']} | {r['전용용도'] or '-'} | {r['라벨']} | {r['AE_변화비율']} | {r['S2_변화비율']} |" for r in conv["rows"][:15]]
    if ev:
        px = ev["화소_검증(오탐률 5%에서 변화화소 탐지율)"]
        un = ev["소반_검증(임상도 변화 확인 소반 vs 확정무변화 소반)"]
        body += ["", "## AlphaEarth 활용 가치 (춘천시 전체 기준)",
                 f"- 오탐률 5%에서 변화화소 탐지율: AlphaEarth {px['AlphaEarth']['변화화소_탐지율']}, Sentinel-2 {px['Sentinel-2']['변화화소_탐지율']}",
                 f"- 소반 판별 AUROC(라벨 없이 계산한 신호): AlphaEarth {un['AlphaEarth 임베딩거리 변화비율']['AUROC']}, "
                 f"Sentinel-2 {un['Sentinel-2 NDVI 변화비율']['AUROC']}",
                 f"- 학습 모델 홀드아웃 macro F1: 최종지도(AE+S2) {ev['학습모델_홀드아웃(모델 보고서)']['최종지도']['macro_f1']}, "
                 f"Mini-GFM(S2) {ev['학습모델_홀드아웃(모델 보고서)']['Mini-GFM']['macro_f1']}"]
    # 변화의 의미 설명: 임베딩(시기) → 지수 추이(방향·지속성) → 임상도·실적·재해(맥락)
    ex = [step("explain_change", unit_id=r["단위ID"]) for r in rk["ranking"][:3]]
    if ex:
        body += ["", "## 상위 후보 변화 해석 (관측 → 기록 → 해석)"]
        for e in ex:
            g = e["근거"]
            body += [f"### {e['소반']} — {e['변화경로']}", f"- 관측: {g['관측']}", f"- 기록: {g['기록']}", f"- 해석: {g['해석']}",
                     f"- 추가 확인: {g['추가확인']}", f"- 불확실성: {g['불확실성']}"]
    uid = ids if area else None
    want = lambda *ks: _has(query, *ks)
    trj = lambda ttl, **kw: (lambda d: ["", f"## {ttl} ({d['해당소반']}곳)", "| 소반 | 읍면동 | 변화구간 | NDVI 변화전 → 최저 → 2025 | 회복률 | 고변화연도 | 수종(2019) |",
                                        "|---|---|---|---|---|---|---|"] + [
        f"| {r['단위ID']} | {r['읍면동']} | {r['변화구간'] or '-'} | {r['NDVI_변화전']} → {r['NDVI_최저']} → {r['NDVI_2025']} | {r['회복률']} | {r['고변화연도'] or '-'} | {r['수종_2019'] or '-'} |"
        for r in d["목록"][:10]])(step("find_trajectory", unit_ids=uid, top_n=10, **kw))
    if want("회복", "복구", "조림"):
        body += trj("교란 후 회복 지연 소반", path="교란 후 회복 지연")
    if want("반복"):
        body += trj("반복 교란 소반", repeated=True)
    if want("지속", "전환", "비산림"):
        body += trj("지속 손실·전환 의심 소반", path="지속 손실·전환 의심")
    if want("수분", "스트레스", "병해충", "고사"):
        body += trj("수분 스트레스 등 식생 이상 의심 소반", path="수분 스트레스 등 식생 이상 의심")
    if want("유사", "비슷", "비교조사", "시범"):
        ref = step("find_trajectory", path="교란 후 회복", unit_ids=uid, top_n=3)["목록"]
        if ref:
            sim = step("find_similar_units", reference_ids=[r["단위ID"] for r in ref], top_n=8)
            body += ["", f"## 회복 양호지({', '.join(r['단위ID'] for r in ref)})와 임베딩이 유사한 소반 (비교·검증 후보)",
                     "| 소반 | 읍면동 | 유사도 | 변화경로 | 임상·수종(2025) | 경사차 ° | 표고차 m |", "|---|---|---|---|---|---|---|"] + [
                f"| {r['단위ID']} | {r['읍면동']} | {r['유사도']} | {r['변화경로']} | {r['임상_2025'] or '-'}·{r['수종_2025'] or '-'} | {r['경사차']} | {r['표고차']} |"
                for r in sim["후보"]]
    if want("갱신", "임상도 불일치", "지도 수정"):
        mu = step("map_update_candidates", emd=area.get("emd"), top_n=10)
        a, b = mu["A_임상도_갱신검토(위성 변화·임상도 유지)"], mu["B_판독수정_가능(임상도 변경·위성 무변화)"]
        body += ["", f"## 임상도 갱신 검토 후보 {a['소반']}곳 · 판독 수정 가능 {b['소반']}곳",
                 "| 구분 | 소반 | 읍면동 | 모델 유형 | 모델 변화 ha | 임상도 변화비율 | 2019 → 2025 수종 |", "|---|---|---|---|---|---|---|"] + [
            f"| {k} | {r['단위ID']} | {r['읍면동']} | {r['FM_주변화유형'] or '-'} | {r['FM_변화면적_ha']} | {r['임상도_변화비율']} | {r['수종_2019'] or '-'} → {r['수종_2025'] or '-'} |"
            for k, d in (("갱신 검토", a), ("판독 수정", b)) for r in d["목록"][:8]]
    if want("구역", "면적표", "경계", "세그멘테이션"):
        z = step("change_zones", emd=area.get("emd"), unit_ids=area.get("unit_ids"), top_n=10)
        body += ["", f"## 변화 구역 {z['패치수']}개 ({z['면적_ha']} ha, 경계 확인 {z['경계확인_패치']}개)",
                 "| 구역 | 유형 | 면적 ha | 변화연도 | 걸친 소반 | 국유림 밖 ha | 경계 확인 |", "|---|---|---|---|---|---|---|"] + [
            f"| {r['patch_id']} | {r['class_ko']} | {r['area_ha']} | {r['change_year']} | {r['소반'] or '-'} | {r['국유림외_ha']} | {r['경계확인'] or '-'} |" for r in z["구역"]]
    if want("임상", "피복", "분류", "시설", "나지", "침엽", "활엽", "혼효"):
        st = step("ae_forest_status", **area)
        if st.get("소반"):
            body += ["", f"## AlphaEarth 분류모델 현황 ({st['지역']}, 소반 {st['소반']}곳)",
                     "| 구분 | 2019 | 2025 |", "|---|---|---|"] + [
                f"| 피복 {l} (ha) | {st['피복_ha'][2019][l]} | {st['피복_ha'][2025][l]} |" for l in T.AE_LC] + [
                f"| 임상 {l} (ha) | {st['임상_ha'][2019][l]} | {st['임상_ha'][2025][l]} |" for l in T.AE_FT] + [
                f"| 임상도와 우세임상 일치율 | {st['임상도와_우세임상_일치'][2019]['우세임상_일치율']} | {st['임상도와_우세임상_일치'][2025]['우세임상_일치율']} |",
                "", f"- 변화유형 분류(2019→2025): 벌채·갱신 {st['변화유형_ha(2019→2025)']['벌채·갱신']} ha, 산림증가 {st['변화유형_ha(2019→2025)']['산림증가']} ha.",
                "- 초지 결과는 정확도가 낮아 판단에 쓰지 않았고, 시설지는 임도·묘지 등 산지 안 산림 외 용도(제지)를 뜻합니다."]
        mu = step("map_update_candidates", emd=area.get("emd"), top_n=10)
        c = mu["C_임상_재판독(AE 분류 우세임상 ≠ 2025 임상도)"]
        d1, d2 = mu["D1_산림외용도(제지)_확대(AE 분류 시설지 +5%p 이상)"], mu["D2_수확기록없는_미립목지화(AE 분류 나지 +10%p 이상)"]
        body += ["", f"## 임상 재판독 후보 {c['소반']}곳 · 제지 확대 {d1['소반']}곳 · 수확기록 없는 미립목지화 {d2['소반']}곳",
                 "| 구분 | 소반 | 읍면동 | 임상도 2025 | AE 분류 | 근거 |", "|---|---|---|---|---|---|"] + [
            f"| 임상 재판독 | {r['단위ID']} | {r['읍면동']} | {r['임상_2025']} | {r['AE_우세임상_2025']} | 평균확률 {r['FT_2025_평균확률']} |" for r in c["목록"][:8]] + [
            f"| {k} | {r['단위ID']} | {r['읍면동']} | 산림비율 {r['LC_2019_산림_비율']}→{r['LC_2025_산림_비율']} | 제지 {r['LC_2025_시설지_비율']}·나지 {r['LC_2025_나지_비율']} | 제지 +{r['시설지_증가']}, 나지 +{r['나지_증가']} |"
            for k, d in (("제지 확대", d1), ("미립목지화", d2)) for r in d["목록"][:6]]
    if want("재조사", "조사 이력", "조사이력", "갱신", "연도별 임상"):
        rp = step("resurvey_priority", emd=area.get("emd"), unit_ids=area.get("unit_ids"), top_n=10)
        body += ["", f"## 임상도 재조사 우선순위 (후보 {rp['후보']}곳)", f"- 점수: {rp['점수']}",
                 "| 소반 | 읍면동 | 점수 | 사유 | 마지막 조사 | 위성 변화연도 | 연도 신뢰도 낮음 | AE 연도별 우세임상 |", "|---|---|---|---|---|---|---|---|"] + [
            f"| {r['단위ID']} | {r['읍면동']} | {r['점수']} | {r['재조사_우선'] or '-'} | {r['마지막조사_최근']} | {r['위성변화연도'] if r['위성변화연도'] else '-'} | "
            f"{'예' if r['연도신뢰도낮음'] else '-'} | {(r['AE연도별_우세임상'] or '-')[:60]} |" for r in rp["목록"]]
    if want("차원", "임베딩 의미", "상관"):
        er = step("embedding_relations", index="NDVI")
        body += ["", "## 임베딩 차원과 NDVI 변화의 관계 (탐색)",
                 "- 상위 차원: " + ", ".join(f"{r['차원']}({r['rho']})" for r in er["상위차원"][:5]),
                 "- 상관은 관련성일 뿐이며 차원에 '수분', '영급' 같은 이름을 붙이는 근거가 아닙니다."]
    body += ["", "## 해석 시 유의", "- 위성 변화는 관측 사실이며 원인이 아닙니다. 위법 여부·법령 적용은 담당자 확인이 필요합니다.",
             "- 변화 해석은 임베딩 차원의 의미가 아니라 임베딩 변화 시기, 지수 추이, 임상도·실적 맥락이 같은 해석을 지지하는지로 판단했습니다.",
             "- 사업실적·판정은 국유림 소반만 대상입니다.", "- LLM 미연결: 규칙 기반 템플릿 보고서입니다(질의 해석 없이 지역 요약·우선순위 위주)."]
    w = step("write_report", title=f"산림변화 의사결정 지원 보고서 — {sm['지역']}", body_markdown="\n".join(body))
    log(f"[완료] 보고서 저장 (수치 추적률 {w['수치_추적률']})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo or not a.query:
        s = run_demo(a.query) if a.query else run_demo()
    elif not api_key():
        sys.exit("ANTHROPIC_API_KEY 또는 claude_key.txt 가 없습니다. --demo 로 도구 동작만 확인할 수 있습니다.")
    else:
        s = run_claude(a.query)
    print(f"\n결과 폴더: {s.run_dir}")
    if s.log and s.log[-1]["tool"] == "write_report":
        print("수치 추적률:", s.log[-1]["output"]["수치_추적률"], "| 미확인:", s.log[-1]["output"]["미확인_수치"])


if __name__ == "__main__":
    main()
