"""[4.4.2] Geo-LLM 산림변화 에이전트 (Claude API tool use, 수동 루프).

질의 해석 → 데이터검색 → 원격탐사 분석 → GIS 공간분석 → 대안 평가 → 결과 제공(의사결정 카드)
수치는 도구가 계산하고, LLM 은 도구 선택·해석·서술만 담당한다.
"""
import json

import anthropic

from . import config as C
from . import tools as T

SYSTEM = f"""당신은 산림청 정책담당자를 돕는 GeoAI 산림변화 분석 에이전트입니다.
대상: 강원특별자치도 춘천시(51110). 보유 근거: 임상도 2019·2025 비교 변화 DB, 국유림 경영계획부(9차),
AlphaEarth 연간 임베딩(2017~2025, GEE), 학습된 변화탐지모형(2019→2025, AlphaEarth+Sentinel-2+DEM).

처리 단계 (순서대로, 필요한 단계만):
1 질의 해석   set_analysis_spec 을 가장 먼저 호출. ambiguities 가 있으면 추측하지 말고 사용자에게 되물을 것.
2 데이터검색  DB 결과를 우선 조회 (query_change_map, query_forest_status, query_mgmt_plan). 출처·기준시점 확인.
3 원격탐사    DB로 부족할 때만 scan_alphaearth_change(임의 연도쌍) / run_change_model(2019→2025) 호출.
4 GIS 분석    hotspot_analysis 로 공간 군집 확인.
5·6 대안 평가  우선순위가 필요하면 rank_priority_units. 사용한 가중치와 민감도(top_stability)를 반드시 제시.
7 결과 제공   make_decision_card 로 카드 생성. 마지막에 카드 경로와 핵심 3줄 요약을 답변.

규칙:
- 면적·비율·순위 등 모든 수치는 도구 출력값을 그대로 인용한다. 직접 계산·추정·반올림 변경 금지.
- 도구 meta.warnings 는 카드 caveats 에 반영한다.
- 임상도 변화연도는 도면 갱신년도이며 실제 발생연도와 다를 수 있음을 명시한다.
- 법령·지침 적용 판단은 하지 않는다(보유 근거 없음). 필요하면 '담당자 확인 필요'로 표시한다.
- 답변은 한국어."""


class ForestChangeAgent:
    def __init__(self, model: str = C.AGENT_MODEL, max_turns: int = 20, verbose: bool = True):
        self.client = anthropic.Anthropic()
        self.model, self.max_turns, self.verbose = model, max_turns, verbose

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def run(self, query: str, session: T.Session | None = None) -> tuple[str, T.Session]:
        session = session or T.Session(query=query)
        messages = [{"role": "user", "content": query}]
        for _ in range(self.max_turns):
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                tools=T.TOOL_SCHEMAS,
                messages=messages,
                thinking={"type": "adaptive"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",             # 안전 분류기 거절 시 서버측 대체모델로 재시도
            )
            messages.append({"role": "assistant", "content": resp.content})
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    self._log(f"\n[Agent] {b.text}")

            if resp.stop_reason == "refusal":
                return "요청이 거절되었습니다: " + str(getattr(resp.stop_details, "explanation", "")), session
            if resp.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                final = "\n".join(b.text for b in resp.content if b.type == "text")
                return final, session
            if resp.stop_reason == "max_tokens":
                raise RuntimeError("응답이 max_tokens 에서 잘렸습니다 (도구 입력 불완전).")

            results = []
            for tu in tool_uses:
                self._log(f"[Tool] {tu.name} {json.dumps(tu.input, ensure_ascii=False)[:200]}")
                try:
                    if tu.name not in T.TOOLS or not isinstance(tu.input, dict):
                        raise ValueError(f"잘못된 도구 호출: {tu.name}")
                    out = T.call(session, tu.name, tu.input)
                    content, err = json.dumps(out, ensure_ascii=False, default=str), False
                    w = out.get("meta", {}).get("warnings")
                    self._log(f"       ✓ {len(content):,} bytes" + (f"  ⚠ {w}" if w else ""))
                except Exception as e:  # 도구 실패는 모델에 오류로 돌려주고 계속 진행
                    content, err = f"{type(e).__name__}: {e}", True
                    self._log(f"       ✗ {content}")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content, "is_error": err})
            messages.append({"role": "user", "content": results})
        return "최대 반복 횟수에 도달했습니다.", session


def run_demo(query: str, region: str = "북산면") -> T.Session:
    """LLM 없이 도구 파이프라인을 고정 순서로 실행 (API 키 없는 환경의 동작 확인용)."""
    s = T.Session(query=query)

    def step(name, **kw):
        out = T.call(s, name, kw)
        print(f"\n[{name}] " + json.dumps(out["data"], ensure_ascii=False, default=str)[:600])
        if out["meta"]["warnings"]:
            print("   ⚠", out["meta"]["warnings"])
        return out["data"]

    spec = step("set_analysis_spec", region=region, year_from=2019, year_to=2025,
                change_types=["harvest", "conversion"], policy_purpose="벌채·산지전용 현장점검 대상 선정")
    rid = spec["region_id"]
    ch = step("query_change_map", region_id=rid, change_types=["harvest", "conversion"])
    step("query_mgmt_plan", region_id=rid)
    hs = step("hotspot_analysis", region_id=rid, cell_m=250, band_m=750)
    rk = step("rank_priority_units", region_id=rid, top_n=5)
    top = rk["ranking"][0]
    step("make_decision_card", title=f"{ch['region']} 산림변화 현장점검 우선순위 (데모)",
         summary=f"{ch['region']}의 {ch['period']} 벌채·전용 면적은 {ch['changed_total_ha']} ha 입니다.",
         findings=[f"벌채·갱신 {ch['by_type']['harvest']['area_ha']} ha, 산지전용 {ch['by_type']['conversion']['area_ha']} ha",
                   f"Gi* 핫스팟 격자 {hs['n_hot']}개에 {hs['hot_changed_ha']} ha 집중"],
         recommendations=[f"1순위 임반 {top['compartment']} (점수 {top['score']}, 순위안정성 {top['top_stability']})"],
         caveats=["데모 모드: 서술문은 템플릿으로 생성됨"])
    return s
