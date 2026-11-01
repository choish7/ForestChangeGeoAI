"""[4.4.2] Geo-LLM 산림변화 에이전트 — Google Gemini 백엔드 (function calling, 수동 루프).

도구·시스템 프롬프트·검증 로직은 Claude 백엔드(agent.py)와 동일하게 공유한다.
인증: 환경변수 GEMINI_API_KEY (키를 코드·파일에 넣지 말 것)
"""
import json
import random
import time

from google import genai
from google.genai import errors, types

from . import config as C
from . import tools as T
from .agent import SYSTEM

RETRYABLE = {429, 500, 503, 504}          # 과부하·속도제한·일시 장애


class GeminiForestChangeAgent:
    def __init__(self, model: str = C.GEMINI_MODEL, max_turns: int = 20, verbose: bool = True):
        self.client = genai.Client()                     # GEMINI_API_KEY 자동 사용
        self.model, self.max_turns, self.verbose = model, max_turns, verbose
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=s["name"], description=s["description"],
                                          parameters_json_schema=s["input_schema"])
                for s in T.TOOL_SCHEMAS])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _generate(self, contents):
        """일시 오류(429/5xx)는 지수 백오프로 재시도하고, 계속 실패하면 대체 모델로 전환."""
        last = None
        for model in [self.model, *[m for m in C.GEMINI_FALLBACK_MODELS if m != self.model]]:
            for attempt in range(C.GEMINI_MAX_RETRIES):
                try:
                    resp = self.client.models.generate_content(model=model, contents=contents, config=self.config)
                    if model != self.model:
                        self._log(f"[LLM] 대체 모델 {model} 로 응답")
                    return resp
                except errors.APIError as e:
                    if e.code not in RETRYABLE:
                        raise
                    last = e
                    wait = min(60, 2 ** attempt * 3) + random.uniform(0, 1)
                    self._log(f"[LLM] {model} {e.code} {e.status} — {wait:.0f}s 후 재시도 "
                              f"({attempt + 1}/{C.GEMINI_MAX_RETRIES})")
                    time.sleep(wait)
            self._log(f"[LLM] {model} 재시도 소진 → 다음 모델")
        raise last

    def run(self, query: str, session: T.Session | None = None) -> tuple[str, T.Session]:
        session = session or T.Session(query=query)
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]
        for _ in range(self.max_turns):
            try:
                resp = self._generate(contents)
            except errors.APIError as e:
                return self._partial_result(session, e), session
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None:
                reason = getattr(resp.prompt_feedback, "block_reason", None) or getattr(cand, "finish_reason", None)
                return f"응답 없음 (사유: {reason})", session
            contents.append(cand.content)                 # thought signature 포함 그대로 보존
            text = "".join(p.text for p in cand.content.parts or [] if p.text and not p.thought)
            if text.strip():
                self._log(f"\n[Agent] {text}")

            calls = resp.function_calls or []
            if not calls:
                return text, session

            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                self._log(f"[Tool] {fc.name} {json.dumps(args, ensure_ascii=False)[:200]}")
                try:
                    if fc.name not in T.TOOLS:
                        raise ValueError(f"잘못된 도구 호출: {fc.name}")
                    out = T.call(session, fc.name, args)
                    payload = json.loads(json.dumps(out, ensure_ascii=False, default=str))
                    w = out.get("meta", {}).get("warnings")
                    self._log(f"       ✓" + (f"  ⚠ {w}" if w else ""))
                    response = {"result": payload}
                except Exception as e:  # 도구 실패는 모델에 오류로 돌려주고 계속 진행
                    self._log(f"       ✗ {type(e).__name__}: {e}")
                    response = {"error": f"{type(e).__name__}: {e}"}
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=fc.id, name=fc.name, response=response)))
            contents.append(types.Content(role="user", parts=parts))
        return "최대 반복 횟수에 도달했습니다.", session

    @staticmethod
    def _partial_result(session: T.Session, err: Exception) -> str:
        """LLM 호출이 끝내 실패해도 이미 계산·생성된 결과는 돌려준다."""
        done = [a["tool"] for a in session.audit]
        msg = [f"LLM 호출 실패로 중단: {getattr(err, 'code', '')} {getattr(err, 'message', err)}",
               f"완료된 도구: {', '.join(done) or '없음'}"]
        cards = sorted((C.OUT / "cards").glob("card_*.md")) if "make_decision_card" in done else []
        if cards:
            msg.append(f"의사결정 카드는 생성됨: {cards[-1]}")
        else:
            msg.append("잠시 후 다시 실행하거나 GEMINI_MODEL 을 다른 모델로 지정하세요.")
        return "\n".join(msg)
