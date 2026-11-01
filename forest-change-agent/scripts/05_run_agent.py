"""[4.4.2] 산림변화 GeoAI 에이전트 실행.

  python scripts/05_run_agent.py "북산면 국유림에서 2019~2025년 벌채·산지전용 현황과 현장점검 우선 임반을 알려줘"
  python scripts/05_run_agent.py --demo            # LLM 없이 도구 파이프라인만 실행
  python scripts/05_run_agent.py -i                # 대화형

백엔드: --provider gemini|claude (기본: GEMINI_API_KEY 가 있으면 gemini)
인증: GEMINI_API_KEY 또는 ANTHROPIC_API_KEY 환경변수
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from fcd import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--demo", action="store_true", help="LLM 없이 고정 도구 파이프라인 실행")
    ap.add_argument("--region", default="북산면", help="--demo 대상 지역")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--provider", choices=["gemini", "claude"], default=C.LLM_PROVIDER)
    ap.add_argument("--model", default=None, help="모델 ID (기본: GEMINI_MODEL / FOREST_AGENT_MODEL)")
    args = ap.parse_args()

    if args.demo:
        from fcd.agent import run_demo
        s = run_demo(args.query or f"{args.region} 벌채·전용 현장점검 우선순위", args.region)
        print(f"\n[데모 완료] 도구 호출 {len(s.audit)}회 → outputs/cards/")
        return

    if args.provider == "gemini":
        from fcd.agent_gemini import GeminiForestChangeAgent
        agent = GeminiForestChangeAgent(model=args.model or C.GEMINI_MODEL)
    else:
        from fcd.agent import ForestChangeAgent
        agent = ForestChangeAgent(model=args.model or C.AGENT_MODEL)
    print(f"[LLM] {args.provider} / {agent.model}")
    if args.interactive:
        while True:
            q = input("\n질의> ").strip()
            if q in ("", "exit", "quit"):
                break
            ans, _ = agent.run(q)
            print(f"\n=== 답변 ===\n{ans}")
    else:
        q = args.query or "북산면 국유림에서 2019~2025년 벌채·산지전용 현황과 현장점검이 필요한 우선 임반을 알려줘"
        ans, s = agent.run(q)
        print(f"\n=== 답변 ===\n{ans}\n\n(도구 호출 {len(s.audit)}회)")


if __name__ == "__main__":
    main()
