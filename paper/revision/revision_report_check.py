"""E8 보고서 검증 범위 확대: 저장된 모든 에이전트 보고서의 (1) 수치 일치(추적률), (2) 의미 정확성 —
보고서 표의 소반ID 행에 함께 적힌 읍면동·순위·점수·순위안정성이 같은 세션의 도구 출력(evidence.json)과 일치하는지,
(3) 근거 추적(도구 호출 기록 유무), (4) 과도한 해석(원인·위법 단정 표현) 여부를 점검한다.
"""
import json
import re
from pathlib import Path

import pandas as pd

RUNS = Path(r"C:\src\ForestChangeGeoAI\forest_decision_agent\outputs")
OUT = Path(__file__).resolve().parent / "results"
OVERCLAIM = re.compile(r"(불법|무단|위법(?!\s*여부)|원인은|때문이다|확정적으로)")


def rank_rows(log):
    rows = {}
    for e in log:
        if e["tool"] == "rank_inspection":
            for i, r in enumerate(e["output"].get("ranking", []), 1):
                rows[r["단위ID"]] = {**r, "순위": i}
    return rows


def main():
    out = []
    for d in sorted(RUNS.glob("2*")):
        rep, ev = d / "report.md", d / "evidence.json"
        if not rep.exists() or not ev.exists():
            continue
        md = rep.read_text(encoding="utf-8")
        log = json.loads(ev.read_text(encoding="utf-8"))["log"]
        qj = json.loads((d / "query.json").read_text(encoding="utf-8")) if (d / "query.json").exists() else {}
        m = re.search(r"수치 추적률: ([\d.]+) \(보고서 수치 (\d+)개", md)
        rr = rank_rows(log)
        checked = ok = 0
        bad = []
        for line in md.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            ids = [c for c in cells if re.fullmatch(r"\d{3}-\d{3}/\d{3}-\d{3}", c)]
            if not ids or ids[0] not in rr or not re.fullmatch(r"\d+", cells[0] or ""):
                continue
            r = rr[ids[0]]
            checked += 1
            place = {r["읍면동"], *[w for w in str(r.get("소재지") or "").split() if w.endswith(("리", "동", "면", "읍"))]}
            claims = {"순위": int(cells[0]), "읍면동": r["읍면동"] if place & set(cells) else None,
                      "점수": r["점수"] if str(r["점수"]) in cells else None}
            good = claims["순위"] == r["순위"] and claims["읍면동"] is not None and claims["점수"] is not None
            ok += good
            if not good:
                bad.append(ids[0])
        body = md.split("## 근거 기록")[0]
        sents = re.split(r"(?<=[.다요])\s+|\n", body)
        over = sorted({w for st in sents if not re.search(r"(않|아닙|아니|판단하지|확인되지|필요)", st) for w in OVERCLAIM.findall(st)})
        out.append({"실행": d.name, "모드": qj.get("mode", "demo/CLI"), "질의": (qj.get("query") or "")[:50],
                    "도구호출": len(log), "도구종류": len({e["tool"] for e in log}),
                    "위성도구사용": any(e["tool"] in ("evaluate_foundation_model", "unit_timeseries") for e in log),
                    "수치검사": int(m.group(2)) if m else None, "수치추적률": float(m.group(1)) if m else None,
                    "순위표_행검사": checked, "순위표_의미일치": ok, "불일치행": bad, "과도해석_표현": over})
    df = pd.DataFrame(out)
    df.to_csv(OUT / "E8_report_checks.csv", index=False, encoding="utf-8-sig")
    print(df.drop(columns=["질의"]).to_string(index=False))


if __name__ == "__main__":
    main()
