"""[4.4.2] 결과 교차검증: 면적 합계·좌표계·수치 근거 일치(grounding) 검사."""
import re

YEAR_RANGE = range(1990, 2031)
_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def area_consistency(parts: dict, total: float, tol: float = 0.01) -> str | None:
    s = sum(parts.values())
    if total and abs(s - total) > max(tol * total, 0.05):
        return f"면적 불일치: 부분합 {s:.2f} ha ≠ 전체 {total:.2f} ha"
    return None


def crs_check(gdf, expected: str) -> str | None:
    if gdf.crs is None or gdf.crs.to_string() != expected:
        return f"좌표계 불일치: {gdf.crs} (기대값 {expected})"
    return None


def collect_numbers(obj, acc=None) -> list[float]:
    """도구 출력(JSON)에 들어있는 모든 수치."""
    acc = [] if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_numbers(v, acc)
            try:                          # {"2021": 3.2} 처럼 키에 든 수치(연도·영급)도 근거로 인정
                acc.append(float(k))
            except (TypeError, ValueError):
                pass
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, acc)
    return acc


def grounding(text: str, evidence: list[float]) -> dict:
    """문장 속 수치가 도구 출력 수치(반올림·백분율 허용)에 근거하는지 검사."""
    ev = [v for v in evidence] + [v * 100 for v in evidence if abs(v) <= 1]
    text = re.sub(r"\b\d{3}-\d{3}\b", " ", text)       # 임반·소반 번호(067-000)는 수치가 아님
    checked, unmatched = 0, []
    for tok in _NUM.findall(text):
        x = float(tok.replace(",", ""))
        if (x.is_integer() and (int(x) in YEAR_RANGE or 0 <= x <= 10)):
            continue                      # 연도·항목번호는 제외
        checked += 1
        if not any(abs(x - v) <= max(0.051, 0.005 * abs(v), 10 ** -(len(tok.split(".")[-1]) if "." in tok else 0) * 0.51)
                   for v in ev):
            unmatched.append(tok)
    return {"numbers_checked": checked, "unmatched": unmatched,
            "grounded_ratio": 1.0 if checked == 0 else round(1 - len(unmatched) / checked, 3)}
