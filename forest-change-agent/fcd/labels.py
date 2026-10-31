"""임상도 두 시기(2019→2025) 속성 비교에 의한 산림변화 라벨 규칙.

임상 코드(FRTP_CD): 0 무립목지/비산림, 1 침엽수림, 2 활엽수림, 3 혼효림
수종군(KOFTR_GROU): 81 미립목지, 82 제지, 83 이하·9x 비산림(경작지·초지 등)
영급(AGCLS_CD): 1(1~10년) … 9
"""
import numpy as np
import pandas as pd

CHANGE_TYPES = {
    "no_change": "무변화",
    "harvest": "벌채·갱신 (영급 초기화 또는 미립목지화)",
    "conversion": "산지전용·개발 (산림→비산림)",
    "gain": "산림 증가 (조림·자연회복)",
    "type_shift": "임상 변화 (검토 필요)",
    "ambiguous": "판정 보류",
    "nonforest": "비산림 유지",
}
TRAIN_CLASSES = ["no_change", "harvest", "conversion", "gain"]   # 학습에 쓰는 신뢰 라벨
FOREST = {"1", "2", "3"}


def classify(df: pd.DataFrame, y0: int = 2019, y1: int = 2025) -> pd.DataFrame:
    """`FRTP_{y}`, `KOFTR_{y}`, `AGCLS_{y}` 컬럼을 가진 표에 change_type / label_conf / change_year 부여."""
    f0 = df[f"FRTP_{y0}"].isin(FOREST)
    f1 = df[f"FRTP_{y1}"].isin(FOREST)
    a0 = pd.to_numeric(df[f"AGCLS_{y0}"], errors="coerce")
    a1 = pd.to_numeric(df[f"AGCLS_{y1}"], errors="coerce")
    k1 = df[f"KOFTR_{y1}"].astype(str)
    same_type = df[f"FRTP_{y0}"] == df[f"FRTP_{y1}"]

    cond = [
        ~f0 & ~f1,
        f0 & ~f1 & (k1 == "81"),                        # 산림 → 미립목지 : 벌채
        f0 & ~f1,                                       # 산림 → 제지·비산림 : 전용
        ~f0 & f1,                                       # 비산림/미립목 → 산림
        f0 & f1 & (a1 <= 1) & (a0 >= 3),                # 영급 초기화 : 벌채 후 갱신
        f0 & f1 & same_type & (a1 >= a0) & (a1 <= a0 + 2),
        f0 & f1 & ~same_type & (a1 >= a0),              # 영급 유지 + 임상만 바뀜
    ]
    choice = ["nonforest", "harvest", "conversion", "gain", "harvest", "no_change", "type_shift"]
    out = df.copy()
    out["change_type"] = np.select(cond, choice, default="ambiguous")
    out["changed"] = out["change_type"].isin(["harvest", "conversion", "gain"]).astype("int8")

    # 발생연도: 2025 임상도 갱신년도가 비교구간(y0,y1] 안이면 약한 라벨로 사용
    yr = pd.to_numeric(out.get("UPDATE_YEAR"), errors="coerce")
    out["change_year"] = yr.where(out["changed"].eq(1) & yr.between(y0 + 1, y1)).astype("Int16")

    out["label_conf"] = np.where(out["change_type"].isin(TRAIN_CLASSES), "high",
                         np.where(out["change_type"] == "type_shift", "low", "none"))
    return out
