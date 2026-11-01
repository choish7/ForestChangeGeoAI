"""영문 화면용 번역: 규칙 기반 보고서와 변화 해석 문장을 영문으로 바꾼다 (API 없이 결정적 규칙).

보고서·해석 문장은 정해진 템플릿에서 만들어지므로 ① 문장 템플릿(정규식) → ② 용어집(긴 말부터) 순서로 바꾸고,
수치는 건드리지 않는다. 지명·주소(읍면동·리·번지)는 한글 그대로 둔다. 남은 한글은 leftovers() 로 확인한다.
"""
from __future__ import annotations

import re
from collections import Counter

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
HANGUL = re.compile(r"[가-힣][가-힣·]*")

# ① 문장 템플릿 — 어순이 바뀌는 문장만 (적용 순서대로)
TEMPLATES = [
    # 변화 해석: 관측
    (r"AlphaEarth 임베딩 변화는 (\S+?)에 가장 컸고\(거리 (\S+?)\), NDVI는 변화 전 (\S+) → 최저 (\S+) → 2025년 (\S+?)\(회복률 (\S+?)\), "
     r"NDMI 감소 (\S+?), NDBI 증가 (\S+?)였다\.",
     r"AlphaEarth embedding change was largest in \1 (distance \2); NDVI went from \3 before the change → minimum \4 → \5 in 2025 "
     r"(recovery ratio \6), NDMI decrease \7, NDBI increase \8."),
    (r"고변화 연도: ([\d,]+)\.", r"High-change years: \1."),
    (r"학습모형은 (.+?)\((\S+?)년\)으로 분류하였다\.", r"The trained model classified it as \1 (\2)."),
    (r"AlphaEarth 분류모델: 산림 비율 (\S+?)→(\S+?), 2025 나지 (\S+?)·시설지\(제지\) (\S+?); 우세 임상 (\S+?)→(\S+?)\(임상도 (\S+?)→(\S+?)\); "
     r"변화유형 벌채·갱신 (\S+) ha, 산림증가 (\S+) ha\.",
     r"AlphaEarth classifiers: forest share \1→\2, 2025 bare \3 · non-forest use \4; dominant forest type \5→\6 (forest type map \7→\8); "
     r"change type harvest/regeneration \9 ha, forest gain \10 ha."),
    # 변화 해석: 기록
    (r"임상도 2019 (.+?) → 2025 (.+?) \(변화비율 (\S+?)\); 500 m 내 산불 (\d+)·산사태 (\d+)건\.",
     r"forest type map 2019 \1 → 2025 \2 (change ratio \3); within 500 m: wildfires \4 · landslides \5."),
    (r"임상도 조사연도 (\S+?)\(마지막 (\S+?)년\)", r"Forest-map survey years \1 (latest \2)"),
    (r", 입목 변화 구간: ", r", stocking change intervals: "),
    (r"; AE 연도별 임상 추정: ", r"; AE yearly forest-type estimate: "),
    (r"(\d+)영급", r"age class \1"),
    # 변화 해석: 불확실성
    (r"연간 값은 변화 화소 평균", "annual values are means over change pixels"),
    (r"연간 값은 소반 전체 평균", "annual values are means over the whole stand"),
    (r"변화구간 끝 연도 (\d+)년은 AlphaEarth 연도별 오탐률이 높은 해\(안정 조각 (\S+?)\)",
     r"the interval end year \1 has a high AlphaEarth false-alarm rate (stable patches \2)"),
    # 보고서
    (r"산림변화 의사결정 지원 보고서 — ", "Forest Change Decision Support Report — "),
    (r"- 국유림 소반 (\d+)개, (\S+) ha\. AlphaEarth 변화면적 (\S+) ha, Sentinel-2 변화면적 (\S+) ha\.",
     r"- \1 national-forest stands, \2 ha. AlphaEarth change area \3 ha, Sentinel-2 change area \4 ha."),
    (r"학습 모델 변화면적\(ha\): ", "Trained-model change area (ha): "),
    (r"신호일치수별 소반\(AE 임베딩거리·S2-TSFM·최종지도 중 변화로 본 수\)",
     "Stands by signal agreement (how many of AE embedding distance, S2-TSFM and final map flag change)"),
    (r"최종지도 변화연도별 소반수", "stands by final-map change year"),
    (r"현장점검 우선순위 \(가중치 ", "Field inspection priority (weights "),
    (r"상위 (\d+)곳 원자료 시계열", r"Source time series of the top \1"),
    (r"## (.+?) 소반 \((\d+)곳\)", r"## \1 stands (\2)"),
    (r"회복 양호지\((.+?)\)와 임베딩이 유사한 소반 \(비교·검증 후보\)",
     r"Stands with embeddings similar to well-recovered sites (\1) (candidates for comparison and validation)"),
    (r"임상도 갱신 검토 후보 (\d+)곳 · 판독 수정 가능 (\d+)곳", r"\1 forest-map update candidates · \2 possible re-interpretations"),
    (r"변화 구역 (\d+)개 \((\S+) ha, 경계 확인 (\d+)개\)", r"\1 change zones (\2 ha, \3 need boundary checks)"),
    (r"AlphaEarth 분류모델 현황 \((.+?), 소반 (\d+)곳\)", r"AlphaEarth classifier status (\1, \2 stands)"),
    (r"피복 (\S+) \(ha\)", r"Land cover: \1 (ha)"),
    (r"임상 (침엽수림|활엽수림|혼효림) \(ha\)", r"Forest type: \1 (ha)"),
    (r"- 변화유형 분류\(2019→2025\): 벌채·갱신 (\S+) ha, 산림증가 (\S+) ha\.",
     r"- Change-type classification (2019→2025): harvest/regeneration \1 ha, forest gain \2 ha."),
    (r"임상 재판독 후보 (\d+)곳 · 제지 확대 (\d+)곳 · 수확기록 없는 미립목지화 (\d+)곳",
     r"\1 forest-type re-interpretation candidates · \2 non-forest-use expansions · \3 unstocked without harvest records"),
    (r"산림비율 (\S+?)→(\S+)", r"forest share \1→\2"),
    (r"제지 (\S+?)·나지 (\S+)", r"non-forest use \1 · bare \2"),
    (r"제지 \+(\S+?), 나지 \+(\S+)", r"non-forest use +\1, bare +\2"),
    (r"평균확률 (\S+)", r"mean probability \1"),
    (r"임상도 재조사 우선순위 \(후보 (\d+)곳\)", r"Forest-map re-survey priority (\1 candidates)"),
    (r"- 상위 차원: ", "- Top dimensions: "),
    (r"수치 추적률: (\S+) \(보고서 수치 (\d+)개 중 도구 출력에서 확인되지 않은 값: (.+?)\)",
     r"numeric traceability: \1 (values among the report's \2 numbers not found in tool outputs: \3)"),
    (r"마지막 조사 (\d+)년 이상 경과", r"\1+ years since last survey"),
    (r"(.+?) 점검 우선순위 소반", r"\1 — inspection priority stands"),
    (r"선택 소반 (\d+)개", r"\1 selected stands"),
    (r"지도 범위 ", "map extent "),
    (r"(\d+)곳", r"\1"),
]
TEMPLATES = [(re.compile(p), r) for p, r in TEMPLATES]

# ② 용어집 — 문장 → 구 → 단어 (길이순으로 자동 정렬)
GLOSSARY = {
    # 해석 문장 (build_trajectory.INTERP)
    "교란(벌채·산불 등) 후 식생이 회복되는 과정과 부합한다.": "Consistent with vegetation recovering after a disturbance (harvest, fire, etc.).",
    "교란 후 식생 회복이 더디게 진행되는 상태와 부합한다.": "Consistent with slow vegetation recovery after a disturbance.",
    "산림이 다른 피복(시설·도로·나지)으로 지속적으로 바뀐 상태와 부합한다.": "Consistent with forest having persistently changed to another cover (facilities, roads, bare land).",
    "최근 교란이 관측되었으나 이후 경과를 판단할 자료가 부족하다.": "A recent disturbance was observed, but there is not yet enough data to judge what follows.",
    "녹색도보다 수분 신호가 더 크게 줄어, 수분 스트레스·병해충 등 식생 이상 가능성이 있다.": "The moisture signal fell more than greenness, suggesting possible vegetation stress (drought, pests, etc.).",
    "식생이 늘어나는 변화(조림지 생장, 산림증가)와 부합한다.": "Consistent with increasing vegetation (plantation growth, forest gain).",
    "식생지수로는 설명되지 않는 구조·피복 변화이거나 자료 영향일 수 있다.": "May be a structural or cover change not explained by vegetation indices, or a data artefact.",
    "변화 신호는 약하지만 식생지수가 꾸준히 높아져 생장과 부합한다.": "The change signal is weak, but vegetation indices rose steadily, consistent with growth.",
    "뚜렷한 변화 신호가 없다.": "No clear change signal.",
    " (‘부합’ 수준의 설명이며 원인 확정이 아님)": " (an explanation at the 'consistent with' level, not a confirmed cause)",
    "벌채·산불·조림 기록, 조림 수종과 원래 수종의 차이": "harvest, fire and planting records; planted vs. original species",
    "조림·복구 사업 시행 여부, 토사 유출, 현장 활착 상태": "whether planting or restoration was carried out, soil runoff, seedling establishment on site",
    "고해상도 영상, 산지전용 인허가, 토지이용 자료": "high-resolution imagery, forest conversion permits, land-use data",
    "다음 해 영상, 벌채·재해 기록": "next year's imagery, harvest and disaster records",
    "강수·기온, 병해충 조사, 관측 품질(구름 잔재)": "precipitation and temperature, pest surveys, observation quality (residual cloud)",
    "초지·관목·산림 구분, 조림 기록": "distinguishing grass, shrub and forest; planting records",
    "원영상, 관측 조건, 다른 지표": "source imagery, observation conditions, other indicators",
    "영급·밀도 변화": "age-class and density change",
    "임상도가 그대로라는 사실만으로 무변화를 확정하지 않음": "an unchanged forest type map alone does not confirm no change",
    "변화가 소반 일부면 신호가 희석됨": "the signal is diluted if change covers only part of the stand",
    "여러 해에 걸친 반복 변화 — 집중 감시 대상": "repeated change over several years — priority for monitoring",
    "판정 기준값은 춘천 자료의 잠정값": "thresholds are provisional values from Chuncheon data",
    "(20 m 지수, 10 m 임베딩)": "(20 m indices, 10 m embeddings)",
    # 보고서 고정 문장
    "- 관측:": "- Observation:", "- 기록:": "- Records:", "- 해석:": "- Interpretation:", "- 추가 확인:": "- Check further:",
    "- 불확실성:": "- Uncertainty:", "사업실적:": "Management records:", "500 m 내": "within 500 m",
    # 자료 출처 (tools.SOURCES)
    "64차원, 연속연도 거리": "64 dimensions, year-to-year distance", "연간 지수": "annual indices", "54밴드": "54 bands",
    "AlphaEarth+S2 지수 HistGradientBoosting, 산림 마스크·최소면적": "AlphaEarth + S2 indices HistGradientBoosting, forest mask and minimum area",
    "Sentinel-2 시계열 마스크 오토인코더 + 분류기": "Sentinel-2 time-series masked autoencoder + classifier",
    "국유림 경영계획부 7·8·9차 산림사업실적 (춘천국유림관리소, 임소반 결합)":
        "national-forest management plan records, 7th, 8th and 9th plans (Chuncheon National Forest Office, joined to compartments/stands)",
    "임상도 2019·2025": "forest type map 2019 and 2025", "춘천시 보전산지 해제 고시": "Chuncheon conservation forest-land release notices",
    "세부내역 + 연속지적도": "details + cadastral map", "산림청 산불·산사태 신고정보 (춘천시 범위)": "Korea Forest Service wildfire and landslide reports (Chuncheon)",
    "임도망도, 1:1,000 수치지형도 도로": "forest road network, 1:1,000 digital topographic map roads",
    "최종지도 모델": "final-map model", "S2-TSFM 모델": "S2-TSFM model","접근성": "accessibility", "사업실적": "management records",
    "임상도": "forest type map", "최종지도": "final map", "없음": "none",
    "초지 결과는 정확도가 낮아 판단에 쓰지 않았고, 시설지는 임도·묘지 등 산지 안 산림 외 용도(제지)를 뜻합니다.":
        "Grass results were not used because of low accuracy; 'non-forest use' means forest roads, graves and other non-forest uses within forest land.",
    "상관은 관련성일 뿐이며 차원에 '수분', '영급' 같은 이름을 붙이는 근거가 아닙니다.":
        "Correlation shows association only; it is no basis for naming a dimension 'moisture' or 'age class'.",
    "위성 변화는 관측 사실이며 원인이 아닙니다. 위법 여부·법령 적용은 담당자 확인이 필요합니다.":
        "Satellite change is an observation, not a cause. Legality and legal application must be confirmed by the officer in charge.",
    "변화 해석은 임베딩 차원의 의미가 아니라 임베딩 변화 시기, 지수 추이, 임상도·실적 맥락이 같은 해석을 지지하는지로 판단했습니다.":
        "Change was interpreted not from the meaning of embedding dimensions but from whether the timing of embedding change, index trends and the forest-map and record context support the same reading.",
    "사업실적·판정은 국유림 소반만 대상입니다.": "Management records and judgements cover national-forest stands only.",
    "LLM 미연결: 규칙 기반 템플릿 보고서입니다(질의 해석 없이 지역 요약·우선순위 위주).":
        "No LLM: rule-based template report (area summary and priorities, without interpreting the query).",
    "임상도 미반영 위성변화 3 + AE 연도별 우세임상 전환 1.5 + 조사 경과(10년=1) + 최종지도 변화비율 − 고오탐 연도 단독신호 1":
        "satellite change not in forest map 3 + AE yearly dominant-type switch 1.5 + years since survey (10 years = 1) + final-map change ratio − lone signal in high-false-alarm year 1",
    "위성 변화가 마지막 조사 이후 — 임상도 미반영": "satellite change after last survey — not in forest map",
    "임상 간(판독 수정 가능)": "between forest types (possible re-interpretation)",
    "국유림 경계 밖으로 이어짐": "extends beyond national-forest boundary",
    "여러 소반에 걸침": "spans several stands",
    "춘천시 전체 기준": "all of Chuncheon",
    "춘천시 전체": "All of Chuncheon",
    "오탐률 5%에서 변화화소 탐지율": "Change-pixel detection rate at 5% false-alarm rate",
    "소반 판별 AUROC(라벨 없이 계산한 신호)": "Stand discrimination AUROC (signal computed without labels)",
    "학습 모델 홀드아웃 macro F1": "Trained-model holdout macro F1",
    "임상도와 우세임상 일치율": "Agreement with forest-map dominant type",
    "임베딩 차원과 NDVI 변화의 관계 (탐색)": "Embedding dimensions vs. NDVI change (exploratory)",
    "상위 후보 변화 해석 (관측 → 기록 → 해석)": "Change interpretation of top candidates (observation → records → interpretation)",
    "판정별 현황 (관측·기록 대조)": "By judgement (observation vs. records)",
    "AlphaEarth 활용 가치": "Value of AlphaEarth",
    "산지전용 고시 필지": "Forest conversion notice parcels",
    "해석 시 유의": "Caveats",
    "근거 기록": "Evidence log",
    "자료 출처": "Data sources",
    "질의:": "Query:",
    "요약": "Summary",
    # 표 머리글·라벨
    "최종지도 변화연도(화소)": "final-map change year (pixels)", "AE 최대 변화 구간": "AE peak change interval",
    "AE 거리 2019→2025": "AE distance 2019→2025", "NDVI 변화전 → 최저 → 2025": "NDVI before → min → 2025",
    "AE 변화비율 중앙값": "AE change ratio (median)", "해제면적 m²": "released area m²", "AE 변화비율": "AE change ratio",
    "S2 변화비율": "S2 change ratio", "AE 변화 ha": "AE change ha", "S2 변화 ha": "S2 change ha", "모델 변화 ha": "model change ha",
    "임상도 변화비율": "forest-map change ratio", "2019 → 2025 수종": "species 2019 → 2025", "임상·수종(2025)": "type·species (2025)",
    "국유림 밖 ha": "outside national forest ha", "연도 신뢰도 낮음": "low-reliability year", "AE 연도별 우세임상": "AE yearly dominant type",
    "위성 변화연도": "satellite change year", "마지막 조사": "last survey", "재해(500 m)": "disasters (500 m)", "수종(2019)": "species (2019)",
    "임상도 2025": "forest map 2025", "AE 분류": "AE class", "걸친 소반": "stands covered", "경계 확인": "boundary check",
    "모델 유형": "model type", "순위안정성": "rank stability", "신호일치": "signal agreement", "변화연도": "change year",
    "고변화연도": "high-change years", "변화구간": "change interval", "변화경로": "change path", "면적 ha": "area ha", "경사차 °": "slope diff. °",
    "표고차 m": "elevation diff. m", "유사도": "similarity", "회복률": "recovery ratio", "읍면동": "eup/myeon/dong", "레코드": "record",
    "주소": "address", "용도": "use", "라벨": "label", "필지": "parcels", "순위": "rank", "점수": "score", "사유": "reason", "근거": "basis",
    "구분": "item", "구역": "zone", "유형": "type", "실적": "records", "판정": "judgement", "소반": "stand", "점수:": "score:",
    "갱신 검토": "update review", "판독 수정": "re-interpretation", "임상 재판독": "type re-interpretation", "제지 확대": "non-forest use expansion",
    "미립목지화": "became unstocked",
    # 판정·경로·유형 값
    "미기록 변화(점검 필요)": "unrecorded change (inspect)", "기타실적·변화관측(확인)": "other records + change (check)",
    "기록일치(수확실적·변화관측)": "records match (harvest + change)", "실적있으나 변화 미관측(이행 확인)": "records but no change (verify)",
    "교란 후 회복 지연": "delayed recovery after disturbance", "교란 후 회복": "recovery after disturbance",
    "지속 손실·전환 의심": "persistent loss / conversion suspected", "최근 교란(회복 판단 이름)": "recent disturbance (too early to judge)",
    "수분 스트레스 등 식생 이상 의심": "possible vegetation stress (e.g. moisture)", "식생 증가·성장 (변화 신호 동반)": "vegetation gain (with change signal)",
    "임베딩 변화·지수 안정": "embedding change, indices stable", "식생 증가·성장 추정": "likely vegetation growth", "반복 교란": "repeated disturbance",
    "전용확정": "confirmed conversion", "전용후보": "conversion candidate", "전용이전": "pre-2019 conversion",
    "벌채·갱신": "harvest/regeneration", "산지전용": "forest conversion", "산림증가": "forest gain", "수확·갱신": "harvest/regeneration",
    "피해·기타": "damage/other", "실적없음": "no records", "솎아베기": "thinning",
    "위성변화": "satellite change", "기록없음": "no records", "재해이력": "disaster history", "접근용이": "accessibility",
    "산불": "wildfire", "산사태": "landslide", "재해": "disasters", "안정": "stable",
    # 임상·수종·영급
    "침활혼효림": "mixed conifer-broadleaf", "침엽수림": "coniferous", "활엽수림": "broadleaf", "혼효림": "mixed",
    "무립목지": "unstocked", "미립목지": "unstocked", "입목지": "stocked", "비산림": "non-forest", "관목덤불": "shrub thicket",
    "기타참나무류": "other oaks", "기타침엽수": "other conifers", "기타활엽수": "other broadleaves", "리기다소나무": "pitch pine",
    "신갈나무": "Mongolian oak", "굴참나무": "cork oak", "낙엽송": "Japanese larch", "잣나무": "Korean pine", "소나무": "red pine",
    "자작나무": "birch", "백합나무": "tulip tree", "물푸레나무": "ash", "벚나무": "cherry", "포플러": "poplar", "호두나무": "walnut",
    "전나무": "fir", "편백나무": "hinoki cypress", "밤나무": "chestnut",
    "산림": "forest", "나지": "bare", "초지": "grass", "시설지": "non-forest use",
    # 사업 종별
    "조림예정지정리": "site preparation", "어린나무가꾸기": "young-stand tending", "무육솎아베기": "tending thinning",
    "수익솎아베기": "commercial thinning", "숲가꾸기 산물": "tending by-products", "기타 산물": "other products", "덩굴류제거": "vine removal",
    "계류보전복원": "stream restoration", "사방댐유지관리": "check-dam maintenance", "산불진화임도": "firefighting forest road",
    "인공갱신": "artificial regeneration", "임도신설": "new forest road", "임도개량": "forest road improvement", "산지사방": "hillside erosion control",
    "간선임도": "trunk forest road", "작업임도": "operational forest road", "종자채취": "seed collection", "버섯채취": "mushroom harvest",
    "수액채취": "sap collection", "기타시설": "other facilities", "사방댐": "check dam", "풀베기": "weeding", "주벌": "final felling",
    "숲가꾸기": "forest tending",
    # 산지전용 용도
    "건축물대장 양성화": "building-register legalisation", "제2종근린생활시설": "type-2 neighbourhood facility", "도로확포장공사": "road widening",
    "하천환경정비사업": "river environment project", "애견체험박물관": "pet museum", "공공사격장": "public shooting range",
    "목재산업단지": "timber industrial complex", "사무소 부속창고": "office warehouse", "종교집회장": "religious assembly hall",
    "농업용창고": "farm warehouse", "농업용 창고": "farm warehouse", "가족자연장지": "family natural burial site",
    "종중자연장지": "clan natural burial site", "문중자연장지": "clan natural burial site", "자연장지": "natural burial site",
    "가족장지": "family burial site", "가족묘지": "family cemetery", "축사 증축": "livestock barn extension", "축산시설": "livestock facility",
    "축사시설": "livestock barn", "축사": "livestock barn", "농가주택": "farmhouse", "단독주택": "detached house", "관광농원": "tourist farm",
    "종교시설": "religious facility", "산업단지": "industrial complex", "진출입로": "access road", "진입도로": "access road",
    "진입로": "access road", "소매점": "retail store", "야영장": "campsite", "스키장": "ski resort", "주차장": "parking lot",
    "묘지": "cemetery", "주택": "house", "도로": "road", "지방도": "provincial road", "호선": "", "협의": "consultation", "허가": "permit",
    "신고": "report", " 및 ": " and ",
    # 남는 조사·단위
    "년": "", "개": "",
}
_GL = sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0]))


def _protect_places(t: str):
    """읍·면·동·리 지명과 주소는 그대로 두기 위해 잠시 표시로 바꾼다."""
    keep = []

    def sub(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"
    t = re.sub(r"강원특별자치도 춘천시[^|\n]*?(?=\s*\||$)", sub, t, flags=re.M)     # 주소
    t = re.sub(r"[가-힣]{1,4}(?:읍|면|동|리)(?![가-힣])", lambda m: m.group(0) if m.group(0) in ("동면",) else m.group(0), t)
    return t, keep


def _place_ok(w: str) -> bool:
    return bool(re.fullmatch(r"[가-힣]{1,4}(?:읍|면|동|리)", w))


def to_en(text):
    """한글 문장/값 → 영문. 문자열이 아니면 그대로."""
    if not isinstance(text, str) or not HANGUL.search(text):
        return text
    t, keep = _protect_places(text)
    for p, r in TEMPLATES:
        t = p.sub(r, t)
    for k, v in _GL:
        if k in t:
            t = _replace_word(t, k, v)
    t = re.sub(r"([a-z])\(", r"\1 (", t)                                        # final map(AE+S2) → final map (AE+S2)
    t = re.sub(r"^(#+ |- |> )?([a-z])", lambda m: (m.group(1) or "") + m.group(2).upper(), t)   # 문장·제목 첫 글자
    t = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], t)
    return re.sub(r"  +", " ", t)


def _replace_word(t: str, k: str, v: str) -> str:
    """지명(…읍·면·동·리) 일부는 바꾸지 않는다 (예: '산불' 이 아닌 지명 안의 글자)."""
    out, i = [], 0
    for m in re.finditer(re.escape(k), t):
        a, b = m.span()
        j, e = a, b
        while j > 0 and "가" <= t[j - 1] <= "힣":
            j -= 1
        while e < len(t) and "가" <= t[e] <= "힣":
            e += 1
        if _place_ok(t[j:e]) and len(k) < e - j:
            continue
        out.append(t[i:a] + v)
        i = b
    return "".join(out) + t[i:]


def translate_report(md: str) -> tuple[str, list]:
    """보고서 Markdown 번역. (영문, 번역에서 빠진 원문 수치 목록 — 정상이면 빈 목록)"""
    en = "\n".join(to_en(line) for line in md.split("\n"))
    src, dst = Counter(n.replace(",", "") for n in NUM.findall(md)), Counter(n.replace(",", "") for n in NUM.findall(en))
    return en, sorted((src - dst).elements())   # 원문 수치가 빠지거나 바뀐 경우만 (라벨에 붙는 'pre-2019' 같은 추가는 제외)


def translate_fields(d: dict) -> dict:
    return {k: to_en(v) for k, v in d.items()}


def leftovers(text: str) -> Counter:
    """번역 후 남은 한글 낱말 (지명 제외) — 용어집 보완용."""
    return Counter(w for w in HANGUL.findall(text) if not _place_ok(w))
