"""산림변화 의사결정 지원 웹 (FastAPI + Leaflet).

  python -m forest_decision_agent.web            → http://127.0.0.1:8800
기능: 질의 실행(Claude 또는 규칙 기반), 지도(소반 판정·산지전용 필지·재해·읍면동), 속성 조회, 장소 검색,
      지역(읍면동·지도 범위·선택 소반) 요약·질의, 결과 보고서 저장(md·html·근거기록).
"""
from __future__ import annotations

import base64
import html
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from . import agent as A
from . import tools as T
from . import translate as TR

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="산림변화 의사결정 지원")
_GJ: dict[str, str] = {}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _ro():
    return sqlite3.connect(f"file:{T.DB}?mode=ro", uri=True)


def geojson(name: str) -> str:
    if name not in _GJ and name == "모델변화패치":   # 최종지도(AE+S2) 변화 패치
        import geopandas as gpd
        g = gpd.read_file(T.FM_DIR / "final_change_patches.gpkg")
        g["geometry"] = g.simplify(3)
        _GJ[name] = g[["patch_id", "class_ko", "area_ha", "mean_prob", "change_year", "geometry"]].round(3).to_crs(4326).to_json(drop_id=True)
    if name not in _GJ:
        g = T._layer(name).copy()
        keep = {"소반": ["단위ID", "읍면동", "판정", "신호일치수", "FM_주변화유형", "FM_변화면적_ha", "실적구분_2020_2025"],
                "산지전용": ["레코드ID", "라벨", "주소", "고시번호"],
                "재해": ["유형", "연도", "단위ID"],
                "읍면동경계": ["EMD_NM"]}[name]
        g["fid"] = range(len(g))
        g = g[keep + ["fid", "geometry"]]
        if name in ("소반", "읍면동경계"):
            g["geometry"] = g.simplify(2)
        _GJ[name] = g.to_crs(4326).to_json(drop_id=True)
    return _GJ[name]


def bbox4326(geoms) -> list:
    import geopandas as gpd
    b = gpd.GeoSeries(geoms, crs=5186).to_crs(4326).total_bounds
    return [round(float(v), 6) for v in b]


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    m = json.loads((T.DB_DIR / "build_meta.json").read_text(encoding="utf-8"))
    emd = sorted(T._layer("읍면동경계")["EMD_NM"])
    return {"llm": bool(A.api_key()), "model": A.MODEL,
            "models": {"최종지도(AE+S2)": m.get("모델", {}).get("최종지도", {}).get("macro_f1"),
                       "S2-TSFM": m.get("모델", {}).get("S2-TSFM", {}).get("macro_f1")}, "emd": emd, "판정": list(m["판정_분포"]), "판정_분포": m["판정_분포"],
            "임계값": m["임계값"], "기준": m.get("변화관측_기준"), "출처": T.SOURCES, "한계": T.LIMITS}


@app.get("/api/layer/{name}")
def layer(name: str):
    if name not in ("소반", "산지전용", "재해", "읍면동경계", "모델변화패치"):
        raise HTTPException(404)
    return Response(geojson(name), media_type="application/geo+json")


@app.get("/api/feature/{name}/{fid}")
def feature(name: str, fid: int):
    g = T._layer(name)
    if not 0 <= fid < len(g):
        raise HTTPException(404)
    row = g.iloc[fid]
    attrs = {k: (None if pd.isna(v) else v) for k, v in row.drop("geometry").items()}
    related = {}
    with _ro() as con:
        if name == "소반":
            uid = row["단위ID"]
            related["사업실적_2019_2025"] = T._records(pd.read_sql('select "연도","사업구분","종별","면적_ha","사업비_천원" from "사업실적" '
                                                                 'where "단위ID"=? order by "연도"', con, params=[uid]))
            related["가까운_재해"] = T._records(pd.read_sql('select "유형","연도","소반거리_m" from "재해" where "단위ID"=? order by "연도"',
                                                       con, params=[uid]))
            try:
                ae = pd.read_sql('select * from "소반_AE분류" where "단위ID"=?', con, params=[uid])
                if len(ae):
                    r0 = ae.iloc[0]
                    related["AlphaEarth_분류모델"] = [{"구분": "피복 2019", **{l: r0[f"LC_2019_{l}_비율"] for l in T.AE_LC}},
                                                    {"구분": "피복 2025", **{l: r0[f"LC_2025_{l}_비율"] for l in T.AE_LC}},
                                                    {"구분": "임상 2019", **{l: r0[f"FT_2019_{l}_비율"] for l in T.AE_FT}},
                                                    {"구분": "임상 2025", **{l: r0[f"FT_2025_{l}_비율"] for l in T.AE_FT}},
                                                    {"구분": "변화유형 ha", **{l: r0[f"AE_{l}_ha"] for l in T.AE_CH}}]
            except Exception:
                pass
            try:
                hh = pd.read_sql('select "조사연도","마지막조사_최근","변화구간","재조사_우선","AE연도별_우세임상" from "소반_임상도이력" where "단위ID"=?', con, params=[uid])
                if len(hh):
                    related["임상도_조사이력"] = T._records(hh)
                yy = pd.read_sql('select "연도","침엽수림_비율","활엽수림_비율","혼효림_비율","무립목지_비율","우세" from "소반_연도별임상" where "단위ID"=? order by "연도"', con, params=[uid])
                if len(yy):
                    related["AE_연도별_임상추정(2017~2025)"] = T._records(yy)
            except Exception:
                pass
            pc = T._layer("산지전용")
            related["주변_산지전용_필지(1 km)"] = T._records(pc[pc.distance(row.geometry) <= 1000][["레코드ID", "라벨", "주소", "전용용도", "AE_변화비율"]])
    return {"layer": name, "fid": fid, "attributes": json.loads(json.dumps(attrs, default=str, ensure_ascii=False)),
            "related": related, "bbox": bbox4326([row.geometry])}


@app.get("/api/timeseries/{name}/{fid}")
def timeseries(name: str, fid: int):
    g = T._layer(name) if name in ("소반", "산지전용") else None
    if g is None or not 0 <= fid < len(g):
        raise HTTPException(404)
    key = {"unit_id": g.iloc[fid]["단위ID"]} if name == "소반" else {"record_id": g.iloc[fid]["레코드ID"]}
    return T.unit_timeseries(T.Session(query="시계열", run_dir=T.RUNS / "_scratch"), **key)["data"]


@app.get("/api/explain/{fid}")
def explain(fid: int, lang: str = "ko"):
    g = T._layer("소반")
    if not 0 <= fid < len(g):
        raise HTTPException(404)
    out = T.explain_change(T.Session(query="해석", run_dir=T.RUNS / "_scratch"), unit_id=g.iloc[fid]["단위ID"])
    if lang == "en":   # 영문 화면: 근거 문장·경로 이름만 번역(수치·연간추이는 그대로)
        out["data"]["근거"] = TR.translate_fields(out["data"]["근거"])
        out["data"]["변화경로"] = TR.to_en(out["data"]["변화경로"])
    return out


@app.get("/api/similar/{fid}")
def similar(fid: int):
    g = T._layer("소반")
    if not 0 <= fid < len(g):
        raise HTTPException(404)
    return T.find_similar_units(T.Session(query="유사", run_dir=T.RUNS / "_scratch"), reference_ids=[g.iloc[fid]["단위ID"]], top_n=8)


@app.get("/api/search")
def search(q: str):
    q = q.strip()
    if len(q) < 1:
        return []
    out = []
    b = T._layer("읍면동경계")
    for i, r in b[b["EMD_NM"].str.contains(q, regex=False)].iterrows():
        out.append({"type": "읍면동", "label": r["EMD_NM"], "emd": r["EMD_NM"], "bbox": bbox4326([r.geometry])})
    su = T._layer("소반")
    hit = su[su["단위ID"].str.contains(q, regex=False) | su["소재지"].fillna("").str.contains(q, regex=False)]
    for fid, r in hit.head(40).iterrows():
        out.append({"type": "소반", "layer": "소반", "fid": int(fid), "label": f"{r['단위ID']} · {r['소재지']}", "sub": r["판정"],
                    "bbox": bbox4326([r.geometry])})
    pc = T._layer("산지전용")
    hit = pc[pc["주소"].fillna("").str.contains(q, regex=False) | pc["레코드ID"].str.contains(q, regex=False)
             | pc["고시번호"].fillna("").str.contains(q, regex=False)]
    for fid, r in hit.head(40).iterrows():
        out.append({"type": "산지전용", "layer": "산지전용", "fid": int(fid), "label": f"{r['주소']}", "sub": f"{r['고시번호']} · {r['라벨']}",
                    "bbox": bbox4326([r.geometry])})
    return out[:80]


class Area(BaseModel):
    emd: str | None = None
    unit_ids: list[str] | None = None
    bbox: list[float] | None = None

    def clean(self):
        return {k: v for k, v in self.model_dump().items() if v}


@app.post("/api/area_summary")
def area_summary(a: Area):
    s = T.Session(query="지역 요약", run_dir=Path(T.RUNS / "_scratch"))
    try:
        d = T.call(s, "area_summary", a.clean())
    except ValueError as e:
        raise HTTPException(400, str(e))
    rk = T.call(s, "rank_inspection", {"top_n": 10, "unit_ids": T.resolve_area(**a.clean())[0]}) if a.clean() else T.call(s, "rank_inspection", {"top_n": 10})
    return {"summary": d["data"], "ranking": rk["data"]["ranking"], "warnings": d["meta"]["warnings"]}


class Query(BaseModel):
    query: str
    area: Area | None = None
    mode: str = "auto"   # auto | llm | rule
    lang: str = "ko"     # ko | en — en 이면 Claude 는 영문으로 작성, 규칙 기반 보고서는 작성 후 규칙 번역(translate.py)


def _job(jid: str, q: Query):
    job = JOBS[jid]
    log = lambda m: job["logs"].append(f"{time.strftime('%H:%M:%S')} {m}")
    area = q.area.clean() if q.area else None
    use_llm = q.mode == "llm" or (q.mode == "auto" and A.api_key())
    s = T.Session(query=q.query)
    job["run"] = s.run_dir.name
    try:
        with LOCK:
            if use_llm:
                if not A.api_key():
                    raise RuntimeError("API 키가 없습니다 (ANTHROPIC_API_KEY 또는 claude_key.txt).")
                log(f"Claude 에이전트 실행 ({A.MODEL})")
                qq = q.query + ("\n\n[Language] Write all answers and the final report (write_report) in English. "
                                "Keep stand IDs and Korean place names as they are." if q.lang == "en" else "")
                try:
                    A.run_claude(qq, area=area, log=log, session=s)
                except Exception as e:   # 자동 모드: API 오류(잔액 부족 등)면 규칙 기반으로 계속
                    if q.mode != "auto":
                        raise
                    log(f"Claude 호출 실패 ({type(e).__name__}) → 규칙 기반으로 전환")
                    use_llm = False
            if not use_llm:
                log("규칙 기반 분석 실행 (LLM 미사용)")
                A.run_offline(q.query, area=area, log=log, session=s)
                rp = s.run_dir / "report.md"
                if q.lang == "en" and rp.exists():   # 한글 원본은 report_ko.md 로 보존, report.md 는 영문
                    en, miss = TR.translate_report(rp.read_text(encoding="utf-8"))
                    rp.replace(s.run_dir / "report_ko.md")
                    hp = s.run_dir / "highlight.json"   # 지도 결과 목록 제목
                    if hp.exists():
                        h = json.loads(hp.read_text(encoding="utf-8"))
                        for g in h.get("groups", []):
                            g["title"] = TR.to_en(g.get("title"))
                        hp.write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
                    rp.write_text(en + (f"\n> Translation check: numbers missing from the English text: {miss}\n" if miss else ""), encoding="utf-8")
                    log("Translated to English" + (f" — numbers missing: {miss}" if miss else " (all numbers kept)"))
        (s.run_dir / "query.json").write_text(json.dumps({"query": q.query, "area": area, "mode": "llm" if use_llm else "rule", "lang": q.lang,
                                                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False), encoding="utf-8")
        job["status"] = "done" if (s.run_dir / "report.md").exists() else "no_report"
        if job["status"] == "no_report":
            log("보고서가 저장되지 않았습니다 (에이전트가 write_report 를 호출하지 않음).")
    except Exception as e:
        log(f"오류: {type(e).__name__}: {e}")
        job["status"] = "error"


@app.post("/api/query")
def query(q: Query):
    if not q.query.strip():
        raise HTTPException(400, "질의가 비어 있습니다.")
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"status": "running", "logs": [], "run": None}
    threading.Thread(target=_job, args=(jid, q), daemon=True).start()
    return {"job": jid}


@app.get("/api/job/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    return JOBS[jid]


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(T.RUNS.glob("2*"), reverse=True):
        if not (d / "report.md").exists():
            continue
        first = (d / "report.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        qj = json.loads((d / "query.json").read_text(encoding="utf-8")) if (d / "query.json").exists() else {}
        out.append({"run": d.name, "title": first, "query": qj.get("query"), "mode": qj.get("mode"), "area": qj.get("area"),
                    "files": sorted(p.name for p in d.iterdir())})
    return out


def _run_dir(run: str) -> Path:
    if not re.fullmatch(r"[\w\-]+", run):
        raise HTTPException(400)
    d = T.RUNS / run
    if not d.is_dir():
        raise HTTPException(404)
    return d


@app.get("/api/runs/{run}/file/{name}")
def run_file(run: str, name: str):
    p = _run_dir(run) / name
    if not re.fullmatch(r"[\w\-.]+", name) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=name if name.endswith((".md", ".json")) else None)


# ---- 보고서 html (외부 라이브러리 없이 이 프로젝트 보고서 형식만 변환) ----
def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def md_to_html(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table>")
            continue
        if ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{_inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("> "):
            out.append(f"<blockquote>{_inline(ln[2:])}</blockquote>")
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    return "\n".join(out)


REPORT_CSS = """body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1f2a1f;line-height:1.55}
h1{font-size:22px;border-bottom:2px solid #2f6b3a;padding-bottom:6px}h2{font-size:17px;margin-top:22px;color:#2f6b3a}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #cfd8cf;padding:4px 6px;text-align:left}
th{background:#eef4ee}blockquote{margin:8px 0;padding:6px 10px;background:#f5f7f5;border-left:3px solid #2f6b3a}
code{background:#f0f0f0;padding:0 3px;font-size:12px}img{max-width:100%;border:1px solid #ddd;margin:8px 0}hr{margin:20px 0}
@media print{body{margin:0}h2{break-after:avoid}table{break-inside:avoid}}"""


@app.get("/api/runs/{run}/report.html", response_class=HTMLResponse)
def report_html(run: str, download: bool = False):
    d = _run_dir(run)
    md = (d / "report.md").read_text(encoding="utf-8")
    body = md_to_html(md)
    qj = json.loads((d / "query.json").read_text(encoding="utf-8")) if (d / "query.json").exists() else {}
    en = (d / "report_ko.md").exists() or qj.get("lang") == "en"
    imgs = "".join(f'<h2>{"Map" if en else "지도"} — {p.stem}</h2><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'
                   for p in sorted(d.glob("*.png")))
    body = body.replace("<hr>", imgs + "<hr>", 1) if "<hr>" in body else body + imgs
    title = html.escape(md.splitlines()[0].lstrip("# "))
    doc = f'<!doctype html><html lang="{'en' if en else 'ko'}"><head><meta charset="utf-8"><title>{title}</title><style>{REPORT_CSS}</style></head><body>{body}</body></html>'
    headers = {"Content-Disposition": f"attachment; filename=report_{run}.html"} if download else {}
    return HTMLResponse(doc, headers=headers)


def main():
    import uvicorn
    print("산림변화 의사결정 지원 웹: http://127.0.0.1:8800  (LLM:", "연결" if A.api_key() else "미연결 → 규칙 기반", ")")
    for n in ("소반", "산지전용", "재해", "읍면동경계", "모델변화패치"):
        geojson(n)
    uvicorn.run(app, host="127.0.0.1", port=8800, log_level="warning")


if __name__ == "__main__":
    main()
