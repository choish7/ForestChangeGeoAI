"""춘천시 보전산지 해제 고시(타용도 전용) → 산지전용 데이터셋.

입력 (고시별 폴더, 춘천시 누리집 고시공고 게시판 첨부파일)
  춘천시 산지전용 고시/
    보전산지해제 세부내역(7_1_3 유형).xls   필지별 변경유형·소재지·면적·전용용도·도엽
    산지구분도.pdf                         전용용도 묶음별 도면 1쪽씩 (세부내역의 용도 묶음 순서와 같음)
    보전산지 해제 고시(...).png             고시문 (고시번호·고시일)
  NOTICES 에 고시를 추가하면 여러 고시를 한 데이터셋으로 모은다.

처리
  1) 세부내역 표 파싱: 지번 "산1-4도" → 산구분·본번·부번·지목, 전용용도·도엽은 묶음 첫 행 값을 아래로 채움
  2) PNU(19자리) = 법정동코드(10, 강원특별자치도 51110…) + 산구분(1 일반, 2 산) + 본번(4) + 부번(4)
     법정동코드: 행정안전부 법정동코드(PublicDataReader code_bdong, 2023-05 기준 42110…) → 2023-06-11 강원특별자치도
     출범으로 앞자리 42110 → 51110 (뒤 5자리 동일)
  3) 산지구분도 PDF 쪽 ↔ 전용용도 묶음 대응, 도면 PNG 추출
  4) (선택) 연속지적도(LSMD_CONT_LDREG_*.shp, 국가공간정보포털)가 프로젝트에 있으면 PNU 로 결합해 필지 폴리곤 생성,
     지적 면적 대조, 임상도 2019·2025 속성(전용 전·후 임상) 부여

출력: output/산지전용/
  춘천시_산지전용_보전산지해제.csv / .gpkg (지적도 있으면 폴리곤 레이어 '산지전용_필지', 없으면 속성 표)
  maps/<고시번호>_p<쪽>.png, 춘천시_법정동코드.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 임상도 코드표(combine_thinning_forest_type) 재사용
OUT = ROOT / "output" / "산지전용"
BDONG = OUT / "춘천시_법정동코드.csv"
NOTICE_DIR = ROOT / "춘천시 산지전용 고시"
BOARD = "https://www.chuncheon.go.kr/cityhall/administrative-info/notice-info/notice-announcement/"
# 고시별로 아는 정보 (게시판 목록이 동적 로딩이라 고시일은 확인된 것만). 키 = "YYYY-번호"
NOTICE_META = {
    "2024-649": {"고시일": "2024-12-13", "제목": "보전산지 해제 고시(타용도 전용)", "담당": "민원담당관"},
}
LAW = "산지관리법 제6조·같은 법 시행령 제5조, 토지이용규제 기본법 제8조"
NUM = re.compile(r"제\s*(\d{4})\s*-\s*(\d+)\s*호")


def discover_notices():
    """폴더의 세부내역 xls 마다 고시 1건. 고시번호는 파일명 → 없으면 같은 폴더 고시문 png 파일명에서.
    산지구분도 pdf 는 파일명에 고시번호가 있으면 그 고시, 없으면 번호 없는 xls 의 고시에 붙인다."""
    xlss = sorted(NOTICE_DIR.glob("*.xls"))
    pngs = [p for p in NOTICE_DIR.glob("*.png") if NUM.search(p.name)]
    pdfs = sorted(NOTICE_DIR.glob("*.pdf"))
    notices = []
    for x in xlss:
        m = NUM.search(x.name) or (NUM.search(pngs[0].name) if len(pngs) == 1 and not any(NUM.search(o.name) is None
                                                                                       for o in xlss if o != x) else None)
        key = f"{m.group(1)}-{m.group(2)}" if m else x.stem
        named = [p for p in pdfs if NUM.search(p.name) and f"{NUM.search(p.name).group(1)}-{NUM.search(p.name).group(2)}" == key]
        unnamed = [p for p in pdfs if not NUM.search(p.name)] if not NUM.search(x.name) else []
        meta = NOTICE_META.get(key, {})
        notices.append({"xls": x, "pdf": (named or unnamed or [None])[0], "key": key,
                        "고시번호": f"춘천시 고시 제{key}호" if m else x.stem, "고시일": meta.get("고시일"),
                        "제목": meta.get("제목", "보전산지 해제 고시"), "근거": LAW, "출처": BOARD})
    return notices


def load_bdong():
    """춘천시 법정동코드 (강원특별자치도 51110 체계). 없으면 PublicDataReader 원자료에서 만든다."""
    if BDONG.exists():
        b = pd.read_csv(BDONG, dtype=str, keep_default_na=False)
        if (b["동리명"] == "").any():  # 동 단위 코드까지 든 표만 재사용
            return b
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/WooilJeong/PublicDataReader/main/PublicDataReader/raw/code_bdong.json"
    d = pd.DataFrame(json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8")))
    # 리 단위 + 리가 없는 동(칠전동·송암동 등 시내 동). 읍·면 자체(뒤 5자리 …000)는 제외
    dong = d["동리명"].isna() & d["읍면동명"].str.contains(r"(?:동|가)$", na=False)
    d = d[(d["시군구코드"] == "42110") & d["말소일자"].isna() & (d["동리명"].notna() | dong)]
    d["동리명"] = d["동리명"].fillna("")
    d = d.assign(법정동코드=("51110" + d["법정동코드"].str[5:]), 법정동코드_구=d["법정동코드"])
    d = d[["법정동코드", "법정동코드_구", "읍면동명", "동리명"]]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(BDONG, index=False, encoding="utf-8-sig")
    return d


def parse_detail(xls):
    raw = pd.read_excel(xls, header=None, dtype=str)
    rows = raw[raw[0].str.fullmatch(r"\d+", na=False)]
    df = pd.DataFrame({
        "일련번호": rows[0].astype(int), "변경유형": rows[2].str.strip(),
        "시군구": rows[4], "읍면동": rows[5], "리": rows[6], "지번지목": rows[7].str.strip(),
        "지적면적_m2": pd.to_numeric(rows[8]), "기지정면적_m2": pd.to_numeric(rows[9]),
        "해제면적_m2": pd.to_numeric(rows[10]), "전용용도": rows[11].astype("string"), "도엽": rows[13].str.strip()})
    # 지번지목 표기: "산1-4도", "828-1 전", "산 53 임" → 공백 제거 후 분해
    m = df["지번지목"].str.replace(r"\s+", "", regex=True).str.extract(r"^(산)?(\d+)(?:-(\d+))?(\D+)$")
    df["산구분"] = m[0].map({"산": "2"}).fillna("1")
    df["본번"] = m[1].astype(int)
    df["부번"] = m[2].fillna("0").astype(int)
    df["지목"] = m[3]
    df["지번"] = m[0].fillna("") + m[1] + m[2].map(lambda v: f"-{v}" if isinstance(v, str) else "")
    # 전용용도·도엽은 묶음 첫 행에만 적혀 있음 → 묶음 번호 부여 후 아래로 채움
    df["용도묶음"] = df["전용용도"].notna().cumsum()
    df[["변경유형", "전용용도", "도엽"]] = df[["변경유형", "전용용도", "도엽"]].ffill()  # 변경유형도 묶음 첫 행에만 적힌 서식이 있음
    df["허가구분"] = df["전용용도"].str.extract(r"\((허가|협의|신고)\)")[0]
    df["전용용도_정리"] = df["전용용도"].str.replace(r"\((허가|협의|신고)\)", "", regex=True).str.strip()
    df["해제구분"] = (df["해제면적_m2"] < df["지적면적_m2"] - 0.5).map({True: "일부", False: "전체"})
    # 고시 합계: '합 계' 행, 없으면 '소계' 행 (여러 쪽 서식이면 마지막 합계)
    label = raw[0].fillna("").str.replace(r"\s+", "", regex=True)
    tot = raw[label.eq("합계")] if label.eq("합계").any() else raw[label.eq("소계")]
    total = float(pd.to_numeric(tot[10]).iloc[-1]) if len(tot) else float("nan")
    return df, total


def extract_maps(pdf, notice_no, n_groups):
    import fitz
    doc = fitz.open(pdf)
    if len(doc) != n_groups:
        print(f"  [경고] 산지구분도 {len(doc)}쪽 ≠ 전용용도 묶음 {n_groups}개 — 도면 대응을 확인하세요")
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"\D", "", notice_no)
    pages = {}
    for i, page in enumerate(doc, 1):
        out = OUT / "maps" / f"{tag}_p{i}.png"
        page.get_pixmap(dpi=110).save(out)
        text = page.get_text()  # PDF 텍스트는 띄어쓰기가 빠져 있음 ("강원특별자치도춘천시동산면원창리149-16번지")
        title = " ".join(re.findall(r"강원특별자치도\s*춘천시[^\n]*번지[^\n]*", text))
        sheet = re.search(r"\(\s*(NJ[^)]+?)\s*\)", text)
        pages[i] = {"도면파일": f"maps/{out.name}", "도면제목": title, "도면_도엽": sheet.group(1) if sheet else None}
    return pages


def attach_cadastre(df):
    """연속지적도가 있으면 PNU 로 필지 폴리곤 결합 + 임상도 속성."""
    shp = sorted(ROOT.glob("**/LSMD_CONT_LDREG*.shp"))
    if not shp:
        return None
    import geopandas as gpd
    parts = []
    for p in shp:
        pnus = sorted(df["PNU"].dropna().unique())
        g = gpd.read_file(p, encoding="cp949", where=f"PNU IN ({','.join(repr(x) for x in pnus)})")
        parts.append(g)
    cad = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(df.merge(cad[["PNU", "geometry"]], on="PNU", how="left"), geometry="geometry", crs=cad.crs)
    gdf["도형면적_m2"] = gdf.area.round(1)
    gdf["지적면적_대조"] = (gdf["도형면적_m2"] / gdf["지적면적_m2"]).round(3)
    # 전용 전·후 임상: 필지 면적 중 임상도 임지(입목지·무립목지) 비율과 면적 최대 임분의 임상·수종
    #   임상도는 산림만 도화하므로, 2019 에 임지였다가 2025 에 빠졌으면 산지전용이 임상도에 반영된 것
    from combine_thinning_forest_type import FRTP, KOFTR
    parcels = gdf.to_crs(5179)
    parcels["_pid"] = range(len(parcels))
    for y in (2019, 2025):
        fm = ROOT / f"임상도 {y} 51110" / "51110.shp"
        if not fm.exists():
            continue
        f = gpd.read_file(fm, encoding="cp949", columns=["FRTP_CD", "KOFTR_GROU", "STORUNST"],
                          bbox=tuple(parcels.total_bounds))
        f = f[f["STORUNST"].astype(str).isin(["1", "2"])]
        ov = gpd.overlay(parcels[["_pid", "geometry"]], f, how="intersection", keep_geom_type=True)
        ov["a"] = ov.area
        share = ov.groupby("_pid")["a"].sum() / parcels.set_index("_pid").area
        top = ov.sort_values("a", ascending=False).drop_duplicates("_pid").set_index("_pid")
        gdf[f"임지비율_{y}"] = parcels["_pid"].map(share).fillna(0).clip(upper=1).round(3).values
        gdf[f"임상_{y}"] = parcels["_pid"].map(top["FRTP_CD"].astype(str).map(FRTP)).values
        gdf[f"수종_{y}"] = parcels["_pid"].map(top["KOFTR_GROU"].astype(str).map(KOFTR)).values
    if {"임지비율_2019", "임지비율_2025"} <= set(gdf.columns):
        drop = gdf["임지비율_2019"] - gdf["임지비율_2025"]
        gdf["임상도_전용반영"] = (drop >= 0.5).map({True: "Y(임지→비임지)", False: "N"}).where(gdf["임지비율_2019"] >= 0.5, "2019 비임지")
    return gdf


def main():
    bd = load_bdong()
    code = {(r.읍면동명, r.동리명): r.법정동코드 for r in bd.itertuples()}
    frames = []
    for n in discover_notices():
        df, total = parse_detail(n["xls"])
        if n["pdf"] is not None:
            pages = extract_maps(n["pdf"], n["고시번호"], df["용도묶음"].nunique())
            df = df.join(df["용도묶음"].map(pages).apply(pd.Series))
        for k in ("고시번호", "고시일", "제목", "근거", "출처"):
            df[k] = n[k]
        df["원본파일"] = n["xls"].name
        df["유형"] = "7-1-3 (산지관리법 시행령 제7조제1항제3호: 산지전용허가·협의·신고로 다른 용도로 변경된 보전산지)"
        df["법정동코드"] = [code.get((a, b if isinstance(b, str) else "")) for a, b in zip(df["읍면동"], df["리"])]
        df["리"] = df["리"].fillna("")
        if df["법정동코드"].isna().any():
            print("  [경고] 법정동코드 없음:", df[df["법정동코드"].isna()][["읍면동", "리"]].drop_duplicates().values.tolist())
        df["PNU"] = df["법정동코드"] + df["산구분"] + df["본번"].map("{:04d}".format) + df["부번"].map("{:04d}".format)
        df["주소"] = "강원특별자치도 " + df["시군구"] + " " + df["읍면동"] + " " + df["리"] + " " + df["지번"]
        check = abs(df["해제면적_m2"].sum() - total) < 0.1
        print(f"[{n['고시번호']}] 필지 {len(df)}개, 해제면적 합 {df['해제면적_m2'].sum():,.2f} m² "
              f"(고시 합계 {total:,.2f} m² {'일치' if check else '불일치'}), 산지구분도 {'있음' if n['pdf'] else '없음'}")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).sort_values(["고시번호", "일련번호"]).reset_index(drop=True)
    for c in ("도면파일", "도면제목", "도면_도엽"):
        if c not in data:
            data[c] = None
    data.insert(0, "레코드ID", data["고시번호"].str.extract(r"(\d{4}-\d+)")[0].fillna("X") + "_" + data["일련번호"].map("{:03d}".format))
    cols = ["레코드ID", "고시번호", "고시일", "일련번호", "PNU", "주소", "읍면동", "리", "지번", "산구분", "지목", "변경유형", "전용용도",
            "전용용도_정리", "허가구분", "지적면적_m2", "기지정면적_m2", "해제면적_m2", "해제구분", "용도묶음", "도엽",
            "도면파일", "도면제목", "도면_도엽", "법정동코드", "유형", "근거", "제목", "출처", "원본파일"]
    data = data[cols]
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "춘천시_산지전용_보전산지해제.csv", index=False, encoding="utf-8-sig")

    gdf = attach_cadastre(data)
    gpkg = OUT / "춘천시_산지전용_보전산지해제.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    import geopandas as gpd
    if gdf is not None:
        gdf.to_file(gpkg, layer="산지전용_필지", driver="GPKG")
        print(f"[지적도] 결합 {gdf.geometry.notna().sum()}/{len(gdf)} 필지 → 폴리곤 레이어")
    else:
        gpd.GeoDataFrame(data).to_file(gpkg, layer="산지전용_필지_속성", driver="GPKG")
        print("[지적도] LSMD_CONT_LDREG_*.shp 없음 → PNU 속성 표만 저장 (지적도를 넣고 다시 실행하면 폴리곤 생성)")

    pd.set_option("display.width", 220)
    summ = data.groupby(["고시번호", "변경유형"], dropna=False).agg(필지=("PNU", "size"), 해제면적_m2=("해제면적_m2", "sum"))
    print(summ.round(1).to_string())
    if gdf is not None:
        miss = gdf[gdf.geometry.isna()]
        print(f"[지적도 미결합] {len(miss)}건 (분할·합병·지번변경 추정):", miss["레코드ID"].tolist()[:20])
        if "임상도_전용반영" in gdf:
            print(gdf.groupby(["고시번호", "임상도_전용반영"]).size().unstack(fill_value=0).to_string())
    print(f"저장: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
