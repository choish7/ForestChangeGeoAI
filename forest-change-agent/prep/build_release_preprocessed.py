"""output/ 의 전처리 결과 중 배포 가능한 것만 forest-change-agent/release_preprocessed/ 로 모은다.
경영계획부(비공개) 유래 레이어·열은 제거하고, 주로 비공개 자료에서 나온 파일은 제외한다."""
import csv, hashlib, shutil, warnings
from pathlib import Path
import pandas as pd, pyogrio, rasterio
warnings.filterwarnings('ignore')
R = Path(r'C:\src\ForestChangeGeoAI'); O = R / 'output'; D = R / 'forest-change-agent' / 'release_preprocessed'
D.mkdir(parents=True, exist_ok=True)
for p in D.iterdir():          # README.md(손으로 쓴 설명)는 남기고 나머지만 새로 만든다
    if p.name == 'README.md':
        continue
    shutil.rmtree(p) if p.is_dir() else p.unlink()
rows = []  # (배포경로, 원본, 처리)
PRIV = lambda c: (c.startswith('국유림') and not c.startswith('국유림관리소')) or '실적' in c   # 국유림관리소기관코드는 공개 원자료 열


def copy(src, sub, note='그대로 복사'):
    dst = D / sub / Path(src).name; dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(O / src, dst); rows.append((dst, src, note))


def gpkg(src, sub, layers):
    """layers: {레이어: '열제거'|'그대로'} — 지정하지 않은 레이어는 빠진다."""
    dst = D / sub / Path(src).name; dst.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    for lyr, mode in layers.items():
        g = pyogrio.read_dataframe(O / src, layer=lyr)
        drop = [c for c in g.columns if mode == '열제거' and PRIV(c)]
        pyogrio.write_dataframe(g.drop(columns=drop), dst, layer=lyr, driver='GPKG')
        notes.append(f'{lyr}' + (f'(열 제거: {", ".join(drop)})' if drop else ''))
    allL = [n for n, _ in pyogrio.list_layers(O / src)]
    gone = [n for n in allL if n not in layers and n != 'layer_styles']
    rows.append((dst, src, '레이어 ' + '; '.join(notes) + (f' | 제외 레이어: {", ".join(gone)}' if gone else '')))


def csv_strip(src, sub):
    d = pd.read_csv(O / src, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    drop = [c for c in d.columns if PRIV(c)]
    dst = D / sub / Path(src).name; dst.parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=drop).to_csv(dst, index=False, encoding='utf-8-sig')
    rows.append((dst, src, f'열 제거: {", ".join(drop)}' if drop else '그대로 복사'))


# 1 경계
for f in ['chuncheon_boundary.geojson'] + [f'{b}.{e}' for b in ('춘천시경계', '춘천시_읍면동경계') for e in ('shp', 'shx', 'dbf', 'prj', 'cpg')]:
    copy(f, '01_경계')
# 2 임상도 전이(변화) 라벨
gpkg('춘천시_산림변화라벨_2013_2019_2025.gpkg', '02_임상도_변화라벨',
     {'변화라벨': '열제거', '도화범위차이': '그대로', '구간별_요약': '그대로', '궤적_요약': '그대로'})
copy('산림변화라벨_구간별_요약.csv', '02_임상도_변화라벨'); copy('산림변화라벨_궤적_요약.csv', '02_임상도_변화라벨')
copy('ae_class_samples.npz', '02_임상도_변화라벨')
# 3 산림 마스크·평가 라벨 래스터
copy('Chuncheon_forestmask_AE_10m.tif', '03_래스터_마스크_라벨'); copy('Chuncheon_forestmask_S2_20m.tif', '03_래스터_마스크_라벨')
src = O / 'Chuncheon_S2_2019_2025_labels.tif'; dst = D / '03_래스터_마스크_라벨' / 'Chuncheon_S2_2019_2025_labels_FC.tif'
with rasterio.open(src) as s:
    keep = [i + 1 for i, n in enumerate(s.descriptions) if n in ('FC_TYPE', 'FC_YEAR')]
    prof = s.profile | {'count': len(keep)}
    with rasterio.open(dst, 'w', **prof) as t:
        for j, b in enumerate(keep, 1):
            t.write(s.read(b), j); t.set_band_description(j, s.descriptions[b - 1])
rows.append((dst, 'Chuncheon_S2_2019_2025_labels.tif', '밴드 FC_TYPE·FC_YEAR 만 (MGMT·MGMT_YEAR 제외)'))
# 4 산지전용 (보전산지 해제 고시)
for f in ['산지전용/춘천시_산지전용_보전산지해제.csv', '산지전용/춘천시_법정동코드.csv', '산지전용/산지전용_라벨_테스트.csv',
          '산지전용/산지전용_라벨_테스트.json', '산지전용/Chuncheon_convlabels_AE_10m.tif', '산지전용/Chuncheon_convlabels_S2_20m.tif']:
    copy(f, '04_산지전용')
gpkg('산지전용/춘천시_산지전용_보전산지해제.gpkg', '04_산지전용', {'산지전용_필지': '열제거'})
gpkg('산지전용/산지전용_라벨.gpkg', '04_산지전용', {'산지전용_라벨': '열제거'})
# 5 숲가꾸기 (공개 사업지)
gpkg('춘천시_산림경영활동_숲가꾸기공간정보.gpkg', '05_숲가꾸기', {'숲가꾸기공간정보': '그대로', '춘천시경계': '그대로', '춘천시_읍면동경계': '그대로'})
gpkg('춘천시_숲가꾸기_임상도_사업실적_결합.gpkg', '05_숲가꾸기',
     {'숲가꾸기_결합': '열제거', '숲가꾸기x임상도2019': '그대로', '숲가꾸기x임상도2025': '그대로'})
# 산불·산사태(춘천시_산불_산사태.gpkg, 춘천시_산불.csv, 춘천시_산사태.csv)는 내부 자료라 배포하지 않는다 (2026-11-01 결정)

# MANIFEST
with open(D / 'MANIFEST.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['배포경로', '바이트', 'sha256', '원본(output/)', '처리'])
    for dst, src, note in rows:
        w.writerow([dst.relative_to(D).as_posix(), dst.stat().st_size, hashlib.sha256(dst.read_bytes()).hexdigest(), src, note])
# 점검: 비공개 유래 열·레이어가 남았는지
left = []
for p in D.rglob('*.gpkg'):
    for n, _ in pyogrio.list_layers(p):
        if n in ('임소반_실적요약', '산림사업실적', '미결합실적', '국유림_임소반', '숲가꾸기x임소반', '실적_검증'):
            left.append(f'{p.name}:{n}')
        left += [f'{p.name}:{n}.{c}' for c in pyogrio.read_info(p, layer=n)['fields'] if PRIV(c)]
for p in D.rglob('*.csv'):
    if p.name == 'MANIFEST.csv':
        continue
    left += [f'{p.name}.{c}' for c in pd.read_csv(p, nrows=0, encoding='utf-8-sig').columns if PRIV(c)]
print('files', len(rows), '| private leftovers:', left or 'none')
for dst, src, note in rows:
    print(f'  {dst.relative_to(D).as_posix():70s} {note[:110]}')
