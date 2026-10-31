# CarbonAce Mini-GFM — PyTorch 사전학습 → 추론

GEE에서 내보낸 **동일 지역 3개 연도 GeoTIFF**를 PyTorch에 입력하여 **마스크 복원 자기지도학습**을 수행하고, 연도별 인코더 특징 변화(거리)를 GeoTIFF로 출력합니다. 작은 CNN 기반의 **GFM 후보/프로토타입**이며, 검증된 파운데이션 모델이나 정답 기반 변화탐지 분류기는 아닙니다.

## 1. 입력 형식

- **18밴드 (기본)**: 2023 [NDVI, NDWI, NDBI, NDMI, Slope, Elevation], 2024 동일 순서, 2025 동일 순서
- **14밴드**: 3개 연도 각각 [NDVI, NDWI, NDBI, NDMI] 총 12개 + Slope + Elevation
- **11밴드**: 3개 연도 각각 [NDVI, NDWI, NDMI] 총 9개 + Slope + Elevation
- **15밴드 (초기 구성)**: 3개 연도 각각 [NDVI, NDWI, NDBI, Slope, Elevation]. 코드에서 **NDMI = -NDBI** 로 변환합니다.
- 각 연도 같은 픽셀 격자·좌표계를 사용하고, 연도별 동일한 계절을 사용해야 합니다. 입력 영상이 복수라면 영상들끼리 픽셀 그리드가 달라도 **각 영상 내부에서 3개 연도 정렬**되어 있으면 됩니다.
- Sentinel-2 GEE 예시 기준 20m, 입력 수치는 NDVI/NDWI/NDMI = -1..1, Slope = degree, Elevation = meter입니다.
- NDBI = (B11 - B8) / (B11 + B8) = -NDMI이므로 정보 중복을 줄여 실제 네트워크에서는 11채널만 사용합니다.
- 구름/NoData는 학습 및 추론에서 무효 처리됩니다. 추론 결과에서도 NoData=-9999를 유지합니다.

## 2. 설치 (Windows PowerShell 예시)

Python 3.11 권장 (검증 환경에서는 Python 3.13도 실행됨). PyTorch GPU(CUDA) 설치는 공식 페이지 https://pytorch.org/get-started/locally/ 에서 장비에 맞게 설치하세요.

```powershell
cd CarbonAce_MiniGFM
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CPU로도 구동 가능하지만 GPU가 권장됩니다. PowerShell 실행 정책 때문에 가상환경 활성화가 안 되면 `.venv\Scripts\python.exe`를 직접 사용하세요.

## 3. 빠른 전체 실행 (합성 입력)

```powershell
python examples/make_demo_data.py
python src/mini_gfm.py train --inputs examples/demo_18bands.tif --outdir runs/demo --epochs 2 --steps 4 --batch 2 --patch 128 --device cpu
python src/mini_gfm.py infer --input examples/demo_18bands.tif --checkpoint runs/demo/best_model.pt --outdir outputs/demo --tile 128 --halo 16 --device cpu --save-embedding
```

`outputs/demo/demo_18bands_change_scores.tif`에 5밴드 변화점수 영상이 저장됩니다. ZIP에는 별도 합성 예제의 체크포인트 및 결과 영상도 포함되어 있습니다. **합성 데이터는 실행 확인용일 뿐 실제 모델 성능 검증이 아닙니다.**

## 4. GEE 실데이터 학습

동봉된 `gee_export_18bands.js`를 GEE Code Editor에서 실행·내보낸 후, 다운로드 원본 파일을 예를 들어 `data/GeoAI_2023_2025_18Bands.tif`로 복사합니다. 20m 영상 하나에 3개 연도가 정렬되어 있어야 합니다.

```powershell
python src/mini_gfm.py train --inputs "data/*.tif" --outdir runs/forest --years 2023 2024 2025 --epochs 50 --steps 200 --batch 4 --patch 128 --dim 24 --device auto
```

입력이 큰 지역 하나여도 학습할 수 있지만, 다양한 지역 영상 여러 개로 학습해야 다른 지역 일반화 가능성을 평가할 수 있습니다. 학습은 지리적으로 겹치지 않는 패치로 내부 train/val 분할합니다(인접 지역 간 공간 자기상관은 남을 수 있음).

**체크포인트**: `runs/forest/best_model.pt` / `history.json` / `config.json`.

## 5. 추론

```powershell
python src/mini_gfm.py infer --input data/GeoAI_2023_2025_18Bands.tif --checkpoint runs/forest/best_model.pt --outdir outputs/forest --tile 256 --halo 16 --device auto --save-embedding
```

산출물:

- `change_scores.tif`: Float32 5밴드. **밴드1:** 2023→2024 임베딩 거리, **밴드2:** 2024→2025 임베딩 거리, **밴드3:** 2023→2025 임베딩 거리, **밴드4:** NDVI(2025)-NDVI(2023), **밴드5:** NDMI(2025)-NDMI(2023)
- `embedding_3year_mean.tif` (`--save-embedding` 사용 시): 24밴드(기본 `--dim 24`)의 학습된 평균 특징맵

**임베딩 거리는 이상 변화의 비정규화 점수이며 변화 발생 확률, 토지피복 클래스, 탄소량, 통계적 유의성 결과가 아닙니다.** 검증된 정답 라벨/현장 조사 등으로 임계값 선정·오탐률을 산정해야 합니다. NDVI/NDMI 차이 밴드는 원 단위의 정규화지수 변화량입니다.

## 6. 구조

```text
GEE GeoTIFF 18/14/11/15 bands
  -> 11 logical features (3 annual indices x 3 years + slope + elevation)
  -> spatial masked inputs
  -> year-shared CNN encoder (three year embeddings)
  -> temporal feature concatenation and masked image reconstruction
  -> model training: masked-index L1 reconstruction loss
  -> inference: three pairwise embedding distances + NDVI/NDMI differences
  -> georeferenced GeoTIFF
```

모델은 단순한 **소형 시공간 자기지도 표현학습** 시제품입니다. "GFM" 수준의 전이 성능을 입증하려면 여러 지역·센서·태스크에서 외부 검증, 다른 기준모델과 비교, 재현성 검증이 필요합니다.

## 7. 주요 유의사항

1. **고도/경사 정적 자료**: 동일 DEM에서 추출한 값이며 연도별 지형 변화로 해석할 수 없습니다.
2. **NDBI와 NDMI 중복**: 현재 공식으로는 단순 부호 반전입니다. 원래 입력에서는 보존하되 모델에는 하나만 넣습니다.
3. **NDWI 종류**: 현재 NDWI는 McFeeters 방식 `(Green-NIR)/(Green+NIR)`입니다.
4. **계절/구름/센서**: 영상 취득 계절과 잔류 구름·음영이 임베딩 변화를 유발할 수 있습니다.
5. **라벨 부재**: 학습은 자기지도 방식이며 클래스별 변화 유형(신규·소실 등)을 자동으로 판정하지 않습니다.
6. **학습 데이터 한계**: 소지역 단일 입력으로는 같은 지역에 맞춘 표현학습 모델일 뿐 넓은 의미의 파운데이션 모델을 주장하기 어렵습니다.
7. **홀수/작은 범위 영상**: 추론은 타일+경계 halo로 전체 범위를 보존하며 16픽셀 미만 입력은 불가합니다.
8. **좌표계**: 출력은 입력 영상의 CRS, transform, 해상도를 그대로 유지합니다.
9. **검증 손실**: 마스크 복원 L1 손실이며 실제 변화탐지 정밀도/F1과 다릅니다.

## 8. 추가 확장 방향

- 토지피복 분류 라벨 또는 실제 변화 폴리곤으로 downstream head 학습
- **지역 외부 검증** (학습에 쓰지 않은 시·군의 타일)
- Sentinel-2 원본 반사율과 SAR, AlphaEarth와의 다중모달 비교
- 임베딩 거리 점수의 보정 및 운영 임계치 도출

# 9. 추가사항
python run_pipeline.py --inputs "data/gfm_2023_2024_2025_input15*.tif" --labels "data/gfm_2023_2024_2025_labels*.tif" --outdir runs/exp01 --years 2023 2024 2025 --epochs 30
python run_pipeline.py --inputs "data/gfm_2023_2024_2025_input15*.tif" --labels "data/gfm_2023_2024_2025_labels*.tif" --outdir runs/exp01 --checkpoint runs/exp01/model/best_model.pt
## 출력 파일 접두어

추론 결과와 평가 지도에는 기본적으로 입력 파일 이름(확장자 제외)이 붙습니다.
예: scene.tif → scene_change_scores.tif, scene_embedding_3year_mean.tif.

모델별로 구분하려면 infer 또는 run_pipeline.py 명령에 `--output-prefix model_A`를 추가하세요.
결과: model_A_change_scores.tif, model_A_embedding_3year_mean.tif,
model_A_eval_map_CHG_Y1Y3_emb_Y1Y3.tif.
기존 파일은 유지되며 다음 실행부터 새 이름으로 생성됩니다.