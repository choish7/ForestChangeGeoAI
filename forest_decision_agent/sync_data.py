"""웹서비스 실행에 필요한 자료를 forest_decision_agent/data/ 로 모은다 (이 폴더만으로 서비스가 돌도록).

  python forest_decision_agent/sync_data.py          # 프로젝트 루트에서. 근거 DB 를 다시 만든 뒤에도 실행
  python forest_decision_agent/sync_data.py --check  # 복사 없이 누락만 확인

- 20 MB 를 넘는 파일(위성 래스터)은 NTFS 하드링크로 둔다. 추가 디스크를 쓰지 않고, 폴더를 다른 곳으로 복사하면 실제 내용이 함께 복사된다.
- 작은 파일(근거 DB·메타·작은 래스터)은 복사한다.
- 원본이 바뀌었으면(크기·수정시각 다름) 새로 링크·복사한다.
"""
import argparse
import os
import shutil
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent                       # 프로젝트 루트 (원본 위치)
DATA = PKG / "data"
BIG = 20 * 2**20

# (원본, data/ 안 경로)
FILES = [(f"output/의사결정DB/{n}", f"의사결정DB/{n}") for n in ("forest_decision.sqlite", "forest_decision.gpkg", "build_meta.json")]
FILES += [(f"output/Chuncheon_AE_{y}_64Bands_int8.tif", f"satellite/Chuncheon_AE_{y}_64Bands_int8.tif") for y in range(2017, 2026)]
FILES += [("output/Chuncheon_S2_2017_2025_54Bands.tif", "satellite/Chuncheon_S2_2017_2025_54Bands.tif")]
FILES += [(f"final_change_2019_2025/{n}", f"final_change_2019_2025/{n}")
          for n in ("final_change_class.tif", "final_change_year.tif", "final_change_prob.tif", "final_change_patches.gpkg")]
FILES += [(f"forest-change-agent/models/s2_tsfm_chuncheon/{n}", f"s2_tsfm_chuncheon/{n}") for n in ("change_class.tif", "change_prob.tif")]


def same(a: Path, b: Path) -> bool:
    if not b.exists():
        return False
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    missing, done = [], []
    for src_rel, dst_rel in FILES:
        src, dst = ROOT / src_rel, DATA / dst_rel
        if not src.exists():
            if not dst.exists():
                missing.append(dst_rel)
            continue                                   # 원본이 없으면(이 폴더만 옮긴 경우) data/ 에 있는 것을 그대로 쓴다
        if same(src, dst) or a.check:
            if not dst.exists():
                missing.append(dst_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        if src.stat().st_size > BIG:
            try:
                os.link(src, dst); done.append(f"link {dst_rel}")
                continue
            except OSError:                            # 다른 드라이브 등 하드링크 불가 → 복사
                pass
        shutil.copy2(src, dst); done.append(f"copy {dst_rel}")
    for d in done:
        print(d)
    print(f"[sync] {len(FILES)}개 중 갱신 {len(done)}개, 누락 {len(missing)}개" + (f": {missing}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
