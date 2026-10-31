"""개정 그림: Fig. 7 독립 사건 기준 변화 시점 차이 분포 (E5b 수확·갱신 실적, E5c 산지전용 고시)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path(__file__).resolve().parent / "results"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
t = json.loads((D / "timing_summary.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
h = t["E5b_수확갱신사업"]["전체"]
series = [("AlphaEarth 최대 변화", h["AE_정점"], "#2166ac"), ("Sentinel-2 NDVI 최대 감소", h["NDVI_최대감소"], "#b2abd2"),
          ("최종지도(AE+S2) 변화연도", h["최종지도_연도"], "#999999")]
xs = range(-2, 6)
w = 0.27
for k, (name, s, c) in enumerate(series):
    d = {int(a): b for a, b in s["시차분포(신호−사건)"].items()}
    tot = sum(d.values())
    ax.bar([x + (k - 1) * w for x in xs], [d.get(x, 0) / tot for x in xs], width=w, color=c,
           label=f"{name} (n={s['n']}, 일치 {s['정확일치']:.2f}, ±1년 {s['±1년']:.2f})")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xticks(list(xs))
ax.set_xlabel("신호 연도 - 사업 연도 (년)")
ax.set_ylabel("소반 비율")
ax.set_title("(a) 국유림 수확·갱신 사업 (2020–2024)", fontsize=11, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

ax = axes[1]
c5 = t["E5c_산지전용고시"]
xs2 = range(-3, 7)
for k, (name, key, col) in enumerate([("AlphaEarth 최대 변화", "AE 시차분포(고시−정점)", "#2166ac"),
                                      ("Sentinel-2 NDVI 최대 감소", "NDVI 시차분포(고시−감소)", "#b2abd2")]):
    d = {int(a): b for a, b in c5[key].items()}
    tot = sum(d.values())
    ax.bar([x + (k - 0.5) * 0.38 for x in xs2], [d.get(x, 0) / tot for x in xs2], width=0.38, color=col, label=name)
ax.axvline(-0.5, color="k", lw=0.8, ls="--")
ym = ax.get_ylim()[1]
ax.text(-0.6, ym * 0.7, "← 고시 뒤 변화", fontsize=8, ha="right")
ax.text(-0.4, ym * 0.7, "고시 전 변화 →", fontsize=8, ha="left")
ax.set_xticks(list(xs2))
ax.set_xlabel("고시 연도 - 신호 연도 (년)")
ax.set_title(f"(b) 보전산지 해제 고시 전용확정 필지 (n={c5['필지']})", fontsize=11, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(D / "fig7_timing.png", dpi=200)
print("ok")
