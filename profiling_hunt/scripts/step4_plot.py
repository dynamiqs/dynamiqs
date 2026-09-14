"""Step 4 figure: the attribution bar chart."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
R = 64_000_000      # 16 * 2000^2, complex128

rows = [
    ("bare ODE\ndY/dt = M@Y", 808_200_192, 12.393, 1.0),
    ("hand-written\nLindblad (diffrax)", 1_536_009_472, 23.393, 7.0),
    ("dynamiqs\ndense", 1_220_209_664, None, 3.0),
    ("dynamiqs cat\nDENSE ops (control)", 1_220_241_152, None, 3.0),
    ("dynamiqs cat\nsparse DIA ops", 900_173_568, None, 0.002),
]

labels = [r[0] for r in rows]
peaks = np.array([r[1] for r in rows], dtype=float)
preds = [r[2] for r in rows]
ops = np.array([r[3] for r in rows], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

ax = axes[0]
x = np.arange(len(rows))
bars = ax.bar(x, peaks / R, color=["tab:blue", "tab:red", "tab:green",
                                   "tab:olive", "tab:purple"], alpha=.85)
for i, p in enumerate(preds):
    if p is not None:
        ax.plot([i - .42, i + .42], [p, p], "k--", lw=2,
                label="hand-derived prediction" if i == 1 else None)
for i, (b, v) in enumerate(zip(bars, peaks / R)):
    ax.text(b.get_x() + b.get_width() / 2, v + .3, f"{v:.2f} R",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("peak_bytes_in_use  /  R      (R = 16N² = 61.04 MiB)")
ax.set_title("Step 4 — peak memory attribution (N=2000, Tsit5, complex128)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(fontsize=9)
ax.grid(alpha=.3, axis="y")
ax.set_ylim(0, 28)

ax = axes[1]
ax.bar(x, np.maximum(ops, 1e-3), color=["tab:blue", "tab:red", "tab:green",
                                        "tab:olive", "tab:purple"], alpha=.85)
ax.set_yscale("log")
for i, v in enumerate(ops):
    ax.text(i, max(v, 1e-3) * 1.25, f"{v:g} R", ha="center", fontsize=9,
            fontweight="bold")
ax.set_ylabel("operator storage / R  (log scale)")
ax.set_title("Operator storage: DIA delivers a 1500× saving over dense")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.grid(alpha=.3, axis="y", which="both")
ax.annotate("", xy=(4, 0.002), xytext=(3, 3.0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.text(3.5, 0.09, "1500×", fontsize=11, fontweight="bold", ha="center")

fig.tight_layout()
fig.savefig("data/step4_attribution.png", dpi=130)
print("wrote data/step4_attribution.png")
