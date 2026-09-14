"""Step 3 figures: the 3a step-count fit and the 3b stage-count fit."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
MIB = 1024**2
S = 128_000_000          # state_bytes = 8 * 4000^2
LIMIT = 45_285_900_288   # Step 1
EFF = 0.89 * LIMIT       # Step 2 effective ceiling for grow-while-holding

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/step3"

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

# ------------------------------------------------------------------ 3a
ax = axes[0]
ms = np.array([11, 40, 101])
peak = np.array([4_480_008_192, 15_616_008_960, 39_040_010_496], dtype=float)
ys = np.array([1_408_000_000, 5_120_000_000, 12_928_000_000], dtype=float)

ax.plot(ms, ys / GIB, "o", ms=9, color="tab:blue",
        label="resident trajectory  sol.ys")
ax.plot(ms, peak / GIB, "s", ms=9, color="tab:red",
        label="peak_bytes_in_use")
grid = np.linspace(0, 110, 50)
ax.plot(grid, (grid * S) / GIB, "-", color="tab:blue", lw=1.5,
        label="fit: 1.000 x S x n_slots")
ax.plot(grid, (2 * S + 3 * grid * S) / GIB, "-", color="tab:red", lw=1.5,
        label="fit: 2.000 S + 3.000 x S x n_slots")

# SaveAt(t1=True): flat
ax.plot([11, 101], [128_000_000 / GIB] * 2, "^--", color="tab:green", ms=9,
        label="SaveAt(t1=True) resident (FLAT: 10 vs 100 steps)")
ax.axhline(LIMIT / GIB, color="purple", lw=1.6, label="bytes_limit 42.18 GiB")
ax.axhline(EFF / GIB, color="orange", ls=":", lw=1.6,
           label="Step 2 effective ceiling ~37.5 GiB")
ax.set_xlabel("number of stored trajectory slots (= max_steps)")
ax.set_ylabel("GiB")
ax.set_title("3a — memory vs stored steps (Euler, N=4000)")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=.3)

# ------------------------------------------------------------------ 3b
ax = axes[1]
st = np.array([2, 4, 7, 14])
names = ["Ralston", "Bosh3", "Tsit5", "Dopri8"]
pk_solve = np.array([946_333_184, 1_202_333_184, 1_586_333_184, 2_482_333_184],
                    dtype=float)
pk_manual = np.array([2_304_005_376, 4_096_006_656, 5_504_004_352, 9_088_010_496],
                     dtype=float)

ax.plot(st, pk_solve / S, "o", ms=10, color="tab:green",
        label="diffeqsolve (real API)")
ax.plot(st, pk_manual / S, "s", ms=10, color="tab:red",
        label="manual solver.step() loop (un-jitted)")
g = np.linspace(0, 15, 40)
ax.plot(g, 5.3932 + 1.0000 * g, "-", color="tab:green", lw=1.6,
        label="fit: 5.393 S + 1.000 S x stages   (b = 1.000 x state)")
b3, a3 = np.polyfit(st, pk_manual / S, 1)
ax.plot(g, a3 + b3 * g, "--", color="tab:red", lw=1.6,
        label=f"fit: {a3:.2f} S + {b3:.2f} S x stages   (b = {b3:.2f} x state)")
ax.plot([1], [805_750_016 / S], "*", ms=16, color="grey",
        label="Euler (not a tableau RK — off the line)")
for x, y, nm in zip(st, pk_solve / S, names):
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(4, -12),
                fontsize=8, color="darkgreen")
ax.set_xlabel("number of solver stages")
ax.set_ylabel("peak_bytes_in_use  /  state_bytes")
ax.set_title("3b — memory vs stages (10 steps fixed, N=4000)")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=.3)

fig.tight_layout()
fig.savefig(f"{OUT}_fits.png", dpi=130)
print(f"wrote {OUT}_fits.png")
