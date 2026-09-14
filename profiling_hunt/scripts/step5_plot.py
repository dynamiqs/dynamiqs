"""Step 5 figure: dynamiqs trajectory memory vs number of saved states."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GIB = 1024**3
R = 64_000_000
LIMIT = 45_285_900_288
EFF = 0.89 * LIMIT

rows = [json.loads(l) for l in open("data/step5_69388.jsonl")]
A = sorted({r["ntsave"]: r for r in rows if r["save_states"] and r["n_exp_ops"] == 0
            and r["max_steps"] == 100000 and r["gradient"] == "none"
            and r["layout"] == "dia"}.values(), key=lambda r: r["ntsave"])
off = [r for r in rows if not r["save_states"] and r["n_exp_ops"] == 0]

nt = np.array([r["ntsave"] for r in A], float)
pk = np.array([r["peak"] for r in A], float)
tr = np.array([r["traj_bytes"] for r in A], float)
floor = off[0]["peak"]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

ax = axes[0]
g = np.linspace(0, 215, 60)
ax.plot(nt, tr / GIB, "o", ms=9, color="tab:blue", label="resident  res.states")
ax.plot(nt, pk / GIB, "s", ms=9, color="tab:red", label="peak_bytes_in_use")
ax.plot(g, g * R / GIB, "-", color="tab:blue", lw=1.5, label="fit: 1.000 R x ntsave")
ax.plot(g, (192_143_892 + 192_000_034 * g) / GIB, "-", color="tab:red", lw=1.5,
        label="fit: 3.002 R + 3.000 R x ntsave")
ax.axhline(floor / GIB, color="tab:green", ls="-.", lw=1.5,
           label=f"save_states=False floor = {floor / GIB:.2f} GiB (flat)")
ax.axhline(LIMIT / GIB, color="purple", lw=1.6, label="bytes_limit 42.18 GiB")
ax.axhline(EFF / GIB, color="orange", ls=":", lw=1.8,
           label="Step 2 effective ceiling ~37.5 GiB")
ax.axvline(208, color="grey", ls="--", lw=1.2)
ax.text(208, 1.5, "  max 208 states\n  (naive: 629)", fontsize=8.5, color="grey")
ax.set_xlabel("ntsave  (number of saved density matrices)")
ax.set_ylabel("GiB")
ax.set_title("Step 5 — dynamiqs save_states=True costs 3× the trajectory\n"
             "(N=2000, R=61.04 MiB, Tsit5)")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=.3)
ax.set_xlim(0, 215)
ax.set_ylim(0, 45)

ax = axes[1]
labels, vals, cols = [], [], []
for g_ in ("none", "forward", "checkpointed", "direct"):
    r = [x for x in rows if x["ntsave"] == 11 and x["save_states"]
         and x["gradient"] == g_ and x["n_exp_ops"] == 0
         and x["max_steps"] == 100000 and x["layout"] == "dia"]
    if r:
        labels.append(f"gradient=\n{g_}")
        vals.append(r[0]["peak_in_R"])
        cols.append("tab:red" if g_ == "direct" else "tab:green")
labels += ["save_states\n=False", "exp_ops only\n(no states)"]
vals += [off[0]["peak_in_R"],
         [x for x in rows if x["n_exp_ops"] == 2][0]["peak_in_R"]]
cols += ["tab:blue", "tab:purple"]
x = np.arange(len(vals))
ax.bar(x, vals, color=cols, alpha=.85)
for i, v in enumerate(vals):
    ax.text(i, v + 0.7, f"{v:.2f} R", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("peak_bytes_in_use / R")
ax.set_title("At ntsave=11: adjoint choice and what you save\n"
             "DirectAdjoint costs +11.000 R = +1 R per saved state")
ax.grid(alpha=.3, axis="y")
ax.set_ylim(0, 54)

fig.tight_layout()
fig.savefig("data/step5_save_path.png", dpi=130)
print("wrote data/step5_save_path.png")
