"""Step 20 analysis - fit the memory laws, report R^2, save the figures.

Usage:
    python step20_analyse.py "data/step20_<jobid>_*.jsonl" "data/step20c_<jobid>.jsonl"
                             [--out data]

Every fit is ordinary least squares on peak_bytes_in_use. Columns that carry a
factor R (= 16 N^2 bytes) have coefficients that read directly in units of R;
the others (const, ntsave, n_obs) read in bytes. A term's relevance is judged by
its CONTRIBUTION (coefficient x column range) as a fraction of the peak's span, not
by its standard error, which degenerates once R^2 -> 1 (see Step 6.8). The reduced
"law" keeps terms contributing >= 0.05 % of the span: terms that are zero within
noise sit at <= 0.03 %, while the smallest real one (the 0.5 R observable switch-on)
sits near 0.5 %.

Floor: every peak is max(K, formula) with K ~ 48 MiB (Step 6.1). K is re-measured
here from the N=16 control cells, and any cell with peak < FLOOR_FACTOR * K is
excluded from the fits so the floor cannot bend a slope.

Outputs in --out: step20_fits.txt, step20_fits.json, step20_A_save_states.png,
step20_B_observables.png, step20_regimes.png.
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MIB = 1024**2
GIB = 1024**3
K_STEP6 = 50_404_608
FLOOR_FACTOR = 2.0

# reference palette (dataviz skill): ordinal blue ramp for N, categorical slots
# for the regimes, recessive ink for fits, references and the floor
N_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
C_SAVE = "#2a78d6"
C_DENSE = "#eb6834"
C_DIA = "#1baf7a"
C_BASE = "#52514e"
C_FIT = "#52514e"
C_FLOOR = "#a3a29c"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 7.5, "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "font.size": 9,
})

REPORT = []


def say(line=""):
    print(line)
    REPORT.append(line)


def load(pattern):
    recs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            recs.extend(json.loads(line) for line in fh if line.strip())
    return recs


def ols(columns, y):
    """columns: {name: array}. Returns coefficients, R^2, prediction, contributions."""
    names = list(columns)
    X = np.column_stack([columns[n] for n in names]).astype(float)
    y = np.asarray(y, float)
    if len(y) <= X.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    span = y.max() - y.min()
    contrib = {n: float(b * (X[:, j].max() - X[:, j].min()))
               for j, (n, b) in enumerate(zip(names, beta))}
    return {"names": names, "beta": dict(zip(names, map(float, beta))), "r2": r2,
            "pred": pred, "resid": resid, "n": len(y),
            "max_abs_resid": float(np.abs(resid).max()),
            "max_rel_resid": float(np.abs(resid / y).max()),
            "contrib_frac": {n: (c / span if span > 0 else 0.0)
                             for n, c in contrib.items()}}


def unit(name):
    return "R" if name.startswith("R") else "B"


def print_fit(fit, tag):
    if fit is None:
        say(f"\n{tag}: not enough points to fit")
        return
    say(f"\n{tag}   n={fit['n']}   R^2 = {fit['r2']:.7f}   "
        f"max|resid| = {fit['max_abs_resid'] / MIB:.3f} MiB   "
        f"max|rel resid| = {fit['max_rel_resid']:.3%}")
    for n in fit["names"]:
        b = fit["beta"][n]
        if unit(n) == "R":
            shown = f"{b:>12.5f} R"
        elif n.startswith("N·"):
            shown = f"{b:>12.3f} B"
        else:
            shown = f"{b / MIB:>12.4f} MiB"
        say(f"    {n:<16s} = {shown}    contribution = "
            f"{fit['contrib_frac'][n]:>9.4%} of span")


def join_terms(terms):
    """terms: [(signed value, format(abs value) -> str)] -> 'a + b − c'."""
    out = ""
    for i, (v, fmt) in enumerate(terms):
        sign = ("−" if v < 0 else "") if i == 0 else (" − " if v < 0 else " + ")
        out += sign + fmt(abs(v))
    return out


def law_string(fit, labels):
    terms = []
    for n in fit["names"]:
        b = fit["beta"][n]
        if unit(n) == "R":
            terms.append((b, lambda v, n=n: f"{v:.3f}·{labels[n]}"))
        else:
            tail = "" if labels[n] == "1" else f"·{labels[n]}"
            terms.append((b / MIB, lambda v, tail=tail: f"{v:.1f} MiB{tail}"))
    return join_terms(terms)


def reduce_fit(columns, y, full, keep_always=("R",), threshold=0.0005):
    """Refit on the terms that move the peak by at least `threshold` of its span."""
    kept = [n for n in full["names"]
            if n in keep_always or abs(full["contrib_frac"][n]) >= threshold]
    return ols({n: columns[n] for n in kept}, y)


def serialise(fit):
    if fit is None:
        return None
    return {k: v for k, v in fit.items() if k not in ("pred", "resid")}


def plain_log_ticks(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def text_panel(ax, lines):
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=7.6, color=INK, transform=ax.transAxes)


def floor_line(ax, y, label=True):
    ax.axhline(y, color=C_FLOOR, ls=":", lw=1.2,
               label="floor K" if label else None)


# ------------------------------------------------------------------ study A
def study_a(ok, K, out):
    say("\n" + "=" * 96)
    say("STUDY A - save_states=True, no exp_ops, Tsit5")
    say("=" * 96)
    A = [r for r in ok if r["study"] == "A" and r["N"] > 16]
    kept = [r for r in A if r["peak"] >= FLOOR_FACTOR * K]
    say(f"cells ok: {len(A)}   kept above {FLOOR_FACTOR:g}·K: {len(kept)}   "
        f"dropped by floor: {len(A) - len(kept)}")
    if len(kept) < 5:
        say("not enough Study A data")
        return {}
    R = np.array([r["R"] for r in kept], float)
    nt = np.array([r["ntsave"] for r in kept], float)
    N = np.array([r["N"] for r in kept])
    y = np.array([r["peak"] for r in kept], float)
    traj = np.array([r["traj_bytes"] for r in kept], float)
    say(f"resident trajectory / (ntsave·R): min {np.min(traj / (nt * R)):.6f}  "
        f"max {np.max(traj / (nt * R)):.6f}")

    cols = {"R·ntsave": R * nt, "R": R, "const": np.ones_like(R), "ntsave": nt}
    labels = {"R·ntsave": "R·n_ts", "R": "R", "const": "1", "ntsave": "n_ts"}
    full = ols(cols, y)
    print_fit(full, "A full   peak = a·R·ntsave + b·R + c + d·ntsave")
    law = reduce_fit(cols, y, full)
    print_fit(law, "A law    (terms >= 0.05% of span)")

    notion = 3.0 * R * (nt + 1)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_notion = 1 - float(((y - notion) ** 2).sum()) / ss_tot
    say(f"\nStep 6 formula 3R(ntsave+1), NO free parameter:  R^2 = {r2_notion:.7f}  "
        f"max|rel err| = {np.abs(notion / y - 1).max():.3%}")

    say("\nper-N fits  peak = slope·ntsave + intercept   (in units of that N's R)")
    per_n = []
    for n_ in sorted(set(N)):
        m = N == n_
        f = ols({"ntsave": nt[m], "const": np.ones(m.sum())}, y[m])
        if f is None:
            continue
        r_ = 16 * n_ * n_
        per_n.append((n_, f["beta"]["ntsave"] / r_, f["beta"]["const"] / r_, f["r2"]))
        say(f"    N={n_:>5d}  slope = {per_n[-1][1]:.5f} R/saved state   "
            f"intercept = {per_n[-1][2]:.4f} R   R^2 = {f['r2']:.7f}")

    say("\nper-ntsave power law  peak ∝ N^p")
    per_nt = []
    for t_ in sorted(set(nt)):
        m = nt == t_
        f = ols({"logN": np.log(N[m]), "const": np.ones(m.sum())}, np.log(y[m]))
        if f is None:
            continue
        per_nt.append((t_, f["beta"]["logN"], f["r2"]))
        say(f"    ntsave={int(t_):>4d}  p = {f['beta']['logN']:.5f}   "
            f"R^2(log) = {f['r2']:.7f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.suptitle("Step 20 · Study A · save_states=True — peak memory vs saved states "
                 "and density-matrix size (Tsit5, complex128)", fontsize=11,
                 fontweight="bold", color=INK, x=0.01, ha="left")
    Ns = sorted(set(N))
    grid_nt = np.linspace(0, nt.max() * 1.05, 100)
    b = law["beta"]

    def law_eval(r_, t_):
        cols_ = {"R·ntsave": r_ * t_, "R": r_ + 0 * t_, "const": 1 + 0 * t_,
                 "ntsave": t_ + 0 * r_}
        return sum(b[n] * cols_[n] for n in law["names"])

    ax = axes[0, 0]
    for i, n_ in enumerate(Ns):
        m = N == n_
        c = N_RAMP[i % len(N_RAMP)]
        ax.plot(grid_nt, law_eval(16 * n_ * n_, grid_nt) / GIB, color=c, lw=1.0,
                ls="--")
        ax.plot(nt[m], y[m] / GIB, "o", color=c, label=f"N = {n_}")
    floor_line(ax, K / GIB)
    ax.set(title="(a) peak vs ntsave, one line per N", xlabel="ntsave",
           ylabel="peak_bytes_in_use [GiB]")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    for i, n_ in enumerate(Ns):
        m = N == n_
        ax.plot(nt[m], y[m] / R[m], "o", color=N_RAMP[i % len(N_RAMP)],
                label=f"N = {n_}")
    ax.plot(grid_nt, 3 * (grid_nt + 1), color=C_FIT, ls="--", lw=1.0,
            label="Step 6: 3·(n_ts + 1)")
    ax.set(title="(b) peak / R: all five N fall on one line", xlabel="ntsave",
           ylabel="peak / R")
    ax.legend(loc="upper left")

    ax = axes[0, 2]
    pn = np.array(per_n)
    ax.plot(pn[:, 0], pn[:, 1], "o-", color=C_SAVE, label="slope [R per saved state]")
    ax.plot(pn[:, 0], pn[:, 2], "s--", color=C_BASE, mfc="none", ms=9,
            label="intercept [R]")
    ax.set(title="(c) per-N fit: slope and intercept, both ≈ 3 R",
           xlabel="N (Hilbert dimension)", ylabel="coefficient [units of R]")
    ax.set_ylim(0, max(pn[:, 1].max(), pn[:, 2].max()) * 1.4)
    for x_, s_ in zip(pn[:, 0], pn[:, 1]):
        ax.annotate(f"{s_:.3f}", (x_, s_), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=INK2)
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    for i, n_ in enumerate(Ns):
        m = N == n_
        ax.plot(law["pred"][m] / GIB, y[m] / GIB, "o", color=N_RAMP[i % len(N_RAMP)],
                label=f"N = {n_}")
    lim = [min(y.min(), law["pred"].min()) / GIB * 0.8,
           max(y.max(), law["pred"].max()) / GIB * 1.2]
    ax.plot(lim, lim, color=C_FIT, lw=1.0, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           title=f"(d) measured vs fitted law, R² = {law['r2']:.6f}",
           xlabel="law prediction [GiB]", ylabel="measured [GiB]")
    plain_log_ticks(ax)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    for i, n_ in enumerate(Ns):
        m = N == n_
        ax.plot(nt[m], 100 * law["resid"][m] / y[m], "o", color=N_RAMP[i % len(N_RAMP)],
                label=f"N = {n_}")
        ax.plot(nt[m], 100 * (y[m] - notion[m]) / y[m], "x",
                color=N_RAMP[i % len(N_RAMP)], ms=5)
    ax.axhline(0, color=C_FIT, lw=0.8)
    ax.set(title="(e) relative residuals  (o fitted law, x Step 6 3R(n+1))",
           xlabel="ntsave", ylabel="(measured − model) / measured [%]")

    text_panel(axes[1, 2], [
        "FITTED LAW (save_states=True)",
        f"  peak = {law_string(law, labels)}",
        f"  R² = {law['r2']:.7f}   n = {law['n']}",
        f"  max |rel. resid| = {law['max_rel_resid']:.3%}",
        "",
        "STEP 6 LAW, no free parameter",
        "  peak = 3·R·(n_ts + 1)",
        f"  R² = {r2_notion:.7f}",
        "",
        "FULL MODEL (all 4 terms)",
        *[f"  {n:<9s} {full['beta'][n]:>10.4f} R" if unit(n) == "R" else
          f"  {n:<9s} {full['beta'][n] / MIB:>10.4f} MiB"
          for n in full["names"]],
        f"  R² = {full['r2']:.7f}",
        "",
        "SCALING WITH N (log-log, per ntsave)",
        f"  p = {min(p for _, p, _ in per_nt):.4f} … {max(p for _, p, _ in per_nt):.4f}"
        "   (R ∝ N², expect p → 2)",
        "",
        f"floor K = {K / MIB:.2f} MiB (N=16 control)",
        f"cells kept (peak ≥ {FLOOR_FACTOR:g}·K): {len(kept)}/{len(A)}",
        "R = 16·N² B (one complex128 ρ)",
    ])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out, "step20_A_save_states.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    say(f"\nfigure -> {path}")
    return {"full": serialise(full), "law": serialise(law), "r2_step6": r2_notion,
            "per_N": per_n, "per_ntsave_power": per_nt}


# ------------------------------------------------------------------ study B
# The save_states=False working set sits in one of two regimes depending on N: ~14 R
# everywhere except a window of sizes (Study C: from R = 96 MiB, N = 2512, up to
# somewhere below N = 3584) where it is ~16 R. That window is a compiler/allocator
# choice, not physics, so the regime is read off the measured n_obs=0 peak, not
# fitted: lo if peak/R < REGIME_SPLIT, hi otherwise.
REGIME_SPLIT = 15.0

B_LABELS = {"R·[lo]": "R·[lo]", "R·[hi]": "R·[hi]",
            "R·[n_E>0]·[lo]": "R·[n_E>0]·[lo]", "R·[n_E>0]·[hi]": "R·[n_E>0]·[hi]",
            "R·n_E": "R·n_E", "N·n_E": "N·n_E", "R·n_ts": "R·n_ts",
            "R·n_E·n_ts": "R·n_E·n_ts", "const": "1"}


def b_columns(rows, hi_of):
    R = np.array([r["R"] for r in rows], float)
    N = np.array([r["N"] for r in rows], float)
    nt = np.array([r["ntsave"] for r in rows], float)
    ne = np.array([r["n_obs"] for r in rows], float)
    on = (ne > 0).astype(float)
    hi = np.array([hi_of[r["N"]] for r in rows], float)
    lo = 1 - hi
    return {"R·[lo]": R * lo, "R·[hi]": R * hi,
            "R·[n_E>0]·[lo]": R * on * lo, "R·[n_E>0]·[hi]": R * on * hi,
            "R·n_E": R * ne, "N·n_E": N * ne, "R·n_ts": R * nt,
            "R·n_E·n_ts": R * ne * nt, "const": np.ones_like(R)}


def b_law_string(fit):
    terms = []
    for n in fit["names"]:
        b = fit["beta"][n]
        if n == "N·n_E":
            terms.append((b, lambda v: f"{v:.2f} B·N·n_E"))
        elif n == "const":
            terms.append((b / MIB, lambda v: f"{v:.1f} MiB"))
        else:
            terms.append((b, lambda v, n=n: f"{v:.4f}·{B_LABELS[n]}"))
    return join_terms(terms)


def predict(fit, cols):
    return sum(fit["beta"][n] * cols[n] for n in fit["names"])


def study_b(ok, K, out):
    say("\n" + "=" * 96)
    say("STUDY B - save_states=False, with exp_ops (quadratures), Tsit5")
    say("=" * 96)
    B = [r for r in ok if r["study"] == "B" and r["N"] > 16]
    kept = [r for r in B if r["peak"] >= FLOOR_FACTOR * K]
    say(f"cells ok: {len(B)}   kept above {FLOOR_FACTOR:g}·K: {len(kept)}   "
        f"dropped by floor: {len(B) - len(kept)}")
    base = [r for r in kept if r["n_obs"] == 0]
    if len(kept) < 20 or not base:
        say("not enough Study B data (needs the n_obs=0 baseline cells)")
        return {}

    say(f"\nBASELINE n_obs=0 (Step 6 law: (7.07 + S)·R = {7.07 + N_STAGES_B:.2f} R)")
    hi_of = {}
    for n_ in sorted({r["N"] for r in base}):
        v = np.array([r["peak_in_R"] for r in base if r["N"] == n_])
        hi_of[n_] = bool(v.mean() >= REGIME_SPLIT)
        say(f"    N={n_:>5d}  peak/R = {v.mean():.4f}   spread over ntsave = "
            f"{v.max() - v.min():.5f} R   regime = {'hi' if hi_of[n_] else 'lo'}")
    kept = [r for r in kept if r["N"] in hi_of]

    results, fits = {}, {}
    for layout in ("dense", "dia"):
        rows = [r for r in kept if r["obs_layout"] in (layout, "none")]
        y = np.array([r["peak"] for r in rows], float)
        cols = b_columns(rows, hi_of)
        say(f"\n--- layout = {layout}  (n_obs=0 cells shared as the baseline) ---")
        obs = [r for r in rows if r["n_obs"] > 0]
        st = np.array([r["obs_bytes"] / r["n_obs"] for r in obs])
        say(f"observable storage per op: {np.mean(st / [r['R'] for r in obs]):.5f} R"
            f" = {np.mean(st / [r['N'] for r in obs]):.2f} B·N")
        full = ols(cols, y)
        print_fit(full, f"B[{layout}] full   9 terms")
        law = reduce_fit(cols, y, full, keep_always=("R·[lo]", "R·[hi]"))
        print_fit(law, f"B[{layout}] LAW    (terms >= 0.05% of span)")

        transient = y - np.array([r["in_use_before"] for r in rows], float)
        tfull = ols(cols, transient)
        tlaw = reduce_fit(cols, transient, tfull, keep_always=("R·[lo]", "R·[hi]"))
        print_fit(tlaw, f"B[{layout}] TRANSIENT law  (peak − bytes resident before "
                  "the solve)")

        say(f"\nB[{layout}] per-N marginal cost, Δ = peak(n_obs) − peak(0) at equal "
            "(N, ntsave):")
        bmap = {(r["N"], r["ntsave"]): r["peak"] for r in base}
        per_n = []
        for n_ in sorted(hi_of):
            sub = [r for r in obs if r["N"] == n_ and (n_, r["ntsave"]) in bmap]
            if len(sub) < 3:
                continue
            ne = np.array([r["n_obs"] for r in sub], float)
            d = np.array([r["peak"] - bmap[(n_, r["ntsave"])] for r in sub], float)
            f = ols({"n_obs": ne, "const": np.ones_like(ne)}, d)
            r_ = 16.0 * n_ * n_
            per_n.append((n_, f["beta"]["n_obs"] / r_, f["beta"]["const"] / r_, f["r2"]))
            say(f"    N={n_:>5d}  per observable = {f['beta']['n_obs'] / r_:.5f} R "
                f"({f['beta']['n_obs'] / n_:8.2f} B·N)   switch-on = "
                f"{f['beta']['const'] / r_:+.4f} R   R^2 = {f['r2']:.6f}")
        results[layout] = {"full": serialise(full), "law": serialise(law),
                           "transient_law": serialise(tlaw), "per_N": per_n}
        fits[layout] = (rows, law, y, tlaw)
    return {"results": results, "fits": fits, "hi_of": hi_of, "kept": kept,
            "n_all": len(B)}


N_STAGES_B = 7


def study_c(ok, K):
    C = sorted([r for r in ok if r["study"] == "C"], key=lambda r: r["N"])
    if not C:
        return []
    say("\n" + "=" * 96)
    say("STUDY C - boundary scan of the save_states=False working set (ntsave=20)")
    say("=" * 96)
    for n_ in sorted({r["N"] for r in C}):
        row = {f"{r['n_obs']}{r['obs_layout']}": r["peak_in_R"] for r in C if r["N"] == n_}
        say(f"    N={n_:>5d}  R={16 * n_ * n_ / MIB:8.2f} MiB   "
            + "   ".join(f"{k:>7s}: {v:8.4f} R" for k, v in sorted(row.items())))
    return C


def figure_b(res_b, C, K, out):
    fits, hi_of, kept = res_b["fits"], res_b["hi_of"], res_b["kept"]
    fig = plt.figure(figsize=(15.5, 13))
    gs = fig.add_gridspec(3, 3)
    fig.suptitle("Step 20 · Study B · save_states=False with observables — "
                 "dependence on N, n_obs, ntsave and observable layout",
                 fontsize=11, fontweight="bold", color=INK, x=0.01, ha="left")
    Ns = sorted(hi_of)
    nts = sorted({r["ntsave"] for r in kept})
    nt_mid = nts[len(nts) // 2]

    for k, layout in enumerate(("dense", "dia")):
        ax = fig.add_subplot(gs[0, k])
        rows, law, _, _ = fits[layout]
        grid_ne = np.arange(0, max(r["n_obs"] for r in rows) + 1)
        for i, n_ in enumerate(Ns):
            sub = sorted([r for r in rows if r["N"] == n_ and r["ntsave"] == nt_mid],
                         key=lambda r: r["n_obs"])
            if not sub:
                continue
            c = N_RAMP[i % len(N_RAMP)]
            fake = [{"R": 16.0 * n_ * n_, "N": n_, "ntsave": nt_mid, "n_obs": e}
                    for e in grid_ne]
            ax.plot(grid_ne, predict(law, b_columns(fake, hi_of)) / (16.0 * n_ * n_),
                    color=c, lw=1.0, ls="--")
            ax.plot([r["n_obs"] for r in sub], [r["peak_in_R"] for r in sub], "o",
                    color=c, label=f"N = {n_}" + (" (hi)" if hi_of[n_] else ""))
        ax.set(title=f"({'ab'[k]}) {layout} observables · ntsave = {nt_mid} · "
               "dashed = law", xlabel="number of observables n_obs",
               ylabel="peak / R")
        ax.legend(loc="upper left" if layout == "dense" else "center right")

    ax = fig.add_subplot(gs[0, 2])
    n_big = max(Ns)
    ne_max = max(r["n_obs"] for r in kept)
    for sel, c, lab in ((lambda r: r["n_obs"] == 0, C_BASE, "n_obs = 0"),
                        (lambda r: r["n_obs"] == ne_max and r["obs_layout"] == "dia",
                         C_DIA, f"n_obs = {ne_max}, DIA"),
                        (lambda r: r["n_obs"] == ne_max and r["obs_layout"] == "dense",
                         C_DENSE, f"n_obs = {ne_max}, dense")):
        sub = sorted([r for r in kept if r["N"] == n_big and sel(r)],
                     key=lambda r: r["ntsave"])
        ax.plot([r["ntsave"] for r in sub], [r["peak_in_R"] for r in sub], "o-",
                color=c, label=lab)
    ax.set(title=f"(c) no ntsave dependence (N = {n_big})", xlabel="ntsave",
           ylabel="peak / R")
    ax.legend(loc="center right")

    ax = fig.add_subplot(gs[1, 0])
    pts = [r for r in kept if r["n_obs"] <= 1] + list(C)
    # (label, colour, marker, shift in R, marker style, line style)
    series = ((lambda r: r["n_obs"] == 0,
               C_BASE, "o", 0.0, dict(ms=4), "-", "n_obs = 0"),
              (lambda r: r["n_obs"] == 1 and r["obs_layout"] == "dia",
               C_DIA, "s", 0.0, dict(ms=9, mfc="none"), "-", "n_obs = 1, DIA"),
              (lambda r: r["n_obs"] == 1 and r["obs_layout"] == "dense",
               C_DENSE, "D", 2.0, dict(ms=4), "", "n_obs = 1, dense  (−2 R)"))
    for sel, c, mk, shift, style, ls, lab in series:
        by_n = {}
        for r in pts:
            if sel(r):
                by_n.setdefault(r["N"], []).append(r["peak_in_R"] - shift)
        xs = sorted(by_n)
        ax.plot(xs, [np.mean(by_n[x]) for x in xs], mk + ls, color=c, label=lab,
                lw=1.0, **style)
    # window edges bracketed by Study C: R = 96 MiB and 192 MiB
    ax.axvspan(np.sqrt(96 * MIB / 16), np.sqrt(192 * MIB / 16), color=GRID,
               alpha=0.6, lw=0, label="96 MiB ≤ R < 192 MiB")
    ax.axhline(7.07 + N_STAGES_B, color=C_FIT, ls="--", lw=0.8,
               label="Step 6: (7.07 + S)·R")
    ax.set(title="(d) the ~16 R window in N (grid + Study C scan)",
           xlabel="N (Hilbert dimension)", ylabel="peak / R")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    for layout, c, mk in (("dense", C_DENSE, "o"), ("dia", C_DIA, "s")):
        rows, _, _, _ = fits[layout]
        ne = np.array([r["n_obs"] for r in rows])
        tr = np.array([(r["peak"] - r["in_use_before"]) / r["R"] for r in rows])
        res_ = np.array([r["in_use_before"] / r["R"] for r in rows])
        ax.plot(ne, tr, mk, color=c, mfc="none", label=f"{layout}: solver transient")
        ax.plot(ne, res_, mk, color=c, label=f"{layout}: resident inputs")
    ax.set(title="(e) resident inputs vs solver transient (all N, ntsave)",
           xlabel="number of observables n_obs", ylabel="bytes / R")
    ax.legend(loc="upper left", ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    lo, hi = np.inf, 0
    for layout, c, mk in (("dense", C_DENSE, "o"), ("dia", C_DIA, "s")):
        _, law, y, _ = fits[layout]
        ax.plot(law["pred"] / GIB, y / GIB, mk, color=c, mfc="none",
                label=f"{layout}: R² = {law['r2']:.6f}")
        lo, hi = min(lo, y.min(), law["pred"].min()), max(hi, y.max(), law["pred"].max())
    lim = [lo / GIB * 0.8, hi / GIB * 1.2]
    ax.plot(lim, lim, color=C_FIT, lw=1.0, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           title="(f) measured vs law", xlabel="law prediction [GiB]",
           ylabel="measured [GiB]")
    plain_log_ticks(ax)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[2, 0])
    for layout, c, mk in (("dense", C_DENSE, "o"), ("dia", C_DIA, "s")):
        rows, law, y, _ = fits[layout]
        ax.plot([r["n_obs"] + (0.15 if layout == "dia" else -0.15) for r in rows],
                100 * law["resid"] / y, mk, color=c, mfc="none", label=layout)
    ax.axhline(0, color=C_FIT, lw=0.8)
    ax.set(title="(g) relative residuals of the law",
           xlabel="number of observables n_obs",
           ylabel="(measured − law) / measured [%]")
    ax.legend(loc="upper right")

    lines = []
    for layout in ("dense", "dia"):
        _, law, _, tlaw = fits[layout]
        lines += [f"LAW, {layout} observables   (R² = {law['r2']:.7f}, n = {law['n']}, "
                  f"max |rel. resid| = {law['max_rel_resid']:.3%})",
                  "  peak = " + b_law_string(law)]
        if tlaw:
            lines += ["  transient (peak − resident inputs) = " + b_law_string(tlaw)]
        lines += [""]
    lines += ["[lo]/[hi]: save_states=False working-set regime of that N, read off",
              "the measured n_obs=0 peak (lo if < 15 R). Study C: hi exactly when",
              "96 MiB ≤ R < 192 MiB (N = 2512…3520 in complex128); see panel (d).",
              "[n_E>0] = 1 if any observable.  N·n_E term: DIA storage, 2 diags × 16 B.",
              "Law keeps the terms that move the peak by ≥ 0.05 % of its span;",
              "full 9-term coefficients are in step20_fits.txt.",
              f"floor K = {K / MIB:.2f} MiB   cells kept (peak ≥ {FLOOR_FACTOR:g}·K): "
              f"{len(kept)}/{res_b['n_all']}   R = 16·N² B"]
    text_panel(fig.add_subplot(gs[2, 1:]), lines)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out, "step20_B_observables.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    say(f"\nfigure -> {path}")


# ------------------------------------------------------------------ regimes
def regimes(ok, K, out):
    common = sorted({r["N"] for r in ok if r["study"] == "A"}
                    & {r["N"] for r in ok if r["study"] == "B"} - {16})
    if not common:
        return
    fig, axes = plt.subplots(1, len(common), figsize=(5.4 * len(common), 4.6),
                             squeeze=False)
    fig.suptitle("Step 20 · the three regimes side by side (peak / R, log scale)",
                 fontsize=11, fontweight="bold", color=INK, x=0.01, ha="left")
    ne_max = max(r["n_obs"] for r in ok if r["study"] == "B")
    series = (
        (lambda r: r["study"] == "A", C_SAVE, "save_states=True"),
        (lambda r: r["study"] == "B" and r["n_obs"] == ne_max
         and r["obs_layout"] == "dense", C_DENSE,
         f"save_states=False, {ne_max} dense obs"),
        (lambda r: r["study"] == "B" and r["n_obs"] == ne_max
         and r["obs_layout"] == "dia", C_DIA, f"save_states=False, {ne_max} DIA obs"),
        (lambda r: r["study"] == "B" and r["n_obs"] == 0, C_BASE,
         "save_states=False, no obs"),
    )
    for ax, n_ in zip(axes[0], common):
        for sel, c, lab in series:
            sub = sorted([r for r in ok if r["N"] == n_ and sel(r)],
                         key=lambda r: r["ntsave"])
            ax.plot([r["ntsave"] for r in sub], [r["peak_in_R"] for r in sub], "o-",
                    color=c, label=lab)
        floor_line(ax, K / (16 * n_ * n_))
        ax.set(yscale="log", title=f"N = {n_}  (R = {16 * n_ * n_ / MIB:.1f} MiB)",
               xlabel="ntsave", ylabel="peak / R")
    axes[0, 0].legend(loc="center right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, "step20_regimes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    say(f"\nfigure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "data"))
    args = ap.parse_args()
    recs = [r for p in args.pattern for r in load(p)]
    if not recs:
        sys.exit(f"no records match {args.pattern}")
    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    say(f"STEP 20 analysis   files: {' '.join(args.pattern)}")
    say(f"records: {len(recs)}   ok: {len(ok)}   OOM/ERR: {len(bad)}")
    for r in bad:
        say(f"    {r['status']}: study {r['study']} N={r['N']} ntsave={r['ntsave']} "
            f"n_obs={r['n_obs']} {r['obs_layout']}  {r.get('err', '')[:120]}")

    floor = [r["peak"] for r in ok if r["N"] == 16]
    K = float(max(floor)) if floor else float(K_STEP6)
    say(f"\nFLOOR  N=16 control peaks: {[f'{p:,d}' for p in floor]}")
    say(f"       K used = {K:,.0f} B = {K / MIB:.3f} MiB   (Step 6: {K_STEP6:,d} B)")

    res_a = study_a(ok, K, args.out)
    res_b = study_b(ok, K, args.out)
    C = study_c(ok, K)
    if res_b:
        figure_b(res_b, C, K, args.out)
    regimes(ok, K, args.out)

    with open(os.path.join(args.out, "step20_fits.txt"), "w") as fh:
        fh.write("\n".join(REPORT) + "\n")
    with open(os.path.join(args.out, "step20_fits.json"), "w") as fh:
        json.dump({"K": K, "A": res_a, "B": res_b.get("results") if res_b else None,
                   "B_regime_hi": res_b.get("hi_of") if res_b else None,
                   "C": [{k: r[k] for k in ("N", "n_obs", "obs_layout", "peak",
                                            "peak_in_R")} for r in C]},
                  fh, indent=1, default=float)
    print(f"\nreport -> {os.path.join(args.out, 'step20_fits.txt')}")


if __name__ == "__main__":
    main()
