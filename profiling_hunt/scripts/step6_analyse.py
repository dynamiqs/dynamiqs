"""Step 6 analysis - multidimensional fit + R^2 for the two Step-5 memory formulas.

Formulas under test, R = 16 N^2 (one complex128 density matrix), S = n_stages:

  (1) save_states=False :  peak = (C0 + S) * R
  (2) save_states=True  :  peak = max( (C0 + S)*R , 3*R*(ntsave+1) ) [+ 1*R*ntsave if Direct]

with C0 = 7 as published (calibrated at nL=2 from the single Tsit5 point, 14.065 R).

TWO STRUCTURAL FACTS the grid forces into the model, neither of which the published
formulas contain:

  * There is a hard fixed floor K. At N=64 every cell in the grid - all four methods,
    both save settings, both gradients, all ten ntsave values - reports the SAME peak
    to the byte. K is device/stack overhead (XLA scratch + cuBLAS workspace), it does
    not scale with R, and it sits in a max() with the formula rather than adding to
    it: at N=64 save_states=True with 10 states is byte-identical to save_states=False,
    so the trajectory buffer and the overhead are never live at the same instant.
  * Consequently every fit is reported twice: over ALL cells, and over the
    FORMULA-DOMINANT subset where the formula's own prediction exceeds 3K, so that
    the max() is unambiguous and the coefficients are actually identified.

Fits reported, in order of evidential value:
  M0  the published formulas, as written, ZERO free parameters.
  M1  the published formulas wrapped in max(K, .), ONE free parameter (K).
  M2  pooled multidimensional linear refit with branch indicators - the coefficients
      recovered from the data instead of assumed.
  M3  per-arm refits, where each individual coefficient is cleanest.

R^2 is always given for two targets, and the difference matters:
  R-units (peak/R): scale-free, every cell weighs the same -> the honest number.
  bytes   (peak)  : unweighted OLS over ~6 orders of magnitude, so the biggest cells
                    dominate and R^2 is flattered -> reported for completeness only.
"""

import glob
import json
import sys
from collections import Counter

import numpy as np

GIB = 1024**3
MIB = 1024**2
C0_DOC = 7.0


def load(paths):
    recs = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def ols(X, y, names):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    s2 = sse / max(n - k, 1)
    try:
        se = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(X.T @ X)), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return dict(beta=beta, se=se, r2=r2, adj=adj, n=n, k=k, resid=resid,
                rmse=float(np.sqrt(s2)), names=names)


def show(tag, f, expect=None):
    print(f"\n  {tag}\n    n={f['n']}  k={f['k']}   R^2 = {f['r2']:.6f}   "
          f"adj R^2 = {f['adj']:.6f}   RMSE = {f['rmse']:.4g}")
    w = max(len(s) for s in f["names"])
    for i, nm in enumerate(f["names"]):
        e = ""
        if expect is not None and nm in expect:
            e = (f"   published {expect[nm]:>6.3f}   "
                 f"dev {f['beta'][i] - expect[nm]:+.4f}")
        print(f"    {nm:<{w}s} = {f['beta'][i]:+14.6f}  +- {f['se'][i]:>12.6f}{e}")


def r2_of(pred, y):
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def rel(pred, y, label):
    a = np.abs(pred / y - 1.0)
    print(f"    {label:<34s} median |rel err| = {np.median(a):8.3%}   "
          f"p90 = {np.quantile(a, .9):8.3%}   max = {a.max():9.2%}")


def main():
    paths = sorted(glob.glob(sys.argv[1])) if len(sys.argv) > 1 else []
    if not paths:
        print("usage: step6_analyse.py '<glob of step6_*.jsonl>'")
        return
    recs = load(paths)
    ok = [r for r in recs if r["status"] == "ok"]
    oomr = [r for r in recs if r["status"] == "OOM"]
    errr = [r for r in recs if r["status"] == "ERR"]

    print("=" * 104)
    print("STEP 6 - GRID SEARCH VALIDATION OF THE TWO MEMORY FORMULAS  (metric: peak_bytes_in_use)")
    print("=" * 104)
    print(f"  shards/files : {len(paths)}")
    print(f"  cells        : {len(recs)}   ok={len(ok)}  OOM={len(oomr)}  ERR={len(errr)}")
    if errr:
        print(f"  ERR kinds    : {Counter(r.get('exc') for r in errr)}")
    print(f"  bytes_limit  : {ok[0]['bytes_limit']:,d} "
          f"({ok[0]['bytes_limit'] / GIB:.4f} GiB)")
    if not ok:
        return

    Ns = sorted({r["N"] for r in ok})
    nts = sorted({r["ntsave"] for r in ok})
    ms = sorted({r["method"] for r in ok})
    stage_of = {m: next(r["n_stages"] for r in ok if r["method"] == m) for m in ms}
    print(f"  N            : {Ns}")
    print(f"  ntsave       : {nts}")
    print(f"  methods      : {[(m, stage_of[m]) for m in ms]}  (label, n_stages)")
    print(f"  peak range   : {min(r['peak'] for r in ok) / MIB:.2f} MiB .. "
          f"{max(r['peak'] for r in ok) / GIB:.4f} GiB  "
          f"({max(r['peak'] for r in ok) / min(r['peak'] for r in ok):.4g}x)")

    print("\n  OOM coverage of the save_states=True arm (cells that did not fit the card):")
    for N in Ns:
        row = [r for r in recs if r["N"] == N and r["save_states"]]
        o = sum(1 for r in row if r["status"] == "OOM")
        if row:
            print(f"    N={N:>5d}  {o:>3d}/{len(row):>3d} OOM   "
                  f"largest ntsave that fits = "
                  f"{max((r['ntsave'] for r in row if r['status'] == 'ok'), default=None)}")

    a = {k: np.array([r[k] for r in ok], dtype=float)
         for k in ("N", "R", "ntsave", "n_stages", "peak")}
    sv = np.array([1.0 if r["save_states"] else 0.0 for r in ok])
    dr = np.array([1.0 if r["is_direct"] else 0.0 for r in ok])
    R, nt, S, peak, Nv = a["R"], a["ntsave"], a["n_stages"], a["peak"], a["N"]
    yR = peak / R
    one = np.ones_like(yR)

    # ---------- the fixed floor K ----------
    K = float(peak.min())
    print("\n" + "=" * 104)
    print("0) THE FIXED FLOOR K - a term neither published formula contains")
    print("=" * 104)
    print(f"    K = min(peak) over the whole grid = {K:,.0f} B = {K / MIB:.3f} MiB")
    plateau = peak <= K * 1.02
    print(f"    cells sitting on that plateau (peak <= 1.02 K): {plateau.sum()}/{len(peak)}")
    for N in Ns:
        m = (Nv == N)
        print(f"      N={N:>5d}  n={m.sum():>4d}  on plateau {int((plateau & m).sum()):>4d}"
              f"   peak min={peak[m].min() / MIB:10.3f} MiB  max={peak[m].max() / MIB:12.3f} MiB")
    m64 = Nv == Ns[0]
    print(f"    at N={Ns[0]}: distinct peak values across all "
          f"{int(m64.sum())} cells = {sorted(set(peak[m64].astype(int)))[:6]}")
    print("    -> K is independent of method, ntsave, save_states and gradient, and it")
    print("       sits in a max() with the formula (not added): at the smallest N the")
    print("       save_states=True and save_states=False cells are byte-identical.")

    pub_floor = (C0_DOC + S) * R
    pub_ramp = 3.0 * R * (nt + 1.0) + dr * 1.0 * R * nt
    pub = np.where(sv > 0, np.maximum(pub_floor, pub_ramp), pub_floor)
    dom = pub >= 3.0 * K          # formula-dominant subset: max() unambiguous

    # ---------- M0 / M1 ----------
    print("\n" + "=" * 104)
    print("A) M0 - THE PUBLISHED FORMULAS AS WRITTEN, ZERO FREE PARAMETERS")
    print("=" * 104)
    for m, lab in ((np.ones_like(sv, bool), "all cells"),
                   (dom, f"formula-dominant (pub >= 3K), n={int(dom.sum())}")):
        print(f"    -- {lab}")
        print(f"       R^2 R-units = {r2_of(pub[m] / R[m], yR[m]):9.6f}    "
              f"R^2 bytes = {r2_of(pub[m], peak[m]):9.6f}")
        rel(pub[m], peak[m], lab)
    for br, lab in ((sv == 0, "formula (1) save_states=False"),
                    (sv > 0, "formula (2) save_states=True")):
        m = br & dom
        if m.sum() > 2:
            print(f"    -- {lab}, dominant only (n={int(m.sum())}):  "
                  f"R^2 R-units = {r2_of(pub[m] / R[m], yR[m]):.6f}")
            rel(pub[m], peak[m], lab)

    print("\n" + "=" * 104)
    print("B) M1 - THE PUBLISHED FORMULAS WRAPPED IN max(K, .), ONE FREE PARAMETER")
    print("=" * 104)
    m1 = np.maximum(K, pub)
    print(f"    K = {K:,.0f} B ({K / MIB:.3f} MiB), taken as min(peak)")
    print(f"    R^2 R-units, all cells = {r2_of(m1 / R, yR):9.6f}    "
          f"R^2 bytes = {r2_of(m1, peak):9.6f}")
    rel(m1, peak, "M1 all cells")
    for N in Ns:
        m = Nv == N
        print(f"      N={N:>5d}  R^2 R-units = {r2_of(m1[m] / R[m], yR[m]):9.6f}   "
              f"median |rel err| = {np.median(np.abs(m1[m] / peak[m] - 1)):7.3%}")

    # ---------- M2 pooled ----------
    print("\n" + "=" * 104)
    print("C) M2 - POOLED MULTIDIMENSIONAL REFIT (branch indicators, coefficients free)")
    print("   peak/R = (1-sv)(a0 + a1 S) + sv(c0 + c1 nt) + c2 sv nt direct")
    print("            + direct(d0 + d1 S) + d2 direct sv")
    print("   The Direct terms are split into a constant and a per-stage part because E7")
    print("   shows the penalty does not scale with ntsave at all; c2 is kept in the model")
    print("   precisely so that the published +1 R per saved state can be refuted by its")
    print("   own coefficient rather than by omission.")
    print("=" * 104)
    names = ["a0 floor intercept", "a1 per stage", "c0 ramp intercept",
             "c1 per saved state", "c2 Direct per state", "d0 Direct const",
             "d1 Direct per stage", "d2 Direct x save"]
    exp = {"a1 per stage": 1.0, "c1 per saved state": 3.0,
           "c2 Direct per state": 1.0, "c0 ramp intercept": 3.0,
           "a0 floor intercept": C0_DOC, "d0 Direct const": 0.0,
           "d1 Direct per stage": 0.0, "d2 Direct x save": 0.0}

    def design(m):
        return np.column_stack([(1 - sv)[m], ((1 - sv) * S)[m], sv[m],
                                (sv * nt)[m], (sv * nt * dr)[m], dr[m],
                                (dr * S)[m], (dr * sv)[m]])

    fdom = ols(design(dom), yR[dom], names)
    show(f"M2 pooled, R-units, formula-dominant subset (n={int(dom.sum())})", fdom, exp)
    pr = (design(dom) @ fdom["beta"]) * R[dom]
    print(f"    R^2 of this fit against BYTES = {r2_of(pr, peak[dom]):.6f}")
    rel(pr, peak[dom], "M2 pooled (dominant)")
    print(f"    -> wrapped in max(K,.) over ALL cells: R^2 R-units = "
          f"{r2_of(np.maximum(K, (design(np.ones_like(sv, bool)) @ fdom['beta']) * R) / R, yR):.6f}")

    allm = np.ones_like(sv, bool)
    fb = ols(design(dom) * R[dom, None], peak[dom], names)
    show("M2 pooled, BYTES target, dominant subset (unweighted OLS - "
         "biggest cells dominate)", fb, exp)
    fall = ols(design(allm), yR, names)
    show("M2 pooled, R-units, ALL cells (shown to expose what the K floor does "
         "to an unguarded fit)", fall, exp)

    # ---------- M3 per-arm ----------
    print("\n" + "=" * 104)
    print("D) M3 - PER-ARM REFITS on the formula-dominant subset")
    print("=" * 104)
    m0 = (sv == 0) & dom
    if m0.sum() > 5:
        X0 = np.column_stack([one[m0], S[m0], dr[m0], (dr * S)[m0], nt[m0]])
        show("(1) save_states=False:  peak/R = C0 + b*S + Direct(d0 + d1 S) "
             "+ (ntsave, must be 0)",
             ols(X0, yR[m0], ["C0 intercept", "per stage", "Direct const",
                              "Direct per stage", "per ntsave (must be ~0)"]),
             {"per stage": 1.0, "C0 intercept": C0_DOC,
              "per ntsave (must be ~0)": 0.0, "Direct const": 0.0,
              "Direct per stage": 0.0})
    m1m = (sv > 0) & dom
    if m1m.sum() > 5:
        X1 = np.column_stack([one[m1m], nt[m1m], (nt * dr)[m1m], S[m1m], dr[m1m],
                              (dr * S)[m1m]])
        show("(2) save_states=True:   peak/R = c0 + c1*ntsave + c2*ntsave*Direct "
             "+ stages + Direct(d0 + d1 S)",
             ols(X1, yR[m1m], ["intercept", "per saved state", "Direct per state",
                               "per stage", "Direct const", "Direct per stage"]),
             {"per saved state": 3.0, "Direct per state": 1.0, "intercept": 3.0,
              "per stage": 0.0, "Direct const": 0.0, "Direct per stage": 0.0})

    # ---------- M4: corrected formulas, coefficients LOCKED to measured values ----
    print("\n" + "=" * 104)
    print("F) M4 - THE CORRECTED FORMULAS, COEFFICIENTS LOCKED (no per-cell fitting)")
    print("=" * 104)
    C1 = 7.0666        # measured floor intercept (E1, rk3/rk5/rk8 agree to 1e-4 R)
    heun = (S == 2)    # Heun is the one tableau that carries no persistent stage cost
    Seff = np.where(heun, 0.0, S)
    floor4 = (C1 + Seff) * R
    floor4_d = (C1 + Seff + 2.0 + Seff) * R            # save=False + Direct
    ramp4 = 3.0 * R * (nt + 1.0)
    ramp4_d = 3.0 * R * nt + (C1 + S) * R              # save=True  + Direct
    f4 = np.where(sv > 0,
                  np.where(dr > 0, ramp4_d, ramp4),
                  np.where(dr > 0, floor4_d, floor4))
    m4 = np.maximum(K, f4)
    print("    save_states=False :  peak = max(K, (7.0666 + S_eff) R)                    "
          "[Direct: + (2 + S_eff) R]")
    print("    save_states=True  :  peak = max(K, 3 R (ntsave+1))                        "
          "[Direct: 3 R ntsave + (7.0666 + S) R]")
    print(f"    K = {K:,.0f} B = {K / MIB:.3f} MiB ;  S_eff = S except Heun (S=2), "
          "where S_eff = 0")
    print(f"    R^2 R-units, ALL {len(peak)} cells = {r2_of(m4 / R, yR):.7f}    "
          f"R^2 bytes = {r2_of(m4, peak):.7f}")
    rel(m4, peak, "M4 all cells")
    for N in Ns:
        m = Nv == N
        print(f"      N={N:>5d}  n={int(m.sum()):>4d}  R^2 R-units = "
              f"{r2_of(m4[m] / R[m], yR[m]):9.6f}   median |rel err| = "
              f"{np.median(np.abs(m4[m] / peak[m] - 1)):8.4%}   "
              f"max = {np.abs(m4[m] / peak[m] - 1).max():8.4%}")
    for lab, m in (("save_states=False", sv == 0), ("save_states=True", sv > 0),
                   ("gradient=none", dr == 0), ("gradient=Direct", dr > 0)):
        print(f"      {lab:<20s} n={int(m.sum()):>4d}  R^2 R-units = "
              f"{r2_of(m4[m] / R[m], yR[m]):9.6f}   median |rel err| = "
              f"{np.median(np.abs(m4[m] / peak[m] - 1)):8.4%}")
    print("\n    side-by-side, R^2 in R-units over all cells:")
    for lab, pr_ in (("M0 published, as written", pub),
                     ("M1 published + max(K,.)", m1),
                     ("M4 corrected, locked", m4)):
        print(f"      {lab:<28s} R^2 = {r2_of(pr_ / R, yR):10.7f}   "
              f"median |rel err| = {np.median(np.abs(pr_ / peak - 1)):9.4%}")

    # ---------- diagnostics ----------
    print("\n" + "=" * 104)
    print("E) DIAGNOSTICS")
    print("=" * 104)
    print("\n  E1. measured floor per method (save_states=False, gradient=none, "
          "formula-dominant N), in R:")
    pts = []
    for mth in ms:
        v = [r["peak"] / r["R"] for r in ok
             if r["method"] == mth and not r["save_states"] and not r["is_direct"]
             and (C0_DOC + r["n_stages"]) * r["R"] >= 3 * K]
        if v:
            st = stage_of[mth]
            pts.append((st, float(np.mean(v))))
            print(f"    {mth} (S={st:>2d})  mean={np.mean(v):8.4f} R  "
                  f"min={min(v):8.4f}  max={max(v):8.4f}  "
                  f"spread={max(v) - min(v):.4f} R  n={len(v):>3d}   "
                  f"C0_eff = mean - S = {np.mean(v) - st:+8.4f}")
    if len(pts) >= 3:
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        fE1 = ols(np.column_stack([np.ones_like(xs), xs]), ys, ["C0", "per stage"])
        f = fE1
        print(f"    affine fit over the {len(pts)} method means: "
              f"floor/R = {f['beta'][0]:.4f} + {f['beta'][1]:.4f} * S   "
              f"R^2 = {f['r2']:.6f}")
        print(f"    residuals per method (R): "
              f"{ {ms[i]: round(float(f['resid'][i]), 4) for i in range(len(pts))} }")
        keep = [i for i, pt in enumerate(pts) if pt[0] != 2]   # drop Heun
        if len(keep) >= 3:
            xs2 = np.array([pts[i][0] for i in keep], float)
            ys2 = np.array([pts[i][1] for i in keep], float)
            f2 = ols(np.column_stack([np.ones_like(xs2), xs2]), ys2, ["C0", "per stage"])
            fE1b = f2
            print(f"    SAME FIT EXCLUDING Heun (S=2), i.e. over S = "
                  f"{[int(v) for v in xs2]}:")
            print(f"      floor/R = {f2['beta'][0]:.4f} + {f2['beta'][1]:.6f} * S   "
                  f"R^2 = {f2['r2']:.8f}   residuals (R) = "
                  f"{[round(float(v), 6) for v in f2['resid']]}")
            print(f"      -> published C0 = {C0_DOC:.3f}, per stage = 1.000: "
                  f"dev {f2['beta'][0] - C0_DOC:+.4f} and {f2['beta'][1] - 1:+.6f}")
            print("      Heun sits at 7.005 R, i.e. 2 R BELOW this line - it behaves as")
            print("      if it carried zero persistent stage buffers (S_eff = 0).")

    print("\n  E2. flatness of save_states=False in ntsave (the core claim of formula 1):")
    for N in Ns:
        for mth in ("rk5",):
            v = [r["peak"] for r in ok if r["N"] == N and not r["save_states"]
                 and r["method"] == mth and not r["is_direct"]]
            if len(v) > 1:
                v = np.array(v, float)
                print(f"    N={N:>5d} {mth}  n={len(v):>3d}  spread = "
                      f"{v.max() - v.min():>13,.0f} B = "
                      f"{(v.max() - v.min()) / (16 * N * N):.6f} R   "
                      f"(mean {v.mean() / (16 * N * N):8.3f} R)")

    print("\n  E3. per-saved-state slope fitted INDEPENDENTLY in each (N, method, gradient)")
    print("      cell of the save_states=True arm [published: 3.000, 4.000 with Direct]")
    print("        N  method    grad    slope(R)  intercept(R)        R^2  npts")
    sl = []
    for N in Ns:
        for mth in ms:
            for g in ("none", "direct"):
                sub = [r for r in ok if r["N"] == N and r["method"] == mth
                       and r["save_states"] and r["gradient"] == g
                       and 3.0 * r["R"] * (r["ntsave"] + 1) >= 3 * K]
                if len(sub) < 3:
                    continue
                x = np.array([r["ntsave"] for r in sub], float)
                y = np.array([r["peak"] / r["R"] for r in sub], float)
                f = ols(np.column_stack([np.ones_like(x), x]), y, ["i", "s"])
                sl.append((N, mth, g, f["beta"][1], f["r2"]))
                print(f"    {N:>5d}  {mth:>6s}  {g:>6s}  {f['beta'][1]:9.4f}  "
                      f"{f['beta'][0]:11.3f}  {f['r2']:10.7f}  {len(x):>4d}")
    for g, want in (("none", 3.0), ("direct", 4.0)):
        v = [s[3] for s in sl if s[2] == g]
        if v:
            print(f"    -> {g:>6s}: slope mean={np.mean(v):.4f} sd={np.std(v):.4f} "
                  f"min={min(v):.4f} max={max(v):.4f}  over {len(v)} cells "
                  f"(published {want:.3f}, dev {np.mean(v) - want:+.4f})")
            print(f"       per-cell R^2: min={min(s[4] for s in sl if s[2] == g):.7f}")

    print("\n  E4. M2 residual (R-units, dominant subset) grouped by axis:")
    res = fdom["resid"]
    for key, lab in ((Nv[dom], "N"), (S[dom], "n_stages"), (nt[dom], "ntsave"),
                     (sv[dom], "save_states"), (dr[dom], "is_direct")):
        print(f"    by {lab}:")
        for v in sorted(set(key)):
            m = key == v
            print(f"      {lab}={v:<9.0f} n={int(m.sum()):>4d}  mean resid="
                  f"{res[m].mean():+10.4f} R   rms={np.sqrt((res[m] ** 2).mean()):9.4f} R")

    print("\n  E5. trajectory resident check: traj_bytes / (ntsave * R) "
          "[save_states=True] and /R [False]")
    tr = np.array([r["traj_in_R"] for r in ok])
    rt = np.where(sv > 0, tr / np.maximum(nt, 1), tr)
    print(f"    ratio over {len(rt)} cells: min={rt.min():.6f} max={rt.max():.6f} "
          f"mean={rt.mean():.6f}")

    print("\n  E7. the gradient=Direct penalty, isolated by PAIRING each Direct cell with")
    print("      its gradient=none twin (same N, method, save_states, ntsave).")
    print("      Published (Finding 5.C): +1 R per SAVED STATE -> grows with ntsave.")
    print("      Alternative: a CONSTANT penalty independent of ntsave.")
    idx = {(r["N"], r["method"], r["save_states"], r["ntsave"]): r for r in ok
           if not r["is_direct"]}
    pairs = []
    for r in ok:
        if not r["is_direct"]:
            continue
        t = idx.get((r["N"], r["method"], r["save_states"], r["ntsave"]))
        if t is not None:
            pairs.append((r, t, (r["peak"] - t["peak"]) / r["R"]))
    print(f"      {len(pairs)} pairs matched")
    for svflag, lab in ((True, "save_states=True"), (False, "save_states=False")):
        nmin = 512 if svflag else 1024   # the False arm needs a higher N to clear K
        sub = [p for p in pairs if p[0]["save_states"] is svflag
               and p[0]["N"] >= nmin]
        if not sub:
            continue
        d = np.array([p[2] for p in sub])
        x = np.array([p[0]["ntsave"] for p in sub], float)
        print(f"      -- {lab}, N>={nmin} (n={len(sub)}): delta mean={d.mean():+8.4f} R  "
              f"sd={d.std():7.4f}  min={d.min():+8.4f}  max={d.max():+8.4f}")
        if len(set(x)) > 2:
            f = ols(np.column_stack([np.ones_like(x), x]), d,
                    ["const (R)", "per saved state (R)"])
            print(f"         delta = {f['beta'][0]:+.4f} + {f['beta'][1]:+.6f} * ntsave"
                  f"   R^2={f['r2']:.6f}")
            print(f"         published slope 1.000 -> dev {f['beta'][1] - 1:+.6f};"
                  f"  constant-only model R^2 = "
                  f"{r2_of(np.full_like(d, d.mean()), d):.6f}")
        for mth in ms:
            dm = np.array([p[2] for p in sub if p[0]["method"] == mth])
            if len(dm):
                print(f"         {mth:>4s} (S={stage_of[mth]:>2d})  mean={dm.mean():+8.4f} R"
                      f"  sd={dm.std():7.4f}  n={len(dm)}")

    print(f"\n  E6. clean exit (bytes_in_use back to 0 after del+gc+clear_caches): "
          f"{sum(1 for r in recs if r.get('clean'))}/{len(recs)}")
    iu = [r["in_use_after"] / r["R"] for r in ok if "in_use_after" in r]
    print(f"      residual in_use after solve, in R: min={min(iu):.4f} "
          f"max={max(iu):.4f} median={np.median(iu):.4f}")

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        cols = dict(zip(Ns, plt.cm.viridis(np.linspace(0, .88, len(Ns)))))
        for N in Ns:
            for g, mk, ls in (("none", "o", "-"), ("direct", "^", "--")):
                sub = sorted([r for r in ok if r["N"] == N and r["save_states"]
                              and r["gradient"] == g and r["method"] == "rk5"],
                             key=lambda r: r["ntsave"])
                if sub:
                    ax[0].plot([r["ntsave"] for r in sub],
                               [r["peak"] / r["R"] for r in sub], mk, ls=ls,
                               color=cols[N], ms=4, lw=.9, label=f"N={N} {g}")
        x = np.linspace(1, 1000, 60)
        ax[0].plot(x, 3 * (x + 1), "k-", lw=1.6, alpha=.7, label="3(ntsave+1)")
        ax[0].plot(x, 4 * x + 3, "r-", lw=1.6, alpha=.7, label="4 ntsave+3 (Direct)")
        ax[0].set(xlabel="ntsave", ylabel="peak / R", xscale="log", yscale="log",
                  title="save_states=True (rk5): the 3 R ramp")
        ax[0].legend(fontsize=6, ncol=2)
        ax[0].grid(alpha=.3, which="both")

        for mth in ms:
            st = stage_of[mth]
            v = [r["peak"] / r["R"] for r in ok if r["method"] == mth
                 and not r["save_states"] and not r["is_direct"]
                 and (C0_DOC + st) * r["R"] >= 3 * K]
            if v:
                ax[1].scatter([st] * len(v), v, s=26, alpha=.75, label=f"{mth} S={st}")
        xs_ = np.linspace(0, 15, 10)
        ax[1].plot(xs_, C0_DOC + xs_, "k--", lw=1.4, label="published  7 + S")
        if "fE1b" in dir():
            ax[1].plot(xs_, fE1b["beta"][0] + fE1b["beta"][1] * xs_, "r-", lw=1.4,
                       label=(f"fit (S=4,7,14)  {fE1b['beta'][0]:.4f}"
                              f" + {fE1b['beta'][1]:.4f} S"))
        ax[1].annotate("Heun (S=2): 7.005 R,\n2 R below the line\n(S_eff = 0)",
                       xy=(2, 7.005), xytext=(3.4, 8.6), fontsize=7,
                       arrowprops=dict(arrowstyle="->", lw=.8))
        ax[1].set(xlabel="n_stages", ylabel="peak / R",
                  title="save_states=False, gradient=none: the floor vs stages")
        ax[1].legend(fontsize=7, loc="upper left")
        ax[1].grid(alpha=.3)

        ax[2].loglog(peak / GIB, pub / GIB, ".", ms=4, alpha=.45,
                     label=f"M0 published as written  R2={r2_of(pub / R, yR):.4f}")
        ax[2].loglog(peak / GIB, m4 / GIB, ".", ms=4, alpha=.6,
                     label=f"M4 corrected, locked     R2={r2_of(m4 / R, yR):.7f}")
        lim = [peak.min() / GIB * .6, peak.max() / GIB * 1.8]
        ax[2].loglog(lim, lim, "k--", lw=1)
        ax[2].set(xlabel="measured peak (GiB)", ylabel="predicted peak (GiB)",
                  title=f"parity, all {len(peak)} surviving cells (R^2 in R-units)",
                  xlim=lim, ylim=lim)
        ax[2].legend(fontsize=7)
        ax[2].grid(alpha=.3, which="both")
        fig.tight_layout()
        out = ("/srv/data/users/nlepage/new_dynamiqs/profiling_hunt/data/"
               "step6_grid.png")
        fig.savefig(out, dpi=130)
        print(f"\n  figure -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  (figure skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
