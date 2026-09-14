"""Step 6 addendum (6.8) - 4-way multidimensional fit, filtered by
(save_states, gradient), each subset fit on ALL 8 interaction terms in
R, ntsave, n_stages jointly:

    peak(bytes) = a0 + a1 R + a2 nt + a3 S + a4 R.nt + a5 R.S + a6 nt.S + a7 R.nt.S

Two contaminations are removed before the fit is trusted (both reported, not
hidden): the max(K, formula) floor (Finding 6.A) - restrict to peak >= 3K; and
the Heun (rk2) tableau's S_eff=0 anomaly (Finding 6.C) - excluded by method name.
A third, smaller effect (an ntsave-linked transient specific to
save_states=False + gradient=Direct, decaying by ntsave~230) is handled by
restricting that one subset to ntsave >= 230.

"Does this term matter" is judged by CONTRIBUTION RANGE
(coefficient * (max-min) of that column, in bytes) rather than by the raw OLS
standard error, which is numerically degenerate once R^2 -> 1 on a design whose
columns span many orders of magnitude (see the writeup in profiling_hunt.md 6.8
for why).
"""

import glob
import json

import numpy as np

MIB = 1024**2
K = 50_404_608.0
TERMS = ["a0 const", "a1 R", "a2 ntsave", "a3 S", "a4 R.ntsave",
         "a5 R.S", "a6 ntsave.S", "a7 R.ntsave.S"]


def load():
    recs = []
    for p in glob.glob("profiling_hunt/data/step6_69401_*.jsonl"):
        for line in open(p):
            if line.strip():
                recs.append(json.loads(line))
    return [r for r in recs if r["status"] == "ok"]


def fit_report(sub, tag):
    R = np.array([r["R"] for r in sub], float)
    S = np.array([r["n_stages"] for r in sub], float)
    nt = np.array([r["ntsave"] for r in sub], float)
    y = np.array([r["peak"] for r in sub], float)
    X = np.column_stack([np.ones_like(R), R, nt, S, R * nt, R * S, nt * S, R * nt * S])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    span = y.max() - y.min()
    print(f"\n{tag}  n={len(sub)}  R^2={r2:.7f}  mean(peak)={y.mean() / MIB:.2f} MiB")
    for j, name in enumerate(TERMS):
        col = X[:, j]
        contrib = beta[j] * (col.max() - col.min())
        frac = contrib / span if span > 0 else 0.0
        print(f"   {name:<12s} = {beta[j]:>14.6g}   contribution = {contrib:>12,.4g} B"
              f"  ({frac:>9.4%} of span)")
    return beta, r2


def main():
    ok = load()
    sub1 = [r for r in ok if not r["save_states"] and not r["is_direct"]
            and r["method"] != "rk2" and r["peak"] >= 3 * K]
    fit_report(sub1, "save=False grad=none")

    sub2 = [r for r in ok if r["save_states"] and not r["is_direct"]
            and r["method"] != "rk2" and r["peak"] >= 3 * K]
    fit_report(sub2, "save=True grad=none")

    sub3 = [r for r in ok if not r["save_states"] and r["is_direct"]
            and r["method"] != "rk2" and r["peak"] >= 3 * K and r["ntsave"] >= 230]
    fit_report(sub3, "save=False grad=Direct (ntsave>=230, past the transient)")

    sub4 = [r for r in ok if r["save_states"] and r["is_direct"]
            and r["method"] != "rk2" and r["peak"] >= 3 * K]
    fit_report(sub4, "save=True grad=Direct")


if __name__ == "__main__":
    main()
