"""Step 2 analysis: fidelity of memory_stats() and its precision as an OOM proxy.

Reads the sweep CSV, writes two figures and a text verdict.

  Fig 1  fidelity     - measured bytes_in_use and peak_bytes_in_use vs the
                        analytic prediction 8*n^2*4^k, plus the ratio panel.
  Fig 2  OOM proxy    - observed OOM points against the theoretical crossing
                        lines for the resident-only (H1) and input-held (H2)
                        hypotheses.
"""

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GIB = 1024**3
LIMIT = 45_285_900_288          # Step 1 measured, job 69371 config B
LIMIT_BRIEF = 45_890_509_210    # what the brief assumed

CSV_IN = sys.argv[1]
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else "step2"


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in ("n", "k", "side", "elements", "predicted_bytes",
                      "bytes_in_use", "peak_bytes_in_use", "bytes_limit",
                      "pool_bytes", "peak_pool_bytes", "largest_alloc_size",
                      "num_allocs", "largest_free_block_bytes", "repeat"):
                r[k] = int(r[k])
            for k in ("ratio_inuse", "ratio_peak", "elapsed_s"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def main() -> None:
    rows = load(CSV_IN)
    ok = [r for r in rows if r["status"] in ("ok", "seed")]
    oom = [r for r in rows if r["status"] == "OOM"]
    err = [r for r in rows if r["status"] == "ERR"]

    print("=" * 100)
    print(f"STEP 2 ANALYSIS  ({CSV_IN})")
    print("=" * 100)
    print(f"  rows total   : {len(rows)}")
    print(f"  successful   : {len(ok)}")
    print(f"  OOM          : {len(oom)}")
    print(f"  other errors : {len(err)}")
    print(f"  LIMIT (measured, Step 1) = {LIMIT:,d} B = {LIMIT / GIB:.4f} GiB")

    # ---------------------------------------------------------------- fidelity
    # Restrict to allocations big enough that the fixed 256 B overhead and
    # allocator granularity are negligible: >= 1 MiB.
    big = [r for r in ok if r["predicted_bytes"] >= 1 << 20]
    pred = np.array([r["predicted_bytes"] for r in big], dtype=float)
    inuse = np.array([r["bytes_in_use"] for r in big], dtype=float)
    peak = np.array([r["peak_bytes_in_use"] for r in big], dtype=float)
    r_in = inuse / pred
    r_pk = peak / pred

    print("\n" + "-" * 100)
    print("1) FIDELITY  (allocations >= 1 MiB, n = %d samples)" % len(big))
    print("-" * 100)
    print(f"  bytes_in_use / predicted      : mean={r_in.mean():.6f} "
          f"min={r_in.min():.6f} max={r_in.max():.6f} std={r_in.std():.2e}")
    print(f"  peak_bytes_in_use / predicted : mean={r_pk.mean():.6f} "
          f"min={r_pk.min():.6f} max={r_pk.max():.6f} std={r_pk.std():.2e}")
    off = inuse - pred
    print(f"  bytes_in_use - predicted      : min={off.min():+,.0f} B "
          f"max={off.max():+,.0f} B  (constant offset, not a factor)")
    uniq = sorted(set(int(x) for x in off))
    print(f"  distinct offsets observed     : {uniq[:10]}"
          f"{' ...' if len(uniq) > 10 else ''}")

    # affine fit  measured = a + b * predicted
    b_in, a_in = np.polyfit(pred, inuse, 1)
    b_pk, a_pk = np.polyfit(pred, peak, 1)
    print(f"\n  affine fit bytes_in_use      = {a_in:+,.1f} + {b_in:.9f} * predicted")
    print(f"  affine fit peak_bytes_in_use = {a_pk:+,.1f} + {b_pk:.9f} * predicted")

    # ------------------------------------------------------------- OOM proxy
    print("\n" + "-" * 100)
    print("2) PRECISION AS AN OOM PROXY")
    print("-" * 100)
    last_ok = {}
    for r in ok:
        if r["status"] == "seed":
            continue
        key = r["n"]
        if key not in last_ok or r["k"] > last_ok[key]["k"]:
            last_ok[key] = r
    oom_by_n = {}
    for r in oom:
        oom_by_n.setdefault(r["n"], r)

    def k_safe(n, limit, factor):
        k = -1
        while 8 * n * n * 4 ** (k + 1) * factor <= limit:
            k += 1
            if k > 25:
                break
        return k

    print(f"{'n':>3} {'last_ok_k':>9} {'resident GiB':>13} {'peak GiB':>10} "
          f"{'oom_k':>5} {'oom_pred GiB':>13} | {'H1 k':>4} {'H2 k':>4} | verdict")
    agree_h1 = agree_h2 = 0
    total = 0
    for n in sorted(last_ok):
        lo = last_ok[n]
        om = oom_by_n.get(n)
        h1 = k_safe(n, LIMIT, 1.0)
        h2 = k_safe(n, LIMIT, 1.25)
        total += 1
        agree_h1 += lo["k"] == h1
        agree_h2 += lo["k"] == h2
        verdict = []
        if lo["k"] == h2:
            verdict.append("H2 ok")
        if lo["k"] == h1:
            verdict.append("H1 ok")
        if not verdict:
            verdict.append("NEITHER")
        print(f"{n:>3} {lo['k']:>9} {lo['bytes_in_use'] / GIB:>13.4f} "
              f"{lo['peak_bytes_in_use'] / GIB:>10.4f} "
              f"{(om['k'] if om else -1):>5} "
              f"{(om['predicted_bytes'] / GIB if om else float('nan')):>13.4f} | "
              f"{h1:>4} {h2:>4} | {','.join(verdict)}")

    print(f"\n  configurations where the last successful k matches:")
    print(f"    H1 (resident only, factor 1.00) : {agree_h1}/{total}")
    print(f"    H2 (input held,    factor 1.25) : {agree_h2}/{total}")

    # how close does peak get to the limit at the last success, and how far
    # over would the OOM iteration have gone?
    print("\n  headroom analysis (winning metric = peak_bytes_in_use):")
    pk_last = np.array([last_ok[n]["peak_bytes_in_use"] for n in sorted(last_ok)],
                       dtype=float)
    print(f"    max peak_bytes_in_use at a surviving iteration : "
          f"{pk_last.max():,.0f} B = {pk_last.max() / GIB:.4f} GiB "
          f"({100 * pk_last.max() / LIMIT:.2f} % of limit)")
    over = []
    for n in sorted(oom_by_n):
        need = oom_by_n[n]["predicted_bytes"] * 1.25
        over.append(need / LIMIT)
    if over:
        over = np.array(over)
        print(f"    H2 requirement at the OOM iteration, as a fraction of limit: "
              f"min={over.min():.3f} (all must be > 1.0 for H2 to be consistent)")

    # ------------------------------------------------------------------ leak
    print("\n" + "-" * 100)
    print("3) LEAK CHECK")
    print("-" * 100)
    per_n = defaultdict(list)
    for r in rows:
        per_n[r["n"]].append(r["num_allocs"])
    finals = {n: v[-1] for n, v in per_n.items()}
    print(f"  num_allocs at the end of each ladder: "
          f"min={min(finals.values())} max={max(finals.values())}")
    print(f"  (a ladder of K iterations should add ~K allocations and no more)")

    # ------------------------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.loglog(pred / GIB, inuse / GIB, "o", ms=4, alpha=.7,
              label="measured bytes_in_use")
    ax.loglog(pred / GIB, peak / GIB, "s", ms=4, alpha=.5,
              label="measured peak_bytes_in_use")
    lo_, hi_ = pred.min() / GIB, pred.max() / GIB
    xs = np.array([lo_, hi_])
    ax.loglog(xs, xs, "k-", lw=1, label="y = predicted (ratio 1.00)")
    ax.loglog(xs, 1.25 * xs, "r--", lw=1, label="y = 1.25 x predicted (H2)")
    ax.axhline(LIMIT / GIB, color="purple", ls=":", lw=1.5,
               label=f"bytes_limit = {LIMIT / GIB:.2f} GiB")
    ax.set_xlabel("predicted bytes  $8n^2 4^k$  [GiB]")
    ax.set_ylabel("measured [GiB]")
    ax.set_title("Step 2 fidelity: measured vs analytic")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")

    ax = axes[1]
    ax.semilogx(pred / GIB, r_in, "o", ms=4, alpha=.7, label="bytes_in_use / predicted")
    ax.semilogx(pred / GIB, r_pk, "s", ms=4, alpha=.5,
                label="peak_bytes_in_use / predicted")
    ax.axhline(1.0, color="k", lw=1)
    ax.axhline(1.25, color="r", ls="--", lw=1)
    ax.set_ylim(0.9, 1.4)
    ax.set_xlabel("predicted bytes [GiB]")
    ax.set_ylabel("measured / predicted")
    ax.set_title("Step 2 fidelity: ratio (exactly 1.00 and 1.25)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    f1 = f"{OUT_PREFIX}_fidelity.png"
    fig.savefig(f1, dpi=130)
    print(f"\n  wrote {f1}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ns = sorted(last_ok)
    res_last = [last_ok[n]["bytes_in_use"] / GIB for n in ns]
    pk_last_l = [last_ok[n]["peak_bytes_in_use"] / GIB for n in ns]
    oom_pred = [oom_by_n[n]["predicted_bytes"] / GIB if n in oom_by_n else np.nan
                for n in ns]
    oom_peak_h2 = [oom_by_n[n]["predicted_bytes"] * 1.25 / GIB if n in oom_by_n
                   else np.nan for n in ns]
    ax.plot(ns, res_last, "o-", color="tab:blue", ms=5,
            label="resident at last SUCCESS (bytes_in_use)")
    ax.plot(ns, pk_last_l, "s-", color="tab:green", ms=5,
            label="peak at last SUCCESS (peak_bytes_in_use)")
    ax.plot(ns, oom_pred, "v", color="tab:orange", ms=7,
            label="predicted resident at the OOM iteration")
    ax.plot(ns, oom_peak_h2, "x", color="tab:red", ms=8,
            label="H2 requirement (1.25x) at the OOM iteration")
    ax.axhline(LIMIT / GIB, color="purple", lw=2,
               label=f"measured bytes_limit = {LIMIT / GIB:.3f} GiB")
    ax.axhline(LIMIT_BRIEF / GIB, color="grey", ls="--", lw=1.2,
               label=f"brief's assumed limit = {LIMIT_BRIEF / GIB:.3f} GiB")
    # The effective ceiling for the grow-while-holding pattern: bracketed below by
    # the largest peak that survived and above by the smallest that failed.
    surv_max = max(pk_last_l)
    fail_min = np.nanmin([v for v in oom_peak_h2 if not np.isnan(v)])
    ax.axhspan(surv_max, fail_min, color="red", alpha=.12,
               label=f"effective ceiling lies in [{surv_max:.2f}, {fail_min:.2f}] GiB")
    ax.axhline(surv_max, color="darkgreen", ls="-.", lw=1.2,
               label=f"largest surviving peak = {surv_max:.3f} GiB "
                     f"({100 * surv_max * GIB / LIMIT:.1f} % of limit)")
    ax.set_yscale("log")
    ax.set_xlabel("seed size n")
    ax.set_ylabel("GiB")
    ax.set_title("Step 2: memory_stats() as an OOM proxy — survival vs the ceiling")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    f2 = f"{OUT_PREFIX}_oom_proxy.png"
    fig.savefig(f2, dpi=130)
    print(f"  wrote {f2}")


if __name__ == "__main__":
    main()
