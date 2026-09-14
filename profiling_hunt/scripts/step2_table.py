"""Regenerate the Step 2 crossing table against the MEASURED ceiling from Step 1.

The brief's table was built on LIMIT_BYTES = 45_890_509_210 (0.95 x physical).
Step 1 measured 45_285_900_288 (0.95 x FREE). This script rebuilds the table and
also emits the transient-aware predictions that the kron test case actually implies.

predicted_bytes(n, k) = 8 * n^2 * 4^k        # float64, 8 bytes/element
"""

GIB = 1024**3
LIMIT_MEASURED = 45_285_900_288   # Step 1, job 69371 config B
LIMIT_BRIEF = 45_890_509_210      # what the brief assumed
K_MAX = 20


def predicted_bytes(n: int, k: int) -> int:
    return 8 * n * n * 4**k


def k_safe(n: int, limit: int, factor: float = 1.0) -> int:
    """Largest k whose *peak* requirement (factor x resident) still fits."""
    k = -1
    while predicted_bytes(n, k + 1) * factor <= limit:
        k += 1
        if k >= K_MAX:
            break
    return k


def main() -> None:
    print("=" * 100)
    print("STEP 2 CROSSING TABLE - regenerated against MEASURED LIMIT_BYTES")
    print("=" * 100)
    print(f"  LIMIT measured (Step 1) = {LIMIT_MEASURED:,d} B = "
          f"{LIMIT_MEASURED / GIB:.4f} GiB")
    print(f"  LIMIT in brief          = {LIMIT_BRIEF:,d} B = "
          f"{LIMIT_BRIEF / GIB:.4f} GiB")
    print(f"  brief is too high by      {LIMIT_BRIEF - LIMIT_MEASURED:,d} B = "
          f"{(LIMIT_BRIEF - LIMIT_MEASURED) / 1024**2:.1f} MiB\n")

    # --- resident-only table (factor 1.0): directly comparable to the brief -----
    print("-" * 100)
    print("A) RESIDENT-ONLY crossing (factor 1.0) - assumes only the result is live")
    print("-" * 100)
    print(f"{'n':>3} {'k_safe':>7} {'GiB@safe':>10} {'k_oom':>6} {'GiB@oom':>10} "
          f"{'brief_ksafe':>12} {'shifted?':>9}")
    brief_shift = []
    for n in range(2, 41):
        ks = k_safe(n, LIMIT_MEASURED)
        kb = k_safe(n, LIMIT_BRIEF)
        shifted = "LOST ONE" if ks < kb else ""
        if ks < kb:
            brief_shift.append(n)
        print(f"{n:>3} {ks:>7} {predicted_bytes(n, ks) / GIB:>10.3f} {ks + 1:>6} "
              f"{predicted_bytes(n, ks + 1) / GIB:>10.3f} {kb:>12} {shifted:>9}")
    print(f"\n  n values that lose an iteration vs the brief's table: {brief_shift}")

    # --- transient-aware table -------------------------------------------------
    # jnp.kron(b, A_step): the input b (1/4 of the output) is a live argument while
    # the output is being written, so the transient high-water mark is 1.25x the
    # result. If the trailing reshape were a separate materialising copy it would
    # be 2.25x. Both hypotheses are tabulated so the run can discriminate.
    print()
    print("-" * 100)
    print("B) TRANSIENT-AWARE crossing - what the kron test case actually implies")
    print("-" * 100)
    print("  H1 factor 1.00 : only the result is ever live (no input held)")
    print("  H2 factor 1.25 : input b (=result/4) held while result is written")
    print("  H3 factor 2.25 : as H2 plus a materialising reshape copy of the result")
    print()
    print(f"{'n':>3} | {'H1 ksafe':>8} {'GiB':>9} | {'H2 ksafe':>8} {'GiB':>9} | "
          f"{'H3 ksafe':>8} {'GiB':>9}")
    for n in range(2, 41):
        row = [f"{n:>3} |"]
        for fac in (1.0, 1.25, 2.25):
            ks = k_safe(n, LIMIT_MEASURED, fac)
            row.append(f"{ks:>8} {predicted_bytes(n, ks) / GIB:>9.3f} |")
        print(" ".join(row))

    print()
    print("-" * 100)
    print("C) DISCRIMINATING CONFIGURATIONS")
    print("-" * 100)
    print("  n where H1 and H2 predict a different OOM iteration (these decide it):")
    disc = [n for n in range(2, 41)
            if k_safe(n, LIMIT_MEASURED, 1.0) != k_safe(n, LIMIT_MEASURED, 1.25)]
    for n in disc:
        k1 = k_safe(n, LIMIT_MEASURED, 1.0)
        print(f"    n={n:>3}  k={k1}: resident {predicted_bytes(n, k1) / GIB:7.3f} GiB "
              f"fits under H1, needs {predicted_bytes(n, k1) * 1.25 / GIB:7.3f} GiB "
              f"under H2 -> OOM")
    print(f"\n  full list: {disc}")

    print()
    print("-" * 100)
    print("D) THE BRIEF'S TWO SPECIAL PROBES, re-evaluated at the measured ceiling")
    print("-" * 100)
    for n, k in ((37, 11), (36, 11)):
        p = predicted_bytes(n, k)
        print(f"  n={n}, k={k}: side={n * 2**k}, elements={n * n * 4**k:,d}")
        print(f"    resident      {p:>18,d} B = {p / GIB:8.4f} GiB")
        print(f"    vs measured limit: {p - LIMIT_MEASURED:+,d} B "
              f"({100 * (p - LIMIT_MEASURED) / LIMIT_MEASURED:+.3f} %)  "
              f"-> {'OVER' if p > LIMIT_MEASURED else 'fits'}")
        print(f"    vs brief limit   : {p - LIMIT_BRIEF:+,d} B "
              f"({100 * (p - LIMIT_BRIEF) / LIMIT_BRIEF:+.3f} %)  "
              f"-> {'OVER' if p > LIMIT_BRIEF else 'fits'}")
        print(f"    transient H2 (1.25x): {p * 1.25 / GIB:8.4f} GiB -> "
              f"{'OVER' if p * 1.25 > LIMIT_MEASURED else 'fits'}")
        print()

    print("  Largest resident that fits under the measured ceiling, by hypothesis:")
    for fac, name in ((1.0, "H1"), (1.25, "H2"), (2.25, "H3")):
        best = max(((n, k_safe(n, LIMIT_MEASURED, fac)) for n in range(2, 41)),
                   key=lambda nk: predicted_bytes(*nk))
        print(f"    {name} (factor {fac}): n={best[0]:>3} k={best[1]:>2} -> "
              f"{predicted_bytes(*best) / GIB:8.4f} GiB resident")


if __name__ == "__main__":
    main()
