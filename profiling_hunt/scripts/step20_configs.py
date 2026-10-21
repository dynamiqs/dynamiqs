"""Emit the Step 20 grid, one config per line:
    study N ntsave save_states n_obs obs_layout

Study A - save_states=True, no exp_ops, 5 N x 5 ntsave = 25 cells.
    N sized so every cell sits well above the ~48 MiB floor K (smallest cell
    3 R (ntsave+1) = 252 MiB) and the largest stays far from the ceiling (19.5 GiB).

Study B - save_states=False, 5 N x 5 ntsave x (n_obs=0 + 5 n_obs x 2 layouts)
    = 275 cells. N is larger than in A because the save_states=False working set is
    only ~14 R: at N=1024 that is 224 MiB (4.7 K), at N=3072 about 2 GiB before any
    observable is added.

Plus floor controls at N=16 (R = 4 KiB), where every formula is far below K, so the
measured peak IS the floor and K is re-measured in this very job.

Emitted in an order that interleaves sizes, so that sharding by
(line index % nshards) gives every shard a comparable mix of cheap and expensive cells.
"""

import sys

NTSAVES = [20, 60, 100, 140, 180]
NS_A = [512, 768, 1024, 1280, 1536]
NS_B = [1024, 1536, 2048, 2560, 3072]
NOBS_B = [1, 2, 4, 8, 16]
LAYOUTS_B = ["dense", "dia"]

rows = [
    ("A", 16, 20, 1, 0, "none"),
    ("B", 16, 20, 0, 0, "none"),
    ("B", 16, 20, 0, 16, "dense"),
]
for nt in NTSAVES:
    for i in range(len(NS_A)):
        rows.append(("A", NS_A[i], nt, 1, 0, "none"))
        rows.append(("B", NS_B[i], nt, 0, 0, "none"))
        for n_obs in NOBS_B:
            for layout in LAYOUTS_B:
                rows.append(("B", NS_B[i], nt, 0, n_obs, layout))

if __name__ == "__main__":
    if len(sys.argv) == 3:  # shard: <shard_id> <nshards>
        sid, nsh = int(sys.argv[1]), int(sys.argv[2])
        rows = [r for i, r in enumerate(rows) if i % nsh == sid]
    for r in rows:
        print(" ".join(str(x) for x in r))
