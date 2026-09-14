"""Emit the Step 6 grid, one config per line: N ntsave save_states method gradient.

Grid (960 cells):
  save_states  : 0, 1
  N            : 64 128 256 512 1024 2048          (powers of 2, R = 16 N^2)
  method       : rk2 rk3 rk5 rk8                   (adaptive only, 2/4/7/14 stages)
  ntsave       : linspace(10, 1000, 10) -> 10 ... 1000 uniformly, 10 datapoints
  gradient     : none, direct

Emitted in an order that interleaves N, so that sharding by (line index % nshards)
gives every shard a comparable mix of cheap and expensive cells.
"""

import sys

import numpy as np

NS = [64, 128, 256, 512, 1024, 2048]
METHODS = ["rk2", "rk3", "rk5", "rk8"]
NTSAVES = [int(x) for x in np.linspace(10, 1000, 10)]
GRADS = ["none", "direct"]
SAVES = [0, 1]

rows = []
for nt in NTSAVES:            # outer: ntsave
    for N in NS:              # then N, so consecutive lines differ in N
        for m in METHODS:
            for g in GRADS:
                for s in SAVES:
                    rows.append((N, nt, s, m, g))

if __name__ == "__main__":
    if len(sys.argv) == 3:    # shard: <shard_id> <nshards>
        sid, nsh = int(sys.argv[1]), int(sys.argv[2])
        rows = [r for i, r in enumerate(rows) if i % nsh == sid]
    for r in rows:
        print(f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]}")
