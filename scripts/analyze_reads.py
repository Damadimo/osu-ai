"""How well did a model read the approach rings? Summarize a live run log.

  uv run scripts/analyze_reads.py runs/<time>-live-Jev-75.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from osu_ai.pilot import BUCKET_MS, HORIZON_MS


def main(path: str) -> None:
    log = json.load(open(path))
    reads = [r for r in log["reads"] if r["said"] != "spinner"]
    near = [r for r in reads if 0 <= r["truth_ms"] < HORIZON_MS]
    far = [r for r in reads if r["truth_ms"] >= HORIZON_MS]
    c, req = log["counts"], log["requests"]
    print(f"{log['pilot']} on {log['map']}")
    print(f"  accuracy {log['accuracy']:.1%}  (300:{c.get('300', 0)} 100:{c.get('100', 0)} "
          f"50:{c.get('50', 0)} miss:{c.get('0', 0)})")
    print(f"  requests answered {req['ok']}/{req['sent']}  refused/failed: {req['errors']}")
    lat = sorted(log["latencies_ms"])
    if lat:
        print(f"  latency {statistics.median(lat):.0f} ms typical, {lat[int(0.95 * (len(lat) - 1))]:.0f} ms p95")
    print(f"  far reads (ring closes in >{HORIZON_MS} ms): {sum(r['right'] for r in far)}/{len(far)} right")
    print(f"  near reads (these become clicks): {sum(r['right'] for r in near)}/{len(near)} right")
    by_bin = defaultdict(list)
    for r in near:
        said = HORIZON_MS + BUCKET_MS / 2 if r["said"] == "later" else int(r["said"].split("-")[0]) + BUCKET_MS / 2
        by_bin[int(r["truth_ms"] // 100) * 100].append(said)
    print("  real time left -> what it said (median bucket middle)")
    for lo in sorted(by_bin):
        said = by_bin[lo]
        print(f"    {lo:3d}-{lo + 100:3d} ms -> {statistics.median(said):4.0f} ms   (n={len(said)})")


if __name__ == "__main__":
    main(sys.argv[1])
