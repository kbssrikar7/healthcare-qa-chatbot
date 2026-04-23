#!/usr/bin/env python3
"""Print timing for POST /ask against a running API (profiling / dashboards)."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument(
        "--question",
        default="What are common symptoms of type 2 diabetes?",
        help="Question body",
    )
    args = p.parse_args()
    base = args.base.rstrip("/")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("API_KEY", "")
    if key:
        headers["X-API-Key"] = key
    body = json.dumps({"question": args.question}).encode()
    req = urllib.request.Request(base + "/ask", data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read().decode()
    elapsed = time.perf_counter() - t0
    data = json.loads(raw)
    print(f"wall_s={elapsed:.3f} api_latency_ms={data.get('latency_ms')}")
    st = data.get("stage_latencies")
    if isinstance(st, dict):
        for k, v in sorted(st.items(), key=lambda kv: (-float(kv[1]), kv[0])):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.1f} ms")


if __name__ == "__main__":
    main()
