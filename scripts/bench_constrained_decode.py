#!/usr/bin/env python3
"""Constrained vs unconstrained decode throughput benchmark.

Measures the per-request decode throughput penalty for JSON-Schema
structured output. Drives a vllm-rust server through
`/v1/chat/completions`, alternating identical greedy requests with
and without a `response_format=json_schema` body.

Usage:
    python scripts/bench_constrained_decode.py \\
        --base-url http://localhost:8111/v1 \\
        --model turboderp/Qwen3-8B-exl3 \\
        --runs 5 \\
        --max-tokens 80 \\
        --concurrency 1 4

Output JSON to stdout:
    {
      "label": "...",
      "model": "...",
      "results": [
        { "concurrency": 1,
          "unconstrained_med_tps": ..., "unconstrained_p99_tps": ...,
          "constrained_med_tps":   ..., "constrained_p99_tps":   ...,
          "slowdown_ratio": (unconstrained / constrained) },
        ...
      ]
    }

Greedy (temperature=0) so the only between-run variance is system-level
(kernel-launch jitter, GPU clock). One warmup run per (concurrency,
mode) is discarded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, List

try:
    import httpx
except ImportError:
    sys.stderr.write("install httpx: pip install httpx\n")
    sys.exit(1)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["foo", "bar"],
    "properties": {
        "foo": {"type": "integer", "minimum": 1, "maximum": 1000},
        "bar": {"type": "string", "minLength": 3, "maxLength": 50},
    },
}

PROMPT = (
    "Output a JSON object with two keys: foo (integer 1..1000) and "
    "bar (a short string). Reply with the JSON object only."
)


@dataclass
class RunStats:
    ttft_ms: float
    total_s: float
    generated_tokens: int
    decode_tps: float


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    max_tokens: int,
    constrained: bool,
    api_key: str | None,
) -> RunStats:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # Disable Qwen3 "thinking" preamble so token counts are
        # comparable across constrained/unconstrained runs.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if constrained:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "S", "strict": True, "schema": SCHEMA},
        }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    ttft: float | None = None
    generated = 0
    async with client.stream(
        "POST", f"{base_url}/chat/completions", json=body, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[6:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                generated += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    decode_window = max(total - ttft, 1e-6)
    decode_tps = max(generated - 1, 0) / decode_window
    return RunStats(
        ttft_ms=ttft * 1000.0,
        total_s=total,
        generated_tokens=generated,
        decode_tps=decode_tps,
    )


async def run_batch(
    base_url: str,
    model: str,
    max_tokens: int,
    constrained: bool,
    concurrency: int,
    api_key: str | None,
) -> List[RunStats]:
    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        coros = [
            one_request(client, base_url, model, max_tokens, constrained, api_key)
            for _ in range(concurrency)
        ]
        return list(await asyncio.gather(*coros))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = max(0, min(len(sorted_v) - 1, int(round((pct / 100.0) * (len(sorted_v) - 1)))))
    return sorted_v[k]


def aggregate(stats_per_run: List[List[RunStats]]) -> dict[str, float]:
    """Reduce N runs × C concurrent samples to representative numbers.

    `med_tps_per_req` — median over all (run, request) pairs.
    `p99_tps_per_req` — p99 over the same pool.
    `agg_tps`        — sum over a single request batch (best run).
    """
    all_tps: List[float] = []
    best_aggregate = 0.0
    for batch in stats_per_run:
        agg = sum(r.decode_tps for r in batch)
        if agg > best_aggregate:
            best_aggregate = agg
        all_tps.extend(r.decode_tps for r in batch)
    return {
        "med_tps_per_req": statistics.median(all_tps) if all_tps else 0.0,
        "p99_tps_per_req": percentile(all_tps, 99.0),
        "best_agg_tps": best_aggregate,
        "num_samples": len(all_tps),
    }


async def measure(
    base_url: str,
    model: str,
    max_tokens: int,
    constrained: bool,
    concurrency: int,
    runs: int,
    api_key: str | None,
) -> dict[str, float]:
    # Warmup (discarded) — engine cold-cache, AWQ marlin shape jit, etc.
    await run_batch(base_url, model, max_tokens, constrained, concurrency, api_key)
    batches = []
    for _ in range(runs):
        batches.append(
            await run_batch(base_url, model, max_tokens, constrained, concurrency, api_key)
        )
    return aggregate(batches)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8111/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--label", default="", help="label embedded in JSON output")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--pretty",
        action="store_true",
        help="also print a human-readable table to stderr",
    )
    args = p.parse_args()

    label = args.label or args.base_url
    out: dict[str, Any] = {
        "label": label,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "results": [],
    }

    if args.pretty:
        print(f"\n=== {label}  model={args.model} ===", file=sys.stderr)
        header = (
            f"{'c':>3}  {'mode':>13}  {'med tps/req':>12}  "
            f"{'p99 tps/req':>12}  {'best agg':>10}  {'samples':>8}"
        )
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)

    for c in args.concurrency:
        unc = asyncio.run(
            measure(args.base_url, args.model, args.max_tokens, False, c, args.runs, args.api_key)
        )
        con = asyncio.run(
            measure(args.base_url, args.model, args.max_tokens, True, c, args.runs, args.api_key)
        )
        ratio = (unc["med_tps_per_req"] / con["med_tps_per_req"]
                 if con["med_tps_per_req"] > 0 else float("inf"))
        out["results"].append({
            "concurrency": c,
            "unconstrained": unc,
            "constrained": con,
            "slowdown_ratio_unc_over_con": ratio,
            "slowdown_pct": (ratio - 1.0) * 100.0,
        })
        if args.pretty:
            for mode, m in (("unconstrained", unc), ("constrained", con)):
                print(
                    f"{c:>3}  {mode:>13}  {m['med_tps_per_req']:>12.2f}  "
                    f"{m['p99_tps_per_req']:>12.2f}  {m['best_agg_tps']:>10.2f}  "
                    f"{m['num_samples']:>8d}",
                    file=sys.stderr,
                )
            print(
                f"     slowdown = {ratio:.2f}x  ({(ratio - 1.0) * 100.0:+.1f}%)",
                file=sys.stderr,
            )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
