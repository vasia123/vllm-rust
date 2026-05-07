#!/usr/bin/env python3
"""OpenAI-compatible decode throughput benchmark.

Works against any OpenAI `/v1/chat/completions` server (vllm-rust, Python
vLLM, sglang, etc.). Sends N concurrent streaming requests, measures
generated-tokens-per-second per request and aggregate, prints a table.

Usage:
    python scripts/bench_decode.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-AWQ \
        --concurrency 1 4 8 \
        --prompt-len 256 \
        --max-tokens 512 \
        --runs 3

Compare two backends on the same machine by running twice with different
--base-url and --label.

When --admin-url is provided, the script also diffs the admin metrics
endpoint (`/admin/metrics`) before/after each concurrency level to surface
speculative-decode acceptance rate and token-bonus multiplier alongside
tok/s. This is the regression guard for Stage 13-I (spec-decode default-on).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List

try:
    import httpx
except ImportError:
    sys.stderr.write("install httpx: pip install httpx\n")
    sys.exit(1)


@dataclass
class RunStats:
    """One end-to-end stream measurement."""

    ttft_ms: float  # time-to-first-token (decode prefill latency)
    total_s: float  # full stream duration
    generated_tokens: int  # tokens after first
    decode_tps: float  # generated_tokens / (total_s - ttft)
    finish_reason: str


@dataclass
class BatchStats:
    """Aggregate over `concurrency` parallel runs."""

    concurrency: int
    runs: List[RunStats] = field(default_factory=list)

    def aggregate_tps(self) -> float:
        """Sum of per-stream decode tps — what the system serves overall."""
        return sum(r.decode_tps for r in self.runs)

    def median_tps(self) -> float:
        return statistics.median(r.decode_tps for r in self.runs) if self.runs else 0.0

    def median_ttft_ms(self) -> float:
        return statistics.median(r.ttft_ms for r in self.runs) if self.runs else 0.0


@dataclass
class SpecSnapshot:
    """Lifetime monotonic counters from /admin/metrics."""

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int

    def diff(self, before: "SpecSnapshot") -> tuple[int, int, int]:
        return (
            self.num_drafts - before.num_drafts,
            self.num_draft_tokens - before.num_draft_tokens,
            self.num_accepted_tokens - before.num_accepted_tokens,
        )


async def fetch_spec_snapshot(
    client: httpx.AsyncClient, admin_url: str
) -> SpecSnapshot | None:
    """Fetch a snapshot of speculative counters; None when spec is off."""
    try:
        resp = await client.get(f"{admin_url}/metrics")
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    body = resp.json()
    spec = body.get("spec_decode")
    if not spec:
        return None
    return SpecSnapshot(
        num_drafts=int(spec["num_drafts"]),
        num_draft_tokens=int(spec["num_draft_tokens"]),
        num_accepted_tokens=int(spec["num_accepted_tokens"]),
    )


async def stream_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str | None,
) -> RunStats:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    ttft: float | None = None
    generated = 0
    finish_reason = "unknown"

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
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

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
        finish_reason=finish_reason,
    )


async def run_concurrent(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    api_key: str | None,
    admin_url: str | None = None,
) -> tuple[BatchStats, tuple[int, int, int] | None]:
    """Returns batch stats plus an optional (drafts, draft_toks, accepted)
    delta when an admin URL is provided and spec decode is active."""
    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        before = (
            await fetch_spec_snapshot(client, admin_url) if admin_url else None
        )
        coros = [
            stream_one(client, base_url, model, prompt, max_tokens, temperature, api_key)
            for _ in range(concurrency)
        ]
        runs = await asyncio.gather(*coros)
        spec_delta: tuple[int, int, int] | None = None
        if admin_url and before is not None:
            after = await fetch_spec_snapshot(client, admin_url)
            if after is not None:
                spec_delta = after.diff(before)
    return BatchStats(concurrency=concurrency, runs=list(runs)), spec_delta


def make_prompt(prompt_len: int) -> str:
    """Generate a realistic user-style prompt of ~`prompt_len` words."""
    base = (
        "Explain the following concept in depth, step by step, with a worked "
        "example and counter-examples where appropriate, then summarise the "
        "tradeoffs at the end. Concept: distributed systems consensus. "
    )
    text = (base * 20).split()
    return " ".join(text[:max(prompt_len, 8)])


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI-compatible decode bench")
    p.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1")
    p.add_argument("--model", required=True, help="model name as the server sees it")
    p.add_argument("--label", default="", help="label printed alongside results")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--prompt-len", type=int, default=256, help="approx words")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--runs", type=int, default=3, help="repeats per concurrency")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--admin-url",
        default=None,
        help="e.g. http://localhost:8000/admin — when set, prints "
        "speculative-decode acceptance rate alongside tok/s",
    )
    args = p.parse_args()

    prompt = make_prompt(args.prompt_len)
    label = args.label or args.base_url

    print(f"\n=== {label}  model={args.model} ===")
    print(f"prompt≈{args.prompt_len}w  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature}  runs={args.runs}\n")
    header = (
        f"{'concurrency':>11}  {'med ttft ms':>12}  {'med tps/req':>12}  "
        f"{'aggregate tps':>14}  {'best agg tps':>14}"
    )
    if args.admin_url:
        header += f"  {'accept rate':>11}  {'tok/draft':>9}"
    print(header)
    print("-" * len(header))

    for c in args.concurrency:
        best_aggregate = 0.0
        last_stats: BatchStats | None = None
        last_spec_delta: tuple[int, int, int] | None = None
        # Throw away the first run as warmup if more than one is requested.
        warmup = args.runs > 1
        for run_idx in range(args.runs):
            stats, spec_delta = asyncio.run(
                run_concurrent(
                    args.base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    c,
                    args.api_key,
                    args.admin_url,
                )
            )
            if warmup and run_idx == 0:
                continue
            agg = stats.aggregate_tps()
            if agg > best_aggregate:
                best_aggregate = agg
                last_stats = stats
                last_spec_delta = spec_delta
        if last_stats is None:
            continue
        line = (
            f"{c:>11}  {last_stats.median_ttft_ms():>12.1f}  "
            f"{last_stats.median_tps():>12.1f}  "
            f"{last_stats.aggregate_tps():>14.1f}  "
            f"{best_aggregate:>14.1f}"
        )
        if args.admin_url:
            if last_spec_delta and last_spec_delta[1] > 0:
                drafts, dtoks, accepted = last_spec_delta
                rate = accepted / dtoks
                # Tokens emitted per draft round, including the always-
                # committed bonus token (1 + accepted/drafts).
                tok_per_draft = 1.0 + accepted / drafts if drafts else 0.0
                line += f"  {rate:>10.1%}  {tok_per_draft:>9.2f}"
            else:
                line += f"  {'n/a':>11}  {'n/a':>9}"
        print(line)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
