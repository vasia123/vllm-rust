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
) -> BatchStats:
    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        coros = [
            stream_one(client, base_url, model, prompt, max_tokens, temperature, api_key)
            for _ in range(concurrency)
        ]
        runs = await asyncio.gather(*coros)
    return BatchStats(concurrency=concurrency, runs=list(runs))


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
    args = p.parse_args()

    prompt = make_prompt(args.prompt_len)
    label = args.label or args.base_url

    print(f"\n=== {label}  model={args.model} ===")
    print(f"prompt≈{args.prompt_len}w  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature}  runs={args.runs}\n")
    print(f"{'concurrency':>11}  {'med ttft ms':>12}  {'med tps/req':>12}  "
          f"{'aggregate tps':>14}  {'best agg tps':>14}")
    print("-" * 75)

    for c in args.concurrency:
        best_aggregate = 0.0
        last_stats: BatchStats | None = None
        # Throw away the first run as warmup if more than one is requested.
        warmup = args.runs > 1
        for run_idx in range(args.runs):
            stats = asyncio.run(
                run_concurrent(
                    args.base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    c,
                    args.api_key,
                )
            )
            if warmup and run_idx == 0:
                continue
            agg = stats.aggregate_tps()
            if agg > best_aggregate:
                best_aggregate = agg
                last_stats = stats
        if last_stats is None:
            continue
        print(
            f"{c:>11}  {last_stats.median_ttft_ms():>12.1f}  "
            f"{last_stats.median_tps():>12.1f}  "
            f"{last_stats.aggregate_tps():>14.1f}  "
            f"{best_aggregate:>14.1f}"
        )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
