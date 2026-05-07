#!/usr/bin/env python3
"""OpenAI-compatible prefill (TTFT) microbenchmark.

Measures time-to-first-token across a sweep of prompt lengths. Each
request gets a unique random prefix so any prefix-cache (vLLM, sglang)
cannot fold the cost down — the second request onwards still pays the
real prefill price. Cap `--max-tokens` low (default 1) so the run is
dominated by prefill, not decode.

Usage:
    python scripts/bench_prefill.py \
        --base-url http://localhost:8765/v1 \
        --model Qwen/Qwen3-4B-AWQ \
        --prompt-lens 64 256 1024 2048 \
        --runs 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import statistics
import sys
import time
from typing import List

try:
    import httpx
except ImportError:
    sys.stderr.write("install httpx: pip install httpx\n")
    sys.exit(1)


async def measure_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
) -> float:
    """Return TTFT in milliseconds for one streamed request."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic; we only care about latency
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    ttft: float | None = None

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
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                # We have the first token — that's all the bench needs.
                # Drain the rest in the background.
                continue

    if ttft is None:
        ttft = time.perf_counter() - t0
    return ttft * 1000.0


def make_prompt(prompt_len: int) -> str:
    """Build a prompt of ~prompt_len words, prefixed with a fresh random
    nonce so prefix-cache hits cannot mask the prefill cost."""
    nonce = secrets.token_hex(16)  # 32 hex chars ≈ unique cache key
    body = (
        "Explain the following concept in depth, step by step, with a worked "
        "example and counter-examples where appropriate, then summarise the "
        "tradeoffs at the end. Concept: distributed systems consensus. "
    )
    words = (body * 50).split()
    target = max(prompt_len - 4, 4)  # leave room for nonce wrapper
    return f"[run-{nonce}] " + " ".join(words[:target])


async def sweep(
    base_url: str,
    model: str,
    prompt_lens: List[int],
    runs: int,
    max_tokens: int,
    warmup: bool,
    api_key: str | None,
) -> List[tuple[int, List[float]]]:
    timeout = httpx.Timeout(600.0, connect=10.0)
    out: List[tuple[int, List[float]]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for plen in prompt_lens:
            samples: List[float] = []
            total_runs = runs + (1 if warmup else 0)
            for i in range(total_runs):
                prompt = make_prompt(plen)
                ttft = await measure_one(
                    client, base_url, model, prompt, max_tokens, api_key
                )
                if warmup and i == 0:
                    continue
                samples.append(ttft)
            out.append((plen, samples))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAI-compatible prefill TTFT bench")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--prompt-lens", type=int, nargs="+",
                   default=[64, 256, 1024, 2048])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=1,
                   help="kept tiny so TTFT dominates the wall time")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the discard-first-run warmup")
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    label = args.label or args.base_url
    print(f"\n=== {label}  model={args.model} ===")
    print(f"max_tokens={args.max_tokens}  runs={args.runs}  "
          f"warmup={'no' if args.no_warmup else 'yes'}\n")
    print(f"{'prompt_len':>10}  {'p50 ms':>10}  {'p90 ms':>10}  "
          f"{'min ms':>10}  {'max ms':>10}  {'n':>3}")
    print("-" * 64)

    results = asyncio.run(
        sweep(
            args.base_url,
            args.model,
            args.prompt_lens,
            args.runs,
            args.max_tokens,
            warmup=not args.no_warmup,
            api_key=args.api_key,
        )
    )
    for plen, samples in results:
        if not samples:
            continue
        s = sorted(samples)
        p50 = statistics.median(s)
        p90 = s[max(0, int(0.9 * len(s)) - 1)]
        print(f"{plen:>10}  {p50:>10.1f}  {p90:>10.1f}  "
              f"{s[0]:>10.1f}  {s[-1]:>10.1f}  {len(s):>3}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
