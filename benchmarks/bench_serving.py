#!/usr/bin/env python3
"""Benchmark client for LLM serving engines (vllm-rust, Python vLLM).

Measures TTFT, token throughput, end-to-end latency, and inter-token latency
by parsing SSE streams from OpenAI-compatible /v1/completions endpoints.

Dependencies: aiohttp (pip install aiohttp)
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp


@dataclass
class RequestResult:
    ttft: float  # seconds
    total_time: float  # seconds
    token_count: int
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Tokens per second."""
        if self.total_time <= 0:
            return 0.0
        return self.token_count / self.total_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Time between consecutive tokens (seconds)."""
        if len(self.token_timestamps) < 2:
            return []
        return [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]


@dataclass
class ScenarioStats:
    name: str
    num_requests: int
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    ttft_mean: float
    throughput_mean: float  # tok/s per request
    throughput_total: float  # tok/s aggregate
    latency_p50: float  # ms
    latency_p95: float  # ms
    latency_p99: float  # ms
    latency_mean: float  # ms
    itl_p50: float  # ms
    itl_p95: float  # ms
    itl_mean: float  # ms
    total_tokens: int
    total_time: float  # wall-clock for entire scenario


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


async def bench_request_streaming(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send one streaming request, parse SSE, record timestamps."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }

    token_timestamps: list[float] = []
    t_first: Optional[float] = None
    t_start = time.perf_counter()

    async with session.post(
        f"{base_url}/v1/completions",
        json=payload,
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

        buffer = ""
        async for chunk in resp.content.iter_any():
            now = time.perf_counter()
            buffer += chunk.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                text = choice.get("text", "")
                finish_reason = choice.get("finish_reason")

                # A token event has non-empty text and no finish_reason
                if text and finish_reason is None:
                    if t_first is None:
                        t_first = now
                    token_timestamps.append(now)
                # Finish event (text="" with finish_reason set) — end of stream
                elif finish_reason is not None:
                    pass

    t_end = time.perf_counter()

    if t_first is None:
        t_first = t_end

    return RequestResult(
        ttft=t_first - t_start,
        total_time=t_end - t_start,
        token_count=len(token_timestamps),
        token_timestamps=token_timestamps,
    )


async def run_scenario(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    warmup: int,
) -> ScenarioStats:
    """Run warmup + measured requests at given concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def run_one(prompt: str) -> RequestResult:
        async with sem:
            async with aiohttp.ClientSession() as session:
                return await bench_request_streaming(
                    session, base_url, model, prompt, max_tokens
                )

    # Warmup
    if warmup > 0:
        warmup_tasks = []
        for i in range(warmup):
            prompt = prompts[i % len(prompts)]
            warmup_tasks.append(run_one(prompt))
        await asyncio.gather(*warmup_tasks)

    # Measured run
    t_scenario_start = time.perf_counter()
    tasks = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        tasks.append(run_one(prompt))

    results: list[RequestResult] = await asyncio.gather(*tasks)
    t_scenario_end = time.perf_counter()
    scenario_wall = t_scenario_end - t_scenario_start

    return compute_stats(results, scenario_wall, concurrency, max_tokens)


def compute_stats(
    results: list[RequestResult],
    wall_time: float,
    concurrency: int,
    max_tokens: int,
) -> ScenarioStats:
    """Compute percentiles and means from request results."""
    ttfts = [r.ttft for r in results]
    latencies_ms = [r.total_time * 1000 for r in results]
    throughputs = [r.throughput for r in results]

    all_itls: list[float] = []
    for r in results:
        all_itls.extend([itl * 1000 for itl in r.inter_token_latencies])

    total_tokens = sum(r.token_count for r in results)
    throughput_total = total_tokens / wall_time if wall_time > 0 else 0.0

    return ScenarioStats(
        name="",
        num_requests=len(results),
        ttft_p50=percentile(ttfts, 50) * 1000,  # ms
        ttft_p95=percentile(ttfts, 95) * 1000,
        ttft_p99=percentile(ttfts, 99) * 1000,
        ttft_mean=(sum(ttfts) / len(ttfts) * 1000) if ttfts else 0.0,
        throughput_mean=(sum(throughputs) / len(throughputs)) if throughputs else 0.0,
        throughput_total=throughput_total,
        latency_p50=percentile(latencies_ms, 50),
        latency_p95=percentile(latencies_ms, 95),
        latency_p99=percentile(latencies_ms, 99),
        latency_mean=(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        itl_p50=percentile(all_itls, 50),
        itl_p95=percentile(all_itls, 95),
        itl_mean=(sum(all_itls) / len(all_itls)) if all_itls else 0.0,
        total_tokens=total_tokens,
        total_time=wall_time,
    )


def print_table(scenarios: list[ScenarioStats]) -> None:
    """Print formatted results table."""
    header = (
        f"{'Scenario':<28} {'TTFT_p50':>9} {'TTFT_p95':>9} "
        f"{'Thr(t/s)':>9} {'ThrTot':>8} "
        f"{'Lat_p50':>9} {'Lat_p95':>9} "
        f"{'ITL_p50':>8} {'ITL_p95':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in scenarios:
        print(
            f"{s.name:<28} "
            f"{s.ttft_p50:>8.1f}ms {s.ttft_p95:>8.1f}ms "
            f"{s.throughput_mean:>8.1f} {s.throughput_total:>7.1f} "
            f"{s.latency_p50:>8.1f}ms {s.latency_p95:>8.1f}ms "
            f"{s.itl_p50:>7.1f}ms {s.itl_p95:>7.1f}ms"
        )


def print_comparison(rust_scenarios: list[ScenarioStats], vllm_scenarios: list[ScenarioStats]) -> None:
    """Print side-by-side comparison (ratio = rust/vllm, <1 means rust is faster)."""
    print()
    print("=== Comparison (ratio = rust/vllm, <1.0 means rust is faster) ===")
    header = f"{'Scenario':<28} {'TTFT':>8} {'Throughput':>11} {'Latency':>9}"
    print(header)
    print("-" * len(header))

    for rs, vs in zip(rust_scenarios, vllm_scenarios):
        ttft_ratio = rs.ttft_p50 / vs.ttft_p50 if vs.ttft_p50 > 0 else 0.0
        thr_ratio = rs.throughput_mean / vs.throughput_mean if vs.throughput_mean > 0 else 0.0
        lat_ratio = rs.latency_p50 / vs.latency_p50 if vs.latency_p50 > 0 else 0.0
        print(
            f"{rs.name:<28} "
            f"{ttft_ratio:>7.2f}x "
            f"{thr_ratio:>10.2f}x "
            f"{lat_ratio:>8.2f}x"
        )


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM serving engine via streaming SSE"
    )
    parser.add_argument("--base-url", required=True, help="Server base URL (e.g. http://localhost:8000)")
    parser.add_argument("--model", required=True, help="Model name for request payload")
    parser.add_argument("--prompts-file", required=True, help="Path to prompts JSON file")
    parser.add_argument("--prompt-length", default="short,medium,long", help="Comma-separated prompt lengths to test")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup requests (discarded)")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of measured requests per scenario")
    parser.add_argument("--concurrency", default="1", help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", default="64", help="Comma-separated max_tokens values")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    # Load prompts
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        print(f"Error: prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    with open(prompts_path) as f:
        all_prompts = json.load(f)

    prompt_lengths = [s.strip() for s in args.prompt_length.split(",")]
    concurrencies = [int(c.strip()) for c in args.concurrency.split(",")]
    max_tokens_list = [int(m.strip()) for m in args.max_tokens.split(",")]

    # Verify server is reachable
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"Warning: /v1/models returned {resp.status}", file=sys.stderr)
    except Exception as e:
        print(f"Error: cannot reach server at {args.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmarking: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Warmup: {args.warmup}, Requests: {args.num_requests}")
    print(f"Prompt lengths: {prompt_lengths}")
    print(f"Max tokens: {max_tokens_list}")
    print(f"Concurrency: {concurrencies}")
    print()

    scenarios: list[ScenarioStats] = []

    for pl in prompt_lengths:
        if pl not in all_prompts:
            print(f"Warning: prompt length '{pl}' not in prompts file, skipping", file=sys.stderr)
            continue
        prompts = all_prompts[pl]

        for mt in max_tokens_list:
            for conc in concurrencies:
                scenario_name = f"{pl}_mt{mt}_c{conc}"
                print(f"  Running: {scenario_name} ...", end="", flush=True)

                try:
                    stats = await run_scenario(
                        base_url=args.base_url,
                        model=args.model,
                        prompts=prompts,
                        max_tokens=mt,
                        concurrency=conc,
                        num_requests=args.num_requests,
                        warmup=args.warmup,
                    )
                    stats.name = scenario_name
                    scenarios.append(stats)
                    print(
                        f" done ({stats.total_tokens} tokens, "
                        f"{stats.throughput_total:.1f} tok/s total)"
                    )
                except Exception as e:
                    print(f" FAILED: {e}", file=sys.stderr)

    print()
    print(f"=== Results: {args.base_url} ===")
    print_table(scenarios)

    # Save to JSON
    if args.output:
        output_data = {
            "base_url": args.base_url,
            "model": args.model,
            "warmup": args.warmup,
            "num_requests": args.num_requests,
            "scenarios": [],
        }
        for s in scenarios:
            output_data["scenarios"].append({
                "name": s.name,
                "num_requests": s.num_requests,
                "ttft_p50_ms": round(s.ttft_p50, 2),
                "ttft_p95_ms": round(s.ttft_p95, 2),
                "ttft_p99_ms": round(s.ttft_p99, 2),
                "ttft_mean_ms": round(s.ttft_mean, 2),
                "throughput_mean_tps": round(s.throughput_mean, 2),
                "throughput_total_tps": round(s.throughput_total, 2),
                "latency_p50_ms": round(s.latency_p50, 2),
                "latency_p95_ms": round(s.latency_p95, 2),
                "latency_p99_ms": round(s.latency_p99, 2),
                "latency_mean_ms": round(s.latency_mean, 2),
                "itl_p50_ms": round(s.itl_p50, 2),
                "itl_p95_ms": round(s.itl_p95, 2),
                "itl_mean_ms": round(s.itl_mean, 2),
                "total_tokens": s.total_tokens,
                "total_time_s": round(s.total_time, 3),
            })

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
