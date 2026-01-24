# Benchmarks

Benchmark suite for comparing vllm-rust against Python vLLM on the same model and hardware.

## Prerequisites

- Python 3.10+ with `aiohttp` installed (`pip install aiohttp`)
- Python vLLM installed (`pip install vllm`) — for comparison runs
- CUDA-capable GPU with enough VRAM for the target model
- vllm-rust built in release mode (`cargo build --release -p vllm-server`)

## Quick Start

### Single Server Benchmark

Test against a single running server:

```bash
# Start your server (either vllm-rust or Python vLLM)
cargo run --release -p vllm-server -- serve --model Qwen/Qwen3-0.6B --port 8100 &

# Run benchmark
python benchmarks/bench_serving.py \
  --base-url http://localhost:8100 \
  --model Qwen/Qwen3-0.6B \
  --prompts-file benchmarks/prompts.json \
  --concurrency 1 \
  --max-tokens 64 \
  --num-requests 10
```

### Full Comparison

Automated comparison between vllm-rust and Python vLLM:

```bash
cd benchmarks
bash run_comparison.sh
```

Override defaults via environment variables:

```bash
MODEL=Qwen/Qwen3-1.7B \
NUM_REQUESTS=50 \
CONCURRENCY=1,2,4,8 \
MAX_TOKENS=64,128,256 \
bash run_comparison.sh
```

## Files

| File | Purpose |
|------|---------|
| `bench_serving.py` | Benchmark client — sends requests, parses SSE, computes metrics |
| `run_comparison.sh` | Orchestration — builds, starts servers, runs benchmarks, compares |
| `prompts.json` | Fixed prompts at known lengths (short/medium/long) |
| `results/` | Output directory for JSON results (created by run_comparison.sh) |

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token (request send → first SSE token event) |
| Throughput/req | Tokens per second per individual request |
| Throughput total | Aggregate tokens per second across all concurrent requests |
| Latency | End-to-end time (request send → stream complete) |
| ITL | Inter-token latency (time between consecutive token events) |

All metrics reported at P50, P95, P99, and mean.

## bench_serving.py Options

```
--base-url         Server URL (required)
--model            Model name for request payload (required)
--prompts-file     Path to prompts JSON (required)
--prompt-length    Comma-separated lengths: short,medium,long (default: all)
--warmup           Warmup requests discarded before measurement (default: 3)
--num-requests     Measured requests per scenario (default: 20)
--concurrency      Comma-separated concurrency levels (default: 1)
--max-tokens       Comma-separated max token counts (default: 64)
--output           Output JSON file path
```

## run_comparison.sh Environment Variables

```
MODEL              Model to benchmark (default: Qwen/Qwen3-0.6B)
RUST_PORT          Port for vllm-rust (default: 8100)
VLLM_PORT          Port for Python vLLM (default: 8200)
NUM_REQUESTS       Requests per scenario (default: 20)
WARMUP             Warmup count (default: 3)
CONCURRENCY        Concurrency levels (default: 1,2,4)
MAX_TOKENS         Max token counts (default: 64,128)
PROMPT_LENGTH      Prompt lengths to test (default: short,medium)
```

## Output Format

Console output:

```
Scenario                     TTFT_p50  TTFT_p95  Thr(t/s)   ThrTot  Lat_p50   Lat_p95  ITL_p50  ITL_p95
short_mt64_c1                  12.3ms    14.1ms     52.1    52.1     1230.0ms  1280.0ms   23.6ms   28.1ms
```

Comparison output:

```
Scenario                       TTFT   Thr/req   ThrTot   Latency
short_mt64_c1                 0.66x     1.52x    1.52x     0.66x
```

Ratios < 1.0 for TTFT/Latency mean rust is faster. Ratios > 1.0 for Throughput mean rust is faster.

## JSON Output Schema

```json
{
  "base_url": "http://localhost:8100",
  "model": "Qwen/Qwen3-0.6B",
  "warmup": 3,
  "num_requests": 20,
  "scenarios": [
    {
      "name": "short_mt64_c1",
      "ttft_p50_ms": 12.34,
      "ttft_p95_ms": 14.12,
      "throughput_mean_tps": 52.1,
      "throughput_total_tps": 52.1,
      "latency_p50_ms": 1230.0,
      "latency_p95_ms": 1280.0,
      "itl_p50_ms": 23.6,
      "total_tokens": 1280,
      "total_time_s": 24.567
    }
  ]
}
```

## Design Decisions

- **External measurement** — no server instrumentation; both servers emit standard SSE events
- **Python client** — pragmatic choice: aiohttp handles SSE, user already has Python for vLLM
- **Sequential servers** — avoids GPU memory contention; only one server runs at a time
- **Fixed prompts** — deterministic text at controlled lengths for reproducibility
- **Greedy decoding** — both servers use argmax by default, eliminating sampling variance
- **Warmup** — discards initial requests to account for CUDA kernel JIT and memory allocation
