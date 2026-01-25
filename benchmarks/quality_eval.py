#!/usr/bin/env python3
"""Model quality evaluation using lm-evaluation-harness.

Compares vllm-rust output quality against Python vLLM using standardized
NLP benchmarks. Uses EleutherAI's lm-evaluation-harness with the
local-completions backend for OpenAI-compatible APIs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SUITES = {
    "quick": ["hellaswag"],
    "standard": ["hellaswag", "arc_easy", "truthfulqa_mc2"],
    "full": ["hellaswag", "arc_easy", "truthfulqa_mc2", "mmlu"],
}


def run_eval(
    base_url: str,
    model: str,
    tasks: list[str],
    limit: int | None,
    output_dir: Path,
) -> dict:
    """Run lm-eval against an OpenAI-compatible endpoint.

    Args:
        base_url: Server URL (e.g., http://localhost:8000/v1/completions)
        model: Model name for the API
        tasks: List of benchmark tasks to run
        limit: Optional sample limit per task
        output_dir: Directory for lm-eval output

    Returns:
        Parsed results dictionary from lm-eval
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        ",".join(tasks),
        "--model_args",
        f"model={model},base_url={base_url},num_concurrent=4,max_retries=3",
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"lm_eval stdout:\n{result.stdout}", file=sys.stderr)
        print(f"lm_eval stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Find the most recent results directory
    result_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not result_dirs:
        print(f"Error: No results found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    latest = max(result_dirs, key=lambda p: p.stat().st_mtime)

    # Find results file (may have timestamp suffix)
    results_files = list(latest.glob("results*.json"))
    if not results_files:
        print(f"Error: No results*.json found in {latest}", file=sys.stderr)
        sys.exit(1)

    results_file = max(results_files, key=lambda p: p.stat().st_mtime)

    with open(results_file) as f:
        return json.load(f)


def get_accuracy(metrics: dict) -> float:
    """Extract accuracy from task metrics, handling different metric names."""
    # Try common accuracy metric names in order of preference
    for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
        if key in metrics:
            return float(metrics[key])
    return 0.0


def compare(rust: dict, vllm: dict, tolerance: float) -> bool:
    """Compare accuracy between rust and vllm results.

    Args:
        rust: Results from vllm-rust
        vllm: Results from Python vLLM
        tolerance: Maximum acceptable accuracy difference

    Returns:
        True if all tasks pass (within tolerance), False otherwise
    """
    ok = True
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON: vllm-rust vs Python vLLM")
    print("=" * 70)
    print(f"{'Task':<25} {'Rust':>10} {'vLLM':>10} {'Diff':>10} {'Status':>10}")
    print("-" * 70)

    rust_results = rust.get("results", {})
    vllm_results = vllm.get("results", {})

    for task in rust_results:
        if task not in vllm_results:
            print(f"{task:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'SKIP':>10}")
            continue

        rust_acc = get_accuracy(rust_results[task])
        vllm_acc = get_accuracy(vllm_results[task])
        diff = rust_acc - vllm_acc

        if abs(diff) <= tolerance:
            status = "OK"
        elif diff < -tolerance:
            status = "DEGRADED"
            ok = False
        else:
            status = "BETTER"

        print(
            f"{task:<25} {rust_acc:>10.4f} {vllm_acc:>10.4f} {diff:>+10.4f} {status:>10}"
        )

    print("=" * 70)

    if ok:
        print("Result: PASS - All benchmarks within tolerance")
    else:
        print(f"Result: FAIL - Some benchmarks degraded beyond {tolerance:.1%} tolerance")

    return ok


def print_results(results: dict) -> None:
    """Print results for a single evaluation run."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Task':<25} {'Accuracy':>15}")
    print("-" * 50)

    for task, metrics in results.get("results", {}).items():
        acc = get_accuracy(metrics)
        print(f"{task:<25} {acc:>15.4f}")

    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model quality evaluation using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick check with hellaswag
  python quality_eval.py --model Qwen/Qwen3-0.6B --suite quick

  # Fast iteration with limited samples
  python quality_eval.py --model Qwen/Qwen3-0.6B --limit 100

  # Rust-only baseline
  python quality_eval.py --model Qwen/Qwen3-0.6B --rust-only

  # Full comparison with saved results
  python quality_eval.py --model Qwen/Qwen3-0.6B --suite standard --output results.json
""",
    )

    parser.add_argument(
        "--rust-url",
        default="http://localhost:8000/v1/completions",
        help="vllm-rust server URL (default: http://localhost:8000/v1/completions)",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8001/v1/completions",
        help="Python vLLM server URL (default: http://localhost:8001/v1/completions)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--suite",
        choices=["quick", "standard", "full"],
        default="quick",
        help="Benchmark suite to run (default: quick)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit samples per task (for fast testing)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Maximum acceptable accuracy difference (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Only evaluate vllm-rust, skip comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save comparison results to JSON file",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/lm_eval_quality"),
        help="Working directory for lm-eval output (default: /tmp/lm_eval_quality)",
    )

    args = parser.parse_args()

    tasks = SUITES[args.suite]
    print(f"Benchmark suite: {args.suite}")
    print(f"Tasks: {', '.join(tasks)}")
    if args.limit:
        print(f"Sample limit: {args.limit}")

    rust_output_dir = args.work_dir / "rust"
    vllm_output_dir = args.work_dir / "vllm"

    print(f"\n[1/2] Evaluating vllm-rust at {args.rust_url}...")
    rust = run_eval(args.rust_url, args.model, tasks, args.limit, rust_output_dir)

    if args.rust_only:
        print_results(rust)
        return

    print(f"\n[2/2] Evaluating Python vLLM at {args.vllm_url}...")
    vllm = run_eval(args.vllm_url, args.model, tasks, args.limit, vllm_output_dir)

    ok = compare(rust, vllm, args.tolerance)

    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "suite": args.suite,
            "tolerance": args.tolerance,
            "limit": args.limit,
            "rust_url": args.rust_url,
            "vllm_url": args.vllm_url,
            "rust_results": rust,
            "vllm_results": vllm,
            "passed": ok,
        }
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
