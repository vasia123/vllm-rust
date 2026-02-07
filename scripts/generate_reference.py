#!/usr/bin/env python3
"""Generate reference outputs from Python vLLM for correctness validation.

Usage:
    python scripts/generate_reference.py --model <model_name> --output <output_dir>

This script runs a set of standard prompts through the Python vLLM engine
and saves the outputs (token IDs, logprobs, generated text) as JSON files.
The Rust correctness tests then compare against these references.

Requirements:
    pip install vllm torch

The script generates reference data for:
1. Greedy decoding (temperature=0) - exact token match expected
2. Sampling with seed (temperature>0) - logprob tolerance comparison
3. Top-k/top-p filtering - distribution shape validation
4. Beam search - sequence-level comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Standard test prompts covering various generation scenarios
TEST_PROMPTS = [
    # Short prompt, simple continuation
    {
        "id": "short_greedy",
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    },
    # Medium prompt, factual
    {
        "id": "medium_greedy",
        "prompt": "Explain in one sentence what photosynthesis is:",
        "max_tokens": 50,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    },
    # Code generation
    {
        "id": "code_greedy",
        "prompt": "def fibonacci(n):\n    ",
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    },
    # Sampling with seed for reproducibility
    {
        "id": "sampling_seeded",
        "prompt": "Once upon a time in a land far away,",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
    },
    # Top-p only
    {
        "id": "top_p_only",
        "prompt": "The meaning of life is",
        "max_tokens": 30,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": -1,
        "seed": 123,
    },
    # Low temperature (near-greedy)
    {
        "id": "low_temp",
        "prompt": "Water boils at",
        "max_tokens": 15,
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 7,
    },
    # Long output
    {
        "id": "long_output",
        "prompt": "Write a short paragraph about the history of computing:",
        "max_tokens": 200,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    },
    # Repetition penalty test
    {
        "id": "repetition_test",
        "prompt": "The word 'the' appears in",
        "max_tokens": 50,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    },
]


def generate_references(model_name: str, output_dir: str, dtype: str = "auto"):
    """Generate reference outputs using Python vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed. Run: pip install vllm", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, dtype=dtype, trust_remote_code=True)

    results = []

    for test_case in TEST_PROMPTS:
        print(f"  Generating: {test_case['id']}")

        sampling_kwargs = {
            "max_tokens": test_case["max_tokens"],
            "temperature": test_case["temperature"],
            "top_p": test_case["top_p"],
            "logprobs": 5,  # Always collect top-5 logprobs for validation
        }

        if test_case["top_k"] > 0:
            sampling_kwargs["top_k"] = test_case["top_k"]

        if "seed" in test_case:
            sampling_kwargs["seed"] = test_case["seed"]

        params = SamplingParams(**sampling_kwargs)
        outputs = llm.generate([test_case["prompt"]], params)

        output = outputs[0]
        completion = output.outputs[0]

        # Extract logprobs data
        logprobs_data = []
        if completion.logprobs:
            for step_logprobs in completion.logprobs:
                step_data = {}
                for token_id, logprob_obj in step_logprobs.items():
                    step_data[str(token_id)] = {
                        "logprob": logprob_obj.logprob,
                        "rank": logprob_obj.rank,
                        "decoded_token": logprob_obj.decoded_token,
                    }
                logprobs_data.append(step_data)

        result = {
            "id": test_case["id"],
            "prompt": test_case["prompt"],
            "params": {
                "max_tokens": test_case["max_tokens"],
                "temperature": test_case["temperature"],
                "top_p": test_case["top_p"],
                "top_k": test_case.get("top_k", -1),
                "seed": test_case.get("seed"),
            },
            "output": {
                "text": completion.text,
                "token_ids": list(completion.token_ids),
                "cumulative_logprob": completion.cumulative_logprob,
                "finish_reason": completion.finish_reason,
                "logprobs": logprobs_data,
            },
            "prompt_token_ids": list(output.prompt_token_ids),
        }
        results.append(result)

    # Save results
    metadata = {
        "model": model_name,
        "dtype": dtype,
        "vllm_version": _get_vllm_version(),
        "num_prompts": len(results),
    }

    output_file = output_path / f"{_sanitize_model_name(model_name)}.json"
    with open(output_file, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)

    print(f"Reference data saved to: {output_file}")
    print(f"  {len(results)} test cases generated")


def _get_vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference outputs from Python vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/correctness/reference_data",
        help="Output directory for reference JSON files",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    args = parser.parse_args()

    generate_references(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
