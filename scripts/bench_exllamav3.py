#!/usr/bin/env python3
"""Decode throughput micro-bench for ExLlamaV3 (low-level path).

Uses the same low-level prefill+forward loop as upstream
`examples/generation_loop.py` — sidesteps the Generator's
paged-attention metadata setup that proved fragile in our
no-flash-attn environment. Apples-to-apples-ish with the
vllm-rust /v1/completions decode tps (same model, same
prompt length, same max_new_tokens, same hardware).

Usage:
    .venv-tmp/bin/python scripts/bench_exllamav3.py \\
        --model-dir <local EXL3 checkpoint snapshot dir> \\
        --prompt-len 128 \\
        --max-tokens 128 \\
        --runs 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Workaround: the wheel `exllamav3==0.0.33` is missing
# `modules/attention/*.py`. Use the GitHub source clone for the
# Python tree; pre-load the wheel's compiled .so so torch's
# JIT loader (which deadlocks in our WSL2 environment under
# torch.utils.cpp_extension.load) is bypassed via sys.modules.
_SRC = "/home/vasis/projects_hobby/vllm-rust/model_cache/exllamav3_src"
_EXT = "/home/vasis/.cache/torch_extensions/py312_cu130/exllamav3_ext"
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)
if os.path.isdir(_EXT):
    sys.path.insert(0, _EXT)

import torch  # noqa: E402
import exllamav3_ext  # noqa: F401, E402  -- preload before exllamav3 imports it

from exllamav3 import (  # noqa: E402
    Config,
    Model,
    Cache,
    CacheLayer_fp16,
    Tokenizer,
)
from exllamav3.generator.sampler.presets import GreedySampler  # noqa: E402


def make_prompt(words: int) -> str:
    return " ".join(["dummy"] * max(1, words))


def run_decode(
    model: Model,
    cache: Cache,
    tokenizer: Tokenizer,
    sampler: GreedySampler,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    batch_shape,
) -> dict:
    """Mirror examples/generation_loop.py timing: prefill is
    excluded from the decode-tps figure."""
    # Prefill all but the last token (so the first decode step
    # consumes a single token, matching how decode tps is measured).
    ctx_ids = prompt_ids.clone()

    prefill_params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": batch_shape,
    }
    torch.cuda.synchronize()
    t_prefill = time.perf_counter()
    model.prefill(input_ids=ctx_ids[:, :-1], params=prefill_params)
    torch.cuda.synchronize()
    ttft_s = time.perf_counter() - t_prefill

    # Decode loop — generates `max_new_tokens` starting with the
    # last prompt token's forward.
    torch.cuda.synchronize()
    t_decode = time.perf_counter()
    generated = 0
    while generated < max_new_tokens:
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": ctx_ids.shape[-1] - 1,
            "batch_shape": batch_shape,
        }
        logits = model.forward(input_ids=ctx_ids[:, -1:], params=params)
        # GreedySampler returns the chosen token ID(s).
        token_id = sampler.forward(logits[:, -1, :].unsqueeze(1))
        ctx_ids = torch.cat([ctx_ids, token_id], dim=-1)
        generated += 1
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_decode

    return {
        "ttft_ms": ttft_s * 1000.0,
        "decode_s": decode_s,
        "new_tokens": generated,
        "decode_tps": generated / decode_s,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-num-tokens", type=int, default=2048)
    args = p.parse_args()

    md = Path(args.model_dir).resolve()
    if not md.is_dir():
        sys.exit(f"model-dir not found: {md}")
    print(f"Loading {md}", flush=True)

    cfg = Config.from_directory(str(md))
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=args.max_num_tokens, layer_type=CacheLayer_fp16)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(cfg)

    prompt = make_prompt(args.prompt_len)
    sampler = GreedySampler()

    ids = tokenizer.encode(prompt, encode_special_tokens=False)
    # Move to GPU
    ids = ids.to("cuda")
    # batch_shape: (cache_bsz, cache_max_seq_len). max must cover prompt + new.
    batch_shape = (1, args.max_num_tokens)

    print(f"prompt_tokens={ids.shape[-1]}  max_new_tokens={args.max_tokens}  runs={args.runs}")

    # Warmup (small) — JIT autotune + filling kernel cache.
    _ = run_decode(model, cache, tokenizer, sampler, ids, 16, batch_shape)

    print()
    print(f"{'run':>3s}  {'ttft ms':>8s}  {'decode s':>8s}  {'tok/s':>8s}")
    print("-" * 36)
    per_run = []
    for i in range(args.runs):
        # Reuse the same cache — past_len=0 in prefill_params overwrites
        # positions 0..prompt_len-1 each call. Creating a fresh Cache here
        # would add a new (unallocated) CacheLayer to every Attention
        # module since cache.alloc() is only run during model.load().
        r = run_decode(model, cache, tokenizer, sampler, ids, args.max_tokens, batch_shape)
        per_run.append(r)
        print(f"{i:>3d}  {r['ttft_ms']:>8.1f}  {r['decode_s']:>8.3f}  {r['decode_tps']:>8.1f}",
              flush=True)
    med_tps = statistics.median(r["decode_tps"] for r in per_run)
    best_tps = max(r["decode_tps"] for r in per_run)
    med_ttft = statistics.median(r["ttft_ms"] for r in per_run)
    print()
    print(f"summary: med_ttft_ms={med_ttft:.1f}  med_decode_tps={med_tps:.1f}  best_decode_tps={best_tps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
