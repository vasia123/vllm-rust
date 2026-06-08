#!/usr/bin/env python3
"""Greedy coherence check for an EXL3 checkpoint via ExLlamaV3 (the reference
implementation, by the same author as the checkpoint).

Purpose: determine whether the degeneration we see in vllm-rust on a real
prompt ("...designed by- a- a- a-") comes from the 2.00bpw MODEL or from our
attention implementation. Runs the SAME prompt through ExLlamaV3 with pure
greedy sampling (temp=0, no repetition penalty) and prints the decoded text.

Mirrors scripts/bench_exllamav3.py for the import workaround + low-level
prefill/forward loop (the high-level Generator's paged metadata is fragile in
this no-flash WSL2 env).

    .venv-tmp/bin/python scripts/exllamav3_coherence.py \\
        --model-dir <snapshot dir> --max-tokens 120 \\
        --prompt "Write a short paragraph about the history of the Eiffel Tower."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--max-num-tokens", type=int, default=2048)
    p.add_argument(
        "--prompt",
        default="Write a short paragraph about the history of the Eiffel Tower.",
    )
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

    # Gemma chat format — match what the server's chat template emits.
    chat = (
        f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    )
    ids = tokenizer.encode(chat, encode_special_tokens=True).to("cuda")
    prompt_len = ids.shape[-1]
    print(f"prompt_tokens={prompt_len}  max_new_tokens={args.max_tokens}")

    sampler = GreedySampler()
    batch_shape = (1, args.max_num_tokens)

    # Prefill all but the last token.
    model.prefill(
        input_ids=ids[:, :-1],
        params={"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": batch_shape},
    )

    # Common Gemma EOS ids: <eos>=1, <end_of_turn>=106.
    eos_ids = {1, 106}
    new_ids = []
    ctx = ids
    for _ in range(args.max_tokens):
        logits = model.forward(
            input_ids=ctx[:, -1:],
            params={
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": ctx.shape[-1] - 1,
                "batch_shape": batch_shape,
            },
        )
        tok = sampler.forward(logits[:, -1, :].unsqueeze(1))
        tid = int(tok.flatten()[0].item())
        new_ids.append(tid)
        ctx = torch.cat([ctx, tok], dim=-1)
        if tid in eos_ids:
            break

    text = tokenizer.decode(torch.tensor([new_ids], device="cuda"))
    if isinstance(text, (list, tuple)):
        text = text[0]
    print("=== ExLlamaV3 greedy output ===")
    print(text)
    print(f"=== ({len(new_ids)} tokens generated) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
