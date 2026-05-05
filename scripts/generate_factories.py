#!/usr/bin/env python3
"""
Phase 2.2 — generate per-architecture ArchFactory stubs from the legacy
match-arm dispatch in `crates/core/src/models/mod.rs`.

The script is non-interactive: it parses the 11 dispatch functions, joins
their arms into a per-architecture capability table, then writes one
`crates/core/src/models/factories/<snake_name>.rs` file per unique
`builder_struct`.

Output is **non-binding**: a human reviews each generated factory before
it is merged. Hand-edited factories (currently `llama`, `qwen2`,
`mixtral`) are skipped via the `--skip` flag.

Usage:
    python3 scripts/generate_factories.py \\
        --mod-rs crates/core/src/models/mod.rs \\
        --out-dir crates/core/src/models/factories \\
        [--skip llama qwen2 mixtral] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ─── Dispatch function recognition ──────────────────────────────────────────

DISPATCH_FNS = {
    "from_config":               "build",
    "from_config_with_quant":    "build_quant",
    "from_config_with_tp":       "build_with_tp",
    "from_config_with_pp":       "build_with_pp",
    "from_config_with_lora":     "build_with_lora",
    "from_config_encoder_decoder": "build_encoder_decoder",
    # Speculative drafts route via SpeculativeFactory; surfaced as a
    # boolean capability for now and handled by a separate hand-written
    # factory in Phase 2.5.
    "mtp_from_config":           "speculative_mtp",
    "eagle1_from_config":        "speculative_eagle1",
    "eagle3_from_config":        "speculative_eagle3",
    "medusa_from_config":        "speculative_medusa",
    "mlp_speculator_from_config": "speculative_mlp",
}

DISPATCH_FN_RE = re.compile(
    r"^\s*pub fn (?P<name>" + "|".join(DISPATCH_FNS.keys()) + r")\(",
    re.MULTILINE,
)


# Match arm patterns. The tricky part is that arms can span many lines
# with `|` continuation. We collapse the body into a single line per arm
# before applying the regex.

ARM_HEAD_RE = re.compile(
    r"""
    (?P<patterns>(?:"[^"]+"\s*(?:\|\s*"[^"]+"\s*)*))   # one or more arms with `|`
    =>\s*
    """,
    re.VERBOSE | re.DOTALL,
)

ARM_NAMES_RE = re.compile(r'"([^"]+)"')


def _read_balanced(text: str, start: int) -> tuple[str, int] | None:
    """
    From `text[start]`, read a single Rust expression terminated by a
    top-level comma (or end-of-block `}`). Tracks (), [], {} so commas
    inside calls don't fool us. Returns (body, end_index_exclusive).
    """
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if escape:
            escape = False
            i += 1
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if in_char:
            if ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "'" and (i + 2 < n) and text[i + 2] == "'":
            # 'x' char literal; skip 3 chars total
            i += 3
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            if depth_brace == 0:
                # End of enclosing block; treat as terminator.
                return text[start:i].strip(), i
            depth_brace -= 1
        elif (
            ch == ","
            and depth_paren == 0
            and depth_bracket == 0
            and depth_brace == 0
        ):
            return text[start:i].strip(), i + 1
        i += 1
    return text[start:].strip(), n

# Captures e.g. `LlamaForCausalLM::new(cfg, vb)?` → ("LlamaForCausalLM", "new", "cfg, vb")
# Looks specifically for the `::method` qualifier so it ignores the
# wrapping `Ok(`, `Box::new(`, `LoraEnabledModel::Llama(` constructors —
# the actual concrete model type is always followed by `::new` (or
# `::from_model_config`, `::new_with_tp`, …).
BUILDER_RE = re.compile(
    r"""
    (?P<struct>[A-Za-z_][A-Za-z0-9_]*)
    ::
    (?P<method>new[A-Za-z_0-9]*|from_[A-Za-z_0-9]+)
    \s*\(
    """,
    re.VERBOSE,
)


@dataclass
class ArmRecord:
    arch_names: list[str] = field(default_factory=list)
    builder_struct: str = ""
    builder_method: str = "new"
    raw_body: str = ""


@dataclass
class ArchTableEntry:
    canonical_struct: str
    aliases: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    builder_calls: dict[str, str] = field(default_factory=dict)
    """capability → raw `Foo::new(...)` body (used to drive code-gen)."""


def find_dispatch_fn_bodies(src: str) -> dict[str, str]:
    """
    Carve out the body of each `pub fn from_config_*` so we don't pull
    arms from one dispatch into another.
    """
    matches = list(DISPATCH_FN_RE.finditer(src))
    bodies: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group("name")
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(src)
        bodies[name] = src[start:end]
    return bodies


def _read_block(text: str, start: int) -> tuple[str, int]:
    """Read a balanced `{ ... }` block starting at `text[start] == '{'`."""
    assert text[start] == "{"
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    return text[start:], n


def extract_arms(body: str) -> list[ArmRecord]:
    arms: list[ArmRecord] = []
    for m in ARM_HEAD_RE.finditer(body):
        names = ARM_NAMES_RE.findall(m.group("patterns"))
        if not names:
            continue
        body_start = m.end()
        # Skip whitespace then peek: `{ ... }` block-arm vs expr-arm.
        while body_start < len(body) and body[body_start].isspace():
            body_start += 1
        if body_start >= len(body):
            continue
        if body[body_start] == "{":
            block_str, _ = _read_block(body, body_start)
            # Strip the wrapping braces so render_factory emits the
            # body cleanly inside the new method's `{ ... }`.
            inner = block_str[1:-1].strip().rstrip(",").strip()
            body_str = inner
        else:
            body_str, _ = _read_balanced(body, body_start)
            if not body_str:
                continue
        if body_str.startswith("Err"):
            # Deprecated / removed arms — surface as 'unsupported'.
            arms.append(ArmRecord(arch_names=names, builder_struct="", raw_body=body_str))
            continue
        # Find first `Struct::method(` that isn't a wrapper (Box, Ok, …)
        # or an enum-variant constructor (LoraEnabledModel::Llama).
        wrapper_blacklist = {"Box", "Ok", "Err", "Some", "None", "LoraEnabledModel"}
        struct = ""
        method = "new"
        for bm in BUILDER_RE.finditer(body_str):
            if bm.group("struct") not in wrapper_blacklist:
                struct = bm.group("struct")
                method = bm.group("method")
                break
        if not struct:
            arms.append(ArmRecord(arch_names=names, builder_struct="", raw_body=body_str))
            continue
        arms.append(ArmRecord(
            arch_names=names,
            builder_struct=struct,
            builder_method=method,
            raw_body=body_str,
        ))
    return arms


def aggregate(src: str) -> dict[str, ArchTableEntry]:
    """
    Group arms by *canonical arch_name* (the first arch_name observed
    in `from_config`). All other dispatch functions look up arms by
    arch_name and contribute their capability to the same group.

    This handles e.g. `MixtralForCausalLM` whose `build` uses
    `MixtralForCausalLM::new` but whose `build_with_tp` uses
    `MixtralTpForCausalLM::new_with_tp`. They share an arch_name and
    therefore one factory.
    """
    bodies = find_dispatch_fn_bodies(src)

    # Pass 1: from `from_config`, learn canonical arch_name → builder_struct.
    arch_to_canonical: dict[str, str] = {}  # arch_name → canonical key
    table: dict[str, ArchTableEntry] = {}

    plain_arms = extract_arms(bodies.get("from_config", ""))
    for arm in plain_arms:
        if not arm.builder_struct:
            for n in arm.arch_names:
                arch_to_canonical[n] = "__deprecated__"
            entry = table.setdefault("__deprecated__", ArchTableEntry(canonical_struct=""))
            entry.aliases.update(arm.arch_names)
            entry.capabilities.add("__deprecated__")
            entry.builder_calls["__deprecated__"] = arm.raw_body
            continue
        canonical = arm.arch_names[0]
        for n in arm.arch_names:
            arch_to_canonical[n] = canonical
        entry = table.setdefault(canonical, ArchTableEntry(canonical_struct=arm.builder_struct))
        entry.aliases.update(arm.arch_names)
        entry.capabilities.add("build")
        entry.builder_calls["build"] = arm.raw_body

    # Pass 2: every other dispatch fn — look up by arch_name, record
    # the body under the matching canonical entry. Arch_names that
    # appear ONLY in a non-`from_config` dispatch (e.g. speculative
    # drafts that are loaded out-of-band) get their own canonical
    # entry under their first-seen alias.
    for fn_name, body in bodies.items():
        if fn_name == "from_config":
            continue
        cap = DISPATCH_FNS[fn_name]
        for arm in extract_arms(body):
            if not arm.builder_struct:
                continue
            canonical = None
            for n in arm.arch_names:
                if n in arch_to_canonical and arch_to_canonical[n] != "__deprecated__":
                    canonical = arch_to_canonical[n]
                    break
            if canonical is None:
                # Speculative draft / encoder-decoder only arch — give
                # it its own canonical key.
                canonical = arm.arch_names[0]
                for n in arm.arch_names:
                    arch_to_canonical[n] = canonical
                table.setdefault(canonical, ArchTableEntry(canonical_struct=arm.builder_struct))
            entry = table[canonical]
            entry.aliases.update(arm.arch_names)
            entry.capabilities.add(cap)
            entry.builder_calls[cap] = arm.raw_body
    return table


def to_snake(name: str) -> str:
    """`MixtralForCausalLM` → `mixtral_for_causal_lm`."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    # Collapse runs of underscores (HF puts `_` literally in some names,
    # e.g. `Nemotron_H_Nano_VL_V2` → `nemotron_h__nano_vl__v2`).
    return re.sub(r"_+", "_", s2)


def short_name(struct: str) -> str:
    """`MixtralForCausalLM` → `mixtral`. Used for filename + factory name."""
    return to_snake(struct).replace("_for_causal_lm", "").replace("_for_conditional_generation", "")


def display_name(struct: str) -> str:
    """`MixtralForCausalLM` → `Mixtral`. Camel-case version of `short_name`."""
    s = struct
    for suffix in (
        "ForCausalLM",
        "ForConditionalGeneration",
        "ForSequenceEmbedding",
        "ForRetrieval",
        "ForClassification",
        "Model",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s or struct


# ─── Code generation ────────────────────────────────────────────────────────

FACTORY_TEMPLATE = '''\
//! AUTO-GENERATED Phase 2 factory for `{struct}`.
//!
//! Generated by `scripts/generate_factories.py` from the legacy match-arm
//! dispatch in `crates/core/src/models/mod.rs`. Bodies are lifted
//! verbatim — re-running the generator overwrites this file.
//!
//! Capabilities: {caps_str}

#![allow(unused_imports, unused_variables)]

use std::any::Any;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::ModelForward;

use super::super::factory::{{ArchFactory, ArchInfo, Capabilities}};
use super::super::*;

pub const ARCH_NAMES: &[&str] = &[
{arch_names_block}
];

static INFO: ArchInfo = ArchInfo::new(
    "{display_name}",
    {capabilities_expr},
);

pub struct {factory_name};
pub static FACTORY: {factory_name} = {factory_name};

impl ArchFactory for {factory_name} {{
    fn arch_names(&self) -> &'static [&'static str] {{
        ARCH_NAMES
    }}
    fn info(&self) -> &'static ArchInfo {{
        &INFO
    }}

{methods}
    fn as_any(&self) -> &dyn Any {{
        self
    }}
}}
'''


def _capabilities_expr(caps: set[str]) -> str:
    """Compose `Capabilities::FOO.union(Capabilities::BAR)`."""
    flag_map = {
        "build_quant":       "QUANTIZED",
        "build_with_tp":     "TP",
        "build_with_pp":     "PP",
        "build_with_lora":   "LORA",
        "build_encoder_decoder": "ENCODER_DECODER",
        "speculative_mtp":   "SPECULATIVE_DRAFT",
        "speculative_eagle1": "SPECULATIVE_DRAFT",
        "speculative_eagle3": "SPECULATIVE_DRAFT",
        "speculative_medusa": "SPECULATIVE_DRAFT",
        "speculative_mlp":   "SPECULATIVE_DRAFT",
    }
    flags = sorted({flag_map[c] for c in caps if c in flag_map})
    if not flags:
        return "Capabilities::empty()"
    if len(flags) == 1:
        return f"Capabilities::{flags[0]}"
    expr = f"Capabilities::{flags[0]}"
    for f in flags[1:]:
        expr += f"\n        .union(Capabilities::{f})"
    return expr


METHOD_SIGS = {
    "build": (
        "fn build(&self, cfg: &ModelConfig, vb: VarBuilder) "
        "-> Result<Box<dyn ModelForward>, ModelError>"
    ),
    "build_quant": (
        "fn build_quant(&self, cfg: &ModelConfig, vb: VarBuilder<'static>, "
        "weight_loader: &dyn crate::quantization::QuantizedWeightLoader) "
        "-> Result<Box<dyn ModelForward>, ModelError>"
    ),
    "build_with_tp": (
        "fn build_with_tp(&self, cfg: &ModelConfig, vb: VarBuilder, "
        "pg: &dyn crate::distributed::ProcessGroup, "
        "tp_ctx: super::super::tp_layers::TpContext) "
        "-> Result<Box<dyn ModelForward>, ModelError>"
    ),
    "build_with_pp": (
        "fn build_with_pp(&self, cfg: &ModelConfig, vb: VarBuilder, "
        "stage: &crate::distributed::PipelineStageConfig) "
        "-> Result<Box<dyn ModelForward>, ModelError>"
    ),
    "build_with_lora": (
        "fn build_with_lora(&self, cfg: &ModelConfig, vb: VarBuilder) "
        "-> Result<super::super::LoraEnabledModel, ModelError>"
    ),
    "build_encoder_decoder": (
        "fn build_encoder_decoder(&self, cfg: &ModelConfig, vb: VarBuilder) "
        "-> Result<Box<dyn crate::engine::ModelForEncoderDecoder>, ModelError>"
    ),
}


def _adapt_body(cap: str, body: str) -> str:
    """
    Adapt a legacy arm body to the new method signature.

    Legacy `from_config_with_quant` builds `weight_loader: Box<dyn ...>`
    once and passes it as `weight_loader.as_ref()`. The new
    `build_quant` method receives `weight_loader: &dyn ...` directly,
    so the `.as_ref()` call must be stripped. Same idea for `pg.as_ref()`
    if/when it appears.
    """
    body = body.rstrip().rstrip(",").strip()
    body = body.replace("weight_loader.as_ref()", "weight_loader")
    return body


def render_factory(short: str, entry: ArchTableEntry) -> str:
    factory_name = "".join(p.capitalize() for p in short.split("_")) + "ArchFactory"
    aliases = sorted(entry.aliases)
    arch_names_block = ",\n".join(f'    "{n}"' for n in aliases) + ","
    caps_str = ", ".join(sorted(entry.capabilities)) or "(none)"

    method_blocks: list[str] = []
    for cap in [
        "build", "build_quant", "build_with_tp",
        "build_with_pp", "build_with_lora", "build_encoder_decoder",
    ]:
        if cap not in entry.capabilities:
            continue
        sig = METHOD_SIGS[cap]
        body_expr = _adapt_body(cap, entry.builder_calls[cap])
        method_blocks.append(f"    {sig} {{\n        {body_expr}\n    }}\n")
    methods = "\n".join(method_blocks)

    return FACTORY_TEMPLATE.format(
        struct=entry.canonical_struct,
        caps_str=caps_str,
        arch_names_block=arch_names_block,
        display_name=display_name(entry.canonical_struct),
        capabilities_expr=_capabilities_expr(entry.capabilities),
        factory_name=factory_name,
        methods=methods,
    )


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod-rs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip", nargs="*", default=[], help="skip these short names (already hand-written)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.mod_rs.read_text()
    table = aggregate(src)

    skip = set(args.skip)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    summary: list[tuple[str, int, str]] = []

    for struct, entry in sorted(table.items()):
        if struct == "__deprecated__":
            # Deprecated arms are handled by a single hand-written
            # `deprecated.rs` (Phase 2.5). Skip generation here.
            continue
        short = short_name(struct)
        if not short:
            continue
        if short in skip:
            skipped += 1
            continue
        path = out_dir / f"{short}.rs"
        if path.exists():
            skipped += 1
            continue
        body = render_factory(short, entry)
        if not args.dry_run:
            path.write_text(body)
        written += 1
        summary.append((short, len(entry.aliases), ",".join(sorted(entry.capabilities))))

    print(f"wrote {written} files, skipped {skipped}")
    for s, n_aliases, caps in summary[:20]:
        print(f"  {s:<40} {n_aliases:>3} aliases   caps={caps}")
    if len(summary) > 20:
        print(f"  ... and {len(summary) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
