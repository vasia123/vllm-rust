#!/usr/bin/env python3
"""
Phase 3a — bulk-migrate the Gemma family's local `XxxRmsNorm` structs
onto `crate::layers::RmsNorm` (variant `ScalePlusOne`).

For each file:
1. Locate the `struct XxxRmsNorm { ... }` + its `impl XxxRmsNorm { ... }`
   + `impl Module for XxxRmsNorm { ... }` block.
2. Replace the whole block with a type alias + helper:
       type XxxRmsNorm = crate::layers::RmsNorm;
       fn xxx_rms_norm(size, eps, vb) -> Result<XxxRmsNorm> { rms_norm_gemma(...) }
3. Rewrite all `XxxRmsNorm::new(` → `xxx_rms_norm(`.
4. Preserve original visibility (`struct` vs `pub(crate) struct`).

Idempotent: if the file already has the type alias, it's left alone.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Match the three blocks (struct decl + impl + impl Module) as a single
# group so we can replace them atomically.
RMSNORM_BLOCK_RE = re.compile(
    r"""
    (?P<vis>(?:pub\([^)]*\)\s+|pub\s+)?)
    struct\s+(?P<name>[A-Z][A-Za-z0-9_]*RmsNorm)\s*\{[^}]*\}\s*

    impl\s+(?P=name)\s*\{[^}]*?fn\s+new\([^}]*?Ok\(Self\s*\{[^}]*?\}\)\s*\}\s*\}\s*

    impl\s+Module\s+for\s+(?P=name)\s*\{[^}]*?fn\s+forward\(\&self,\s*xs:\s*&Tensor\)[^}]*?
    let\s+scale\s*=\s*\(&?self\.weight\.to_dtype\(DType::F32\)\?\s*\+\s*1\.0\)\?[^}]*?
    \}\s*\}
    """,
    re.VERBOSE | re.DOTALL,
)


def to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


REPLACEMENT = '''\
{vis}type {name} = crate::layers::RmsNorm;

#[inline]
{vis}fn {fn_name}(size: usize, eps: f64, vb: VarBuilder) -> Result<{name}> {{
    crate::layers::rms_norm_gemma(size, eps, vb)
}}'''


def migrate_file(path: Path, dry_run: bool) -> bool:
    src = path.read_text()
    if "type Gemma" in src and "RmsNorm = crate::layers::RmsNorm" in src:
        # Already migrated.
        return False

    m = RMSNORM_BLOCK_RE.search(src)
    if m is None:
        return False

    name = m.group("name")
    vis = m.group("vis")
    fn_name = to_snake(name)
    replacement = REPLACEMENT.format(vis=vis, name=name, fn_name=fn_name)

    new_src = src[: m.start()] + replacement + src[m.end():]

    # Rewrite call sites.
    new_src = re.sub(
        rf"\b{re.escape(name)}::new\s*\(",
        f"{fn_name}(",
        new_src,
    )

    if dry_run:
        print(f"would migrate {path}: {name} → {fn_name}")
    else:
        path.write_text(new_src)
        print(f"migrated {path}: {name} → {fn_name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n = 0
    for f in args.files:
        if migrate_file(f, args.dry_run):
            n += 1
    print(f"\n{n} file(s) {'would be ' if args.dry_run else ''}migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
