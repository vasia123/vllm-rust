import re, sys

SRC = "/home/vasis/projects_hobby/vllm-rust/reference/llama.cpp/ggml/src/ggml-common.h"
text = open(SRC).read()

# Tables we need: (c_type -> rust_type)
WANT = {
    "kmask_iq2xs":  "u8",
    "ksigns_iq2xs": "u8",
    "iq2xs_grid":   "u64",
    "iq2s_grid":    "u64",
    "iq3xxs_grid":  "u32",
    "iq3s_grid":    "u32",
    "kvalues_iq4nl":"i8",
}

# Match GGML_TABLE_BEGIN(type, name, n) ... GGML_TABLE_END()
pat = re.compile(r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*(\w+)\s*,\s*(\d+)\s*\)(.*?)GGML_TABLE_END\(\)", re.S)

out = []
out.append("//! GGML I-quant codebook tables, extracted byte-exact from")
out.append("//! `reference/llama.cpp/ggml/src/ggml-common.h` by")
out.append("//! `scripts/extract_iq_tables.py`. DO NOT EDIT BY HAND — regenerate.")
out.append("//!")
out.append("//! Source commit: see `reference/llama.cpp` HEAD at generation time.")
out.append("#![allow(clippy::all)]")
out.append("#![cfg_attr(rustfmt, rustfmt_skip)]")
out.append("")

found = {}
for m in pat.finditer(text):
    name, n, body = m.group(1), int(m.group(2)), m.group(3)
    if name not in WANT:
        continue
    rty = WANT[name]
    # parse numeric tokens (hex or decimal), strip trailing commas / comments
    body = re.sub(r"//.*", "", body)
    toks = re.findall(r"0x[0-9a-fA-F]+|-?\d+", body)
    vals = []
    for t in toks:
        if t.lower().startswith("0x"):
            v = int(t, 16)
        else:
            v = int(t)
        vals.append(v)
    assert len(vals) == n, f"{name}: parsed {len(vals)} != declared {n}"
    found[name] = (rty, n, vals)

# Emit in a deterministic order
for name in WANT:
    if name not in found:
        print(f"!!! MISSING TABLE: {name}", file=sys.stderr); sys.exit(1)
    rty, n, vals = found[name]
    suffix = {"u8":"","u32":"","u64":"","i8":""}[rty]
    def fmt(v):
        if rty == "u64":
            return f"0x{v & 0xFFFFFFFFFFFFFFFF:016x}"
        if rty == "u32":
            return f"0x{v & 0xFFFFFFFF:08x}"
        return str(v)
    out.append(f"pub static {name.upper()}: [{rty}; {n}] = [")
    line = "    "
    per = 8 if rty in ("u64",) else (8 if rty=="u32" else 16)
    for i, v in enumerate(vals):
        line += fmt(v) + ", "
        if (i+1) % per == 0:
            out.append(line.rstrip()); line = "    "
    if line.strip():
        out.append(line.rstrip())
    out.append("];")
    out.append("")

dst = "/home/vasis/projects_hobby/vllm-rust/crates/core/src/quantization/gguf/iq/tables.rs"
import os
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write("\n".join(out) + "\n")
print("wrote", dst)

# --- CUDA header (same tables as __device__ const, for the dequant kernel) ---
CTY = {"u8": "uint8_t", "u32": "uint32_t", "u64": "uint64_t", "i8": "int8_t"}
cu = []
cu.append("// GGML I-quant codebook tables, extracted byte-exact from")
cu.append("// reference/llama.cpp/ggml/src/ggml-common.h by scripts/extract_iq_tables.py.")
cu.append("// DO NOT EDIT BY HAND — regenerate. Mirrors crates/core/.../iq/tables.rs.")
cu.append("#pragma once")
cu.append("#include <stdint.h>")
cu.append("")
for name in WANT:
    rty, n, vals = found[name]
    cty = CTY[rty]
    def cfmt(v):
        if rty == "u64":
            return f"0x{v & 0xFFFFFFFFFFFFFFFF:016x}ULL"
        if rty == "u32":
            return f"0x{v & 0xFFFFFFFF:08x}U"
        return str(v)
    cu.append(f"__device__ const {cty} {name}[{n}] = {{")
    line = "    "
    per = 6 if rty in ("u64", "u32") else 16
    for i, v in enumerate(vals):
        line += cfmt(v) + ", "
        if (i + 1) % per == 0:
            cu.append(line.rstrip()); line = "    "
    if line.strip():
        cu.append(line.rstrip())
    cu.append("};")
    cu.append("")
cdst = "/home/vasis/projects_hobby/vllm-rust/crates/core/kernels/iq_tables.cuh"
open(cdst, "w").write("\n".join(cu) + "\n")
print("wrote", cdst)

for name in WANT:
    rty,n,vals = found[name]
    print(f"  {name}: {rty} x{n}  first={vals[0]} last={vals[-1]}")
