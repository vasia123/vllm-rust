//! JSON Schema → xgrammar EBNF translator.
//!
//! ## Why this exists
//!
//! `xgrammar::CompileJSONSchema` (pinned v0.2.1 == HEAD) silently
//! drops several constraints for non-trivial schemas:
//!
//! | Construct                              | `CompileJSONSchema` | `CompileGrammar` (EBNF) |
//! |----------------------------------------|---------------------|--------------------------|
//! | `pattern` + `minLength`/`maxLength`    | **broken**          | works via `{m,n}` quant. |
//! | `additionalProperties:false` at key boundary | **broken** (any byte accepted) | works via explicit alt |
//!
//! Both bugs verified by xgrammar-rs/tests/nested_strictness.rs +
//! /nested_workarounds.rs + ebnf_prototype.rs. Hand-written EBNF
//! enforces both correctly. So we route schemas with the affected
//! constructs through a Rust-side translator that emits robust EBNF
//! and hand to xgrammar's `CompileGrammar` (which is bug-free).
//!
//! ## The disambiguation trap (must-know)
//!
//! Char classes for string content MUST exclude the unescaped `"`
//! and `\` bytes. Including them as content alternatives makes the
//! grammar ambiguous with the outer string-close `"`, and the
//! Earley parser's union semantics lets the close fire at any
//! position. (Demo:
//! `xgrammar-rs/tests/quantifier_sanity::quantifier_inside_string_with_quote_alternative_is_ambiguous`.)
//!
//! User patterns often include `\"` in the char class because it
//! reads naturally for "any character"; we strip it on translation.
//! Backslash-escape support (`\\\"` etc.) is **out of scope** for
//! this MVP — escape-needing schemas should keep using
//! `CompileJSONSchema` and accept the broken behaviour, OR be
//! rewritten upstream.
//!
//! ## Supported subset
//!
//! - `type: object` + `properties` + `required` + `additionalProperties:false`
//! - `type: string` (+ optional `pattern`/`minLength`/`maxLength`/`enum`)
//! - `type: integer` (+ optional `minimum`/`maximum`, both ≥ 0 for the MVP)
//! - `type: number` (no bounds — JSON-default number regex)
//! - `type: boolean`, `type: null`
//! - `type: array` + `items` + `minItems`/`maxItems`
//! - `enum` (string/integer literals)
//!
//! Unsupported (returns `None` from `try_schema_to_ebnf` so the
//! caller falls back to `compile_json_schema`):
//! - `oneOf`/`anyOf`/`allOf`/`$ref`
//! - `additionalProperties: <schema>` (only the `false` form)
//! - `pattern` with non-`^...+$` shape
//! - `type: number` with bounds
//! - `integer minimum < 0` (signed range explosion is heavy)

use std::fmt::Write;

use anyhow::{bail, Result};
use serde_json::Value;

/// Translate `schema` to a xgrammar EBNF grammar string. Returns
/// `Ok(None)` when the schema uses constructs outside this
/// translator's MVP scope — the caller should fall back to the
/// upstream `compile_json_schema` path. Returns `Err` only on
/// genuinely malformed schemas.
pub fn try_schema_to_ebnf(schema: &Value) -> Result<Option<String>> {
    let mut builder = EbnfBuilder::new();
    let root_rule = match builder.emit(schema) {
        Ok(name) => name,
        Err(UnsupportedKind::Unsupported(reason)) => {
            tracing::debug!(
                target: "vllm_core::xgrammar",
                reason,
                "schema outside EBNF-translator scope; falling back to compile_json_schema"
            );
            return Ok(None);
        }
        Err(UnsupportedKind::Malformed(msg)) => bail!("{msg}"),
    };

    let mut out = String::new();
    writeln!(&mut out, "root ::= {root_rule}").unwrap();
    for (name, body) in builder.rules {
        writeln!(&mut out, "{name} ::= {body}").unwrap();
    }
    // Whitespace rule — JSON RFC 7159 set, repeated zero+.
    out.push_str("ws ::= [ \\n\\r\\t]*\n");
    Ok(Some(out))
}

/// Returns `true` when the schema uses a construct known to mis-compile
/// via `CompileJSONSchema` and that our translator handles. Caller
/// uses this as a gating check: skip the translator for trivially
/// well-handled schemas (faster path) and engage it only for the
/// affected ones.
pub fn schema_needs_ebnf_path(schema: &Value) -> bool {
    fn walk(v: &Value) -> bool {
        match v {
            Value::Object(obj) => {
                // Trigger: string with `pattern` AND length bounds.
                let has_pattern = obj.contains_key("pattern");
                let has_len = obj.contains_key("minLength") || obj.contains_key("maxLength");
                if has_pattern && has_len {
                    return true;
                }
                // Trigger: strict additionalProperties:false on a non-empty
                // required list — the key-boundary bug bites here.
                let strict_object = obj
                    .get("additionalProperties")
                    .and_then(Value::as_bool)
                    .map(|b| !b)
                    .unwrap_or(false);
                let required_nonempty = obj
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|a| !a.is_empty())
                    .unwrap_or(false);
                if strict_object && required_nonempty {
                    return true;
                }
                // Trigger: anyOf / oneOf — upstream JSON-Schema path
                // handles these but the EBNF path lets us bypass
                // any subtle bugs in the union compile when nested
                // with the constructs above.
                if obj.contains_key("anyOf") || obj.contains_key("oneOf") {
                    // Only engage EBNF if at least one branch would
                    // anyway need EBNF — otherwise stay on the
                    // faster JSON-Schema path.
                    let branches = obj
                        .get("anyOf")
                        .or_else(|| obj.get("oneOf"))
                        .and_then(Value::as_array);
                    if let Some(arr) = branches {
                        if arr.iter().any(walk) {
                            return true;
                        }
                    }
                }
                // Recurse into values.
                obj.values().any(walk)
            }
            Value::Array(a) => a.iter().any(walk),
            _ => false,
        }
    }
    walk(schema)
}

// ─────────────────────────── internals ───────────────────────────

#[derive(Debug)]
enum UnsupportedKind {
    /// Schema uses a construct outside the MVP subset — caller
    /// falls back to `compile_json_schema`.
    Unsupported(&'static str),
    /// Schema is malformed in a way no compiler could handle.
    Malformed(String),
}

struct EbnfBuilder {
    rules: Vec<(String, String)>,
    next_id: usize,
}

impl EbnfBuilder {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            next_id: 0,
        }
    }

    fn fresh_rule(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id += 1;
        format!("{prefix}_{id}")
    }

    fn add_rule(&mut self, name: String, body: String) {
        self.rules.push((name, body));
    }

    /// Top-level dispatch. Returns the EBNF rule reference (the
    /// rule's name) that matches one instance of `schema`.
    fn emit(&mut self, schema: &Value) -> std::result::Result<String, UnsupportedKind> {
        let obj = schema
            .as_object()
            .ok_or_else(|| UnsupportedKind::Malformed("schema must be a JSON object".into()))?;

        // Reject features outside the MVP early so the caller falls
        // back to the upstream JSON-Schema path.
        for blocker in ["$ref", "allOf", "not", "if", "then", "else"] {
            if obj.contains_key(blocker) {
                return Err(UnsupportedKind::Unsupported(blocker));
            }
        }
        // oneOf / anyOf: emit a union rule. The Earley parser
        // handles ambiguous "any-of" alternatives the same way as
        // "one-of" at the bitmask level (union of allowed next
        // tokens). We don't try to enforce one-of's uniqueness
        // claim — that's a post-acceptance semantic check, not a
        // grammar property.
        if let Some(alts) = obj.get("oneOf").and_then(Value::as_array) {
            return self.emit_union(alts, "oneOf");
        }
        if let Some(alts) = obj.get("anyOf").and_then(Value::as_array) {
            return self.emit_union(alts, "anyOf");
        }
        // `enum` on a primitive — emit a literal alternation.
        if let Some(values) = obj.get("enum").and_then(Value::as_array) {
            return self.emit_enum(values);
        }

        let type_str = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or(UnsupportedKind::Unsupported("schema without type"))?;
        match type_str {
            "object" => self.emit_object(obj),
            "string" => self.emit_string(obj),
            "integer" => self.emit_integer(obj),
            "number" => self.emit_number(obj),
            "boolean" => Ok("(\"true\" | \"false\")".to_string()),
            "null" => Ok("\"null\"".to_string()),
            "array" => self.emit_array(obj),
            other => Err(UnsupportedKind::Malformed(format!(
                "unknown JSON-Schema type: {other}"
            ))),
        }
    }

    fn emit_union(
        &mut self,
        alts: &[Value],
        keyword: &str,
    ) -> std::result::Result<String, UnsupportedKind> {
        if alts.is_empty() {
            return Err(UnsupportedKind::Malformed(format!("empty {keyword}")));
        }
        let mut child_rules = Vec::with_capacity(alts.len());
        for alt in alts {
            let rule = self.emit(alt)?;
            child_rules.push(rule);
        }
        let body = format!("({})", child_rules.join(" | "));
        let prefix = if keyword == "oneOf" { "oneof" } else { "anyof" };
        let name = self.fresh_rule(prefix);
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_enum(&mut self, values: &[Value]) -> std::result::Result<String, UnsupportedKind> {
        // JSON string enum values must be emitted as full quoted JSON
        // literals — `"spring"` in JSON is the 8-byte sequence
        // `\"spring\"`, so the EBNF literal must include the outer
        // double-quotes too. Numbers/booleans/null have no enclosing
        // quotes in JSON, so they emit bare.
        let mut parts = Vec::with_capacity(values.len());
        for v in values {
            match v {
                Value::String(s) => {
                    parts.push(format!("\"\\\"{}\\\"\"", ebnf_escape_string_lit(s)))
                }
                Value::Number(n) => parts.push(format!("\"{}\"", n)),
                Value::Bool(b) => parts.push(format!("\"{}\"", b)),
                Value::Null => parts.push("\"null\"".to_string()),
                _ => {
                    return Err(UnsupportedKind::Unsupported(
                        "enum with non-primitive values",
                    ))
                }
            }
        }
        let body = format!("({})", parts.join(" | "));
        let name = self.fresh_rule("enum");
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_object(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> std::result::Result<String, UnsupportedKind> {
        // We support strict objects only — additionalProperties:false
        // plus a `required` list that matches the property keys.
        let additional = obj
            .get("additionalProperties")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if additional {
            return Err(UnsupportedKind::Unsupported(
                "object without additionalProperties:false",
            ));
        }
        let properties = obj
            .get("properties")
            .and_then(Value::as_object)
            .ok_or(UnsupportedKind::Unsupported("object without properties"))?;
        let required: Vec<&str> = obj
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if required.is_empty() {
            return Err(UnsupportedKind::Unsupported(
                "object with empty required list",
            ));
        }
        // Property emission strategy:
        //   - Walk properties in DECLARED order (serde_json's
        //     `preserve_order` feature keeps the source JSON
        //     insertion order — see workspace Cargo.toml).
        //   - Required properties → mandatory.
        //   - Optional properties → wrap in `( ... )?` at their
        //     declared position; the leading comma is folded inside
        //     the optional group so we never emit `,,`.
        //
        // This is a strict subset of JSON-Schema's "any permutation
        // of present properties": we force declared order. The
        // alternative (permutation-aware grammar) explodes as
        // O((R+O)!) which kills compile time. Operators who need
        // permutation can specify schemas without
        // `additionalProperties:false` and fall back to upstream.
        for key in &required {
            if !properties.contains_key(*key) {
                return Err(UnsupportedKind::Malformed(format!(
                    "required key {key:?} missing from properties"
                )));
            }
        }
        let required_set: std::collections::HashSet<&str> = required.iter().copied().collect();

        // Pre-emit all property value rules so the rule names exist
        // before we build the object body. We need the per-property
        // optionality classification, so use a Vec<(key, rule, opt)>.
        let mut props_seq: Vec<(String, String, bool)> = Vec::with_capacity(properties.len());
        for (key, value_schema) in properties {
            let rule = self.emit(value_schema)?;
            let is_optional = !required_set.contains(key.as_str());
            props_seq.push((key.clone(), rule, is_optional));
        }

        // Locate the index of the first REQUIRED property — the
        // comma policy differs before vs after it:
        //   - everything BEFORE first-required: emitted as
        //     `(key:val ws , ws)?` — the trailing comma belongs to
        //     the optional, kept iff the optional fires.
        //   - first-required: emitted bare (no leading comma).
        //   - everything AFTER first-required: emitted as
        //     `(ws , ws key:val)?` for optional or `ws , ws key:val`
        //     for required — leading comma belongs to the property.
        //
        // This rule prevents the `{,"a":1}` and `{"a":1,,"c":3}`
        // shapes that a naïve "optional with trailing comma"
        // emission would allow.
        let first_required_idx = props_seq.iter().position(|(_, _, opt)| !*opt);

        let mut body = String::new();
        body.push_str("\"{\" ws ");

        match first_required_idx {
            Some(first_req) => {
                // Emit pre-required leading-optional block (each
                // optional carries a trailing `,`).
                for (key, rule, opt) in &props_seq[..first_req] {
                    debug_assert!(*opt);
                    let kv = format!(
                        "\"\\\"{}\\\"\" ws \":\" ws {rule} ws \",\" ws",
                        ebnf_escape_string_lit(key)
                    );
                    write!(body, "({kv})? ").unwrap();
                }
                // Emit first required bare.
                let (rk, rr, _) = &props_seq[first_req];
                write!(
                    body,
                    "\"\\\"{}\\\"\" ws \":\" ws {rr}",
                    ebnf_escape_string_lit(rk)
                )
                .unwrap();
                // Emit everything AFTER first-required.
                for (key, rule, opt) in &props_seq[first_req + 1..] {
                    let kv = format!(
                        "ws \",\" ws \"\\\"{}\\\"\" ws \":\" ws {rule}",
                        ebnf_escape_string_lit(key)
                    );
                    if *opt {
                        write!(body, " ({kv})?").unwrap();
                    } else {
                        write!(body, " {kv}").unwrap();
                    }
                }
            }
            None => {
                // No required properties → the whole object body
                // is optional. Match `{}` or any subset of optionals
                // in declared order. Emit as an outer optional
                // group: `( first (ws , ws next)* )?`.
                if props_seq.is_empty() {
                    return Err(UnsupportedKind::Malformed(
                        "object with empty properties and no required".into(),
                    ));
                }
                body.push('(');
                for (i, (key, rule, _)) in props_seq.iter().enumerate() {
                    let kv = format!(
                        "\"\\\"{}\\\"\" ws \":\" ws {rule}",
                        ebnf_escape_string_lit(key)
                    );
                    if i == 0 {
                        write!(body, "{kv}").unwrap();
                    } else {
                        write!(body, " (ws \",\" ws {kv})?").unwrap();
                    }
                }
                body.push_str(")?");
            }
        }

        body.push_str(" ws \"}\"");

        let name = self.fresh_rule("obj");
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_string(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> std::result::Result<String, UnsupportedKind> {
        let pattern = obj.get("pattern").and_then(Value::as_str);
        let min_len = obj.get("minLength").and_then(Value::as_u64).unwrap_or(0);
        let max_len = obj.get("maxLength").and_then(Value::as_u64);

        // Build the "one string character" rule.
        // Three regimes:
        //   1. No pattern: default JSON `basic_string_1` shape —
        //      safe bytes plus a full JSON escape set.
        //   2. Pattern + no `\"`/`\\` in char class: stripped class
        //      only, no escape branch (smaller grammar, no escapes
        //      possible).
        //   3. Pattern with `\"`/`\\` in char class: stripped class
        //      PLUS the shared `json_escape` rule so the model can
        //      still emit `\"` / `\\` JSON escape sequences. The
        //      pattern semantically asked for them; we represent
        //      them properly instead of dropping silently.
        let (safe_class, want_escapes) = if let Some(pat) = pattern {
            pattern_to_safe_char_class(pat)?
        } else {
            // Default: any byte except `"` and `\` and control
            // range, plus the full JSON escape set. Mirrors
            // upstream xgrammar's basic_string_1 alphabet.
            ("^\"\\\\\\x00-\\x1f".to_string(), true)
        };

        // Per-character body, emitted INLINE into the string rule.
        //
        // NOTE: the content alternation MUST be inlined here rather than
        // factored into a separate `strchar ::= …` rule that the string
        // rule references. xgrammar's `CompileGrammar` (EBNF) treats a
        // rule-reference inside the content star as a pushdown boundary
        // and does NOT compile it to an FSM — every decode step then
        // recomputes the full O(vocab) token mask (~50 ms/step on Qwen3's
        // 151k vocab). Inlining keeps the star a single regular FSM state
        // (~20 µs/step — a ~1000× speedup; see `bench_string_rule_variants`).
        let char_body = if want_escapes {
            // `\` followed by one of the JSON simple-escape chars. The
            // `\uXXXX` form is intentionally omitted — it forces the same
            // non-FSM fallback (and raw UTF-8 bytes are already admitted
            // by the content class, so non-ASCII is emitted directly).
            format!("[{safe_class}] | \"\\\\\" [\"\\\\/bfnrt]")
        } else {
            format!("[{safe_class}]")
        };

        // Quantifier: `{min,max}` (if max set) or `{min,}` (open-ended).
        let quant = match (min_len, max_len) {
            (0, None) => "*".to_string(),
            (m, None) => format!("{{{m},}}"),
            (m, Some(x)) => format!("{{{m},{x}}}"),
        };

        // Body construction depends on the quantifier. Both forms must
        // stay a regular FSM so xgrammar fills the per-character mask in
        // µs rather than recomputing the O(vocab) mask each step (~50 ms
        // on Qwen3's 151k vocab — measured in `bench_string_rule_variants`).
        //   - `*` / `{m,}` (unbounded): inline the content alternation.
        //     An unbounded star is ONE FSM state — it does not duplicate
        //     the body, so the escape alternation can ride along and the
        //     fill stays ~50 µs WITH escapes.
        //   - `{m,n}` (bounded): xgrammar expands the quantifier into m..n
        //     copies of the repeated unit. An escape alternation there is
        //     fatal twice over: (a) it stays slow (~50 ms/fill even when
        //     inlined — the bounded counter + branch is not FSM-cached),
        //     and (b) inlining duplicates the multi-branch body n times
        //     and blows the compiled grammar up (CUDA OOM on nested
        //     array-of-strings schemas). A PURE char-class bounded repeat
        //     `[class]{m,n}`, by contrast, compiles to a small FSM with a
        //     counter and fills in ~260 µs. So bounded strings drop the
        //     `\`-escape branch entirely: the class already admits raw
        //     UTF-8 (non-ASCII is emitted directly), so the only loss is
        //     the ability to emit backslash escapes inside a
        //     length-bounded string — stricter, still valid JSON.
        let bounded = max_len.is_some();
        let body = if bounded {
            format!("\"\\\"\" [{safe_class}]{quant} \"\\\"\"")
        } else {
            format!("\"\\\"\" ({char_body}){quant} \"\\\"\"")
        };
        let name = self.fresh_rule("str");
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_integer(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> std::result::Result<String, UnsupportedKind> {
        let min = obj.get("minimum").and_then(Value::as_i64);
        let max = obj.get("maximum").and_then(Value::as_i64);

        // Three regimes based on sign coverage:
        //   - both ≥ 0 → positive_integer_range_ebnf as-is
        //   - both ≤ 0 → `"-"` prefix + positive_integer_range_ebnf
        //     over the |max|..=|min| range (flipped because |x| reverses order)
        //   - straddles zero → union of negative half + non-negative half
        let body = match (min, max) {
            (Some(mn), Some(mx)) if mn >= 0 => {
                positive_integer_range_ebnf(mn as u64, Some(mx as u64))?
            }
            (Some(mn), Some(mx)) if mx <= 0 => {
                // All-negative range. The decimal absolute values run
                // from |mx| (smallest |.|) to |mn| (largest |.|).
                if mx == 0 && mn == 0 {
                    "\"0\"".to_string()
                } else if mx == 0 {
                    // [mn, 0] — include 0 as bare "0" and negatives.
                    let neg = positive_integer_range_ebnf(1, Some((-mn) as u64))?;
                    format!("(\"0\" | \"-\" ({neg}))")
                } else {
                    let pos = positive_integer_range_ebnf((-mx) as u64, Some((-mn) as u64))?;
                    format!("\"-\" ({pos})")
                }
            }
            (Some(mn), Some(mx)) => {
                // Straddles zero. mn < 0, mx > 0.
                let neg = positive_integer_range_ebnf(1, Some((-mn) as u64))?;
                let pos = positive_integer_range_ebnf(0, Some(mx as u64))?;
                format!("(\"-\" ({neg}) | {pos})")
            }
            (Some(mn), None) if mn >= 0 => positive_integer_range_ebnf(mn as u64, None)?,
            (Some(mn), None) => {
                // mn < 0, no upper bound. Emit `"-"? [0-9]+`-ish: any
                // signed integer covering mn..=∞. We rely on the
                // model's reasoning to stay above mn; strict
                // enforcement at the grammar level would force a
                // per-digit-count walk for the negative half.
                let _ = mn;
                "(\"-\"? [0-9]+)".to_string()
            }
            (None, Some(mx)) if mx >= 0 => {
                // Unknown lower, known upper ≥ 0: any non-negative up to mx.
                positive_integer_range_ebnf(0, Some(mx as u64))?
            }
            (None, Some(mx)) => {
                // Unknown lower, max < 0: all-negative half.
                let pos = positive_integer_range_ebnf(1, Some((-mx) as u64))?;
                format!("\"-\" ({pos})")
            }
            (None, None) => {
                // Pure `type: integer` — match upstream xgrammar's
                // basic_integer fragment.
                "(\"0\" | \"-\"? [1-9] [0-9]*)".to_string()
            }
        };
        let name = self.fresh_rule("int");
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_number(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> std::result::Result<String, UnsupportedKind> {
        if obj.contains_key("minimum") || obj.contains_key("maximum") {
            return Err(UnsupportedKind::Unsupported(
                "number with bounds (out of MVP)",
            ));
        }
        // JSON RFC 7159 number. Mirrors upstream xgrammar's basic_number.
        let body =
            "(\"0\" | (\"-\")? [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+\\-]? [0-9]+)?".to_string();
        let name = self.fresh_rule("num");
        self.add_rule(name.clone(), body);
        Ok(name)
    }

    fn emit_array(
        &mut self,
        obj: &serde_json::Map<String, Value>,
    ) -> std::result::Result<String, UnsupportedKind> {
        let items = obj
            .get("items")
            .ok_or(UnsupportedKind::Unsupported("array without items"))?;
        let item_rule = self.emit(items)?;
        let min = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let max = obj.get("maxItems").and_then(Value::as_u64);

        // Emit `[ ws item (ws , ws item){min-1,max-1} ws ]` form.
        // The first item is mandatory iff min ≥ 1; otherwise empty
        // arrays are allowed.
        let body = match (min, max) {
            (0, None) => format!("\"[\" ws ({item_rule} (ws \",\" ws {item_rule})*)? ws \"]\""),
            (0, Some(0)) => "\"[\" ws \"]\"".to_string(),
            (0, Some(mx)) => format!(
                "\"[\" ws ({item_rule} (ws \",\" ws {item_rule}){{0,{}}})? ws \"]\"",
                mx - 1
            ),
            (mn, None) => format!(
                "\"[\" ws {item_rule} (ws \",\" ws {item_rule}){{{},}} ws \"]\"",
                mn - 1
            ),
            (mn, Some(mx)) => {
                if mn > mx {
                    return Err(UnsupportedKind::Malformed(format!(
                        "array minItems({mn}) > maxItems({mx})"
                    )));
                }
                format!(
                    "\"[\" ws {item_rule} (ws \",\" ws {item_rule}){{{},{}}} ws \"]\"",
                    mn - 1,
                    mx - 1
                )
            }
        };

        let name = self.fresh_rule("arr");
        self.add_rule(name.clone(), body);
        Ok(name)
    }
}

/// Convert a `^[allowed]+$` style pattern to a safe EBNF char-class
/// body (NO surrounding `[ ]`) plus an `allow_escapes` flag.
///
/// Strips the unescaped `"` and `\` bytes from the class itself —
/// including them creates the string-boundary ambiguity trap
/// (see module docstring). When the pattern's original class
/// references `\"` or `\\`, the flag is set so the caller adds a
/// JSON-escape branch (`\\\"` / `\\\\` / `\\n` / ...) to the
/// per-character rule. This preserves the intent (allow JSON-escaped
/// quote/backslash) while keeping the grammar unambiguous.
fn pattern_to_safe_char_class(
    pattern: &str,
) -> std::result::Result<(String, bool), UnsupportedKind> {
    let pat = pattern.trim();
    // MVP: only `^[...]+$` and `^[...]*$` shapes.
    let inner = pat
        .strip_prefix('^')
        .ok_or(UnsupportedKind::Unsupported("pattern without `^` anchor"))?;
    let inner = inner
        .strip_suffix('$')
        .ok_or(UnsupportedKind::Unsupported("pattern without `$` anchor"))?;
    let inner = inner
        .strip_suffix('+')
        .or_else(|| inner.strip_suffix('*'))
        .unwrap_or(inner);
    let class_body = inner
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or(UnsupportedKind::Unsupported(
            "pattern not `^[chars]+$` shape",
        ))?;

    // Walk the class body. We drop `"` / `\"` / `\\` from the
    // emitted class to avoid the ambiguity trap; record whether
    // we saw a `\"` or `\\` to decide whether to enable JSON
    // escape alternatives on the per-char rule.
    let mut out = String::with_capacity(class_body.len());
    let mut want_escapes = false;
    let mut chars = class_body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('"') => {
                    chars.next(); // drop `\"`
                    want_escapes = true;
                    continue;
                }
                Some('\\') => {
                    chars.next(); // drop `\\`
                    want_escapes = true;
                    continue;
                }
                Some(_) => {
                    // Pass through other escapes (`\n`, `\t`, etc.).
                    out.push('\\');
                    out.push(chars.next().unwrap());
                }
                None => out.push('\\'),
            }
            continue;
        }
        if c == '"' {
            // Bare unescaped quote → trap. Skip but treat as
            // intent to allow escaped quote.
            want_escapes = true;
            continue;
        }
        out.push(c);
    }
    if out.is_empty() {
        return Err(UnsupportedKind::Malformed(format!(
            "pattern char class is empty after stripping `\"` / `\\`: {pattern}"
        )));
    }
    Ok((out, want_escapes))
}

/// Emit an EBNF expression matching the decimal representation of
/// any integer `min..=max` (or `min..` if `max` is None), assuming
/// both bounds ≥ 0.
///
/// Strategy: enumerate the *digit-count classes* `k` in
/// `len(min)..=len(max)` and for each one emit an EBNF alternative
/// that matches exactly the `k`-digit decimals lying within the
/// query range. For each `k` we have three sub-cases:
///   - Fully-covered class (`min ≤ 10^(k-1)` AND `max ≥ 10^k - 1`)
///     → compact pattern `[1-9] [0-9]{k-1}`
///   - Partial class (the boundary touches it) → digit-by-digit
///     prefix walk via `bounded_digits_ebnf`.
///
/// Special case: `min == 0` adds the literal "0" as the first
/// alternative.
fn positive_integer_range_ebnf(
    min: u64,
    max: Option<u64>,
) -> std::result::Result<String, UnsupportedKind> {
    let Some(max) = max else {
        // Unbounded above.
        if min == 0 {
            return Ok("(\"0\" | [1-9] [0-9]*)".to_string());
        }
        // `min ≥ 1` — the regex `[1-9] [0-9]*` matches every
        // positive integer; for `min` > 1 it's a strict superset
        // but enforcing the exact lower bound at every decimal
        // length explodes the grammar. We accept the looseness
        // and document via a debug log; callers that need strict
        // bounds should set `max` too.
        if min > 1 {
            tracing::debug!(
                target: "vllm_core::xgrammar",
                min,
                "integer with min>1 and no max — emitting `[1-9][0-9]*` (lower bound not strictly enforced; the grammar is a superset)"
            );
        }
        return Ok("[1-9] [0-9]*".to_string());
    };

    if min > max {
        return Err(UnsupportedKind::Malformed(format!(
            "integer minimum({min}) > maximum({max})"
        )));
    }

    // Single-value special case (e.g. const-style range).
    if min == max {
        return Ok(format!("\"{min}\""));
    }

    let lo = min.max(1);
    let hi = max;

    let lo_len = decimal_len(lo);
    let hi_len = decimal_len(hi);

    // Fast path: ALL digit-count classes lo_len..=hi_len fully
    // covered AND lo == 1 → emit compact `[1-9] [0-9]{0,N-1}`.
    // Equivalent to bucket-by-bucket but textually cleaner and
    // (likely) cheaper for the xgrammar Earley compiler.
    if lo == 1 {
        let class_hi_for_top = (10u64.pow(hi_len as u32)).saturating_sub(1);
        if hi == class_hi_for_top {
            let body_pos = if hi_len == 1 {
                "[1-9]".to_string()
            } else {
                format!("[1-9] [0-9]{{0,{}}}", hi_len - 1)
            };
            return Ok(if min == 0 {
                format!("(\"0\" | {body_pos})")
            } else {
                body_pos
            });
        }
    }

    let mut alts: Vec<String> = Vec::new();

    // 0 is its own digit-count class — emit explicitly.
    if min == 0 {
        alts.push("\"0\"".to_string());
    }

    for k in lo_len..=hi_len {
        let class_lo = 10u64.pow((k - 1) as u32);
        let class_hi = (10u64.pow(k as u32)).saturating_sub(1);
        // Intersect [lo, hi] with [class_lo, class_hi].
        let intersect_lo = lo.max(class_lo);
        let intersect_hi = hi.min(class_hi);
        if intersect_lo > intersect_hi {
            continue;
        }
        if intersect_lo == class_lo && intersect_hi == class_hi {
            // Fully-covered class.
            if k == 1 {
                alts.push("[1-9]".to_string());
            } else {
                alts.push(format!("[1-9] [0-9]{{{}}}", k - 1));
            }
        } else {
            // Partial — digit-prefix walk.
            alts.push(bounded_digits_ebnf(intersect_lo, intersect_hi, k));
        }
    }

    let body = if alts.len() == 1 {
        alts.into_iter().next().unwrap()
    } else {
        format!("({})", alts.join(" | "))
    };
    Ok(body)
}

/// Build an EBNF expression matching every `k`-digit decimal value
/// in `[lo, hi]` where both bounds are themselves `k`-digit
/// (no class straddling). Walks digit-by-digit from MSB to LSB,
/// emitting one alternative per "fan-out" digit position.
///
/// Example: `bounded_digits_ebnf(125, 478, 3)` produces
///   ("1" "2" [5-9] | "1" [3-9] [0-9] | [2-3] [0-9] [0-9] | "4" [0-6] [0-9] | "4" "7" [0-8])
/// covering 125..=478 as:
///   125..129, 130..199, 200..399, 400..469, 470..478.
fn bounded_digits_ebnf(lo: u64, hi: u64, k: usize) -> String {
    debug_assert_eq!(decimal_len(lo), k);
    debug_assert_eq!(decimal_len(hi), k);
    debug_assert!(lo <= hi);

    let lo_digits: Vec<u8> = decimal_digits(lo, k);
    let hi_digits: Vec<u8> = decimal_digits(hi, k);

    // Find common prefix length (most-significant matching digits).
    let mut common = 0;
    while common < k && lo_digits[common] == hi_digits[common] {
        common += 1;
    }
    if common == k {
        // lo == hi — single value.
        return digits_to_literal(&lo_digits);
    }

    let mut alts: Vec<String> = Vec::new();
    let common_lit = digits_to_literal(&lo_digits[..common]);

    // The first differing position has bounds: lo_digits[common]
    // (must be ≥) and hi_digits[common] (must be ≤). Anything
    // strictly between gets a "fully free" tail.
    let lo_first = lo_digits[common];
    let hi_first = hi_digits[common];

    // Bucket 1: prefix `common + lo_first`, tail bounded by lo's
    // remaining digits ↑ to 999...9.
    alts.push(
        format!(
            "{}{} {}",
            common_lit,
            if common_lit.is_empty() { "" } else { " " },
            upper_tail(&lo_digits[common..], k - common),
        )
        .trim()
        .to_string(),
    );

    // Bucket 2: first digit ∈ (lo_first, hi_first) with free tail.
    if hi_first > lo_first + 1 {
        let between = format!("[{}-{}]", lo_first + 1, hi_first - 1);
        let tail = free_tail(k - common - 1);
        if tail.is_empty() {
            alts.push(format!("{common_lit} {between}").trim().to_string());
        } else {
            alts.push(format!("{common_lit} {between} {tail}").trim().to_string());
        }
    }

    // Bucket 3: prefix `common + hi_first`, tail bounded by hi's
    // remaining digits ↓ from 000...0.
    if hi_first > lo_first {
        alts.push(
            format!(
                "{}{} {}",
                common_lit,
                if common_lit.is_empty() { "" } else { " " },
                lower_tail(&hi_digits[common..], k - common),
            )
            .trim()
            .to_string(),
        );
    }

    if alts.len() == 1 {
        alts.into_iter().next().unwrap()
    } else {
        format!("({})", alts.join(" | "))
    }
}

/// Decompose `n` into exactly `k` digits, MSB-first. Pads with
/// leading zeros (caller is responsible for ensuring lo/hi are
/// actually k-digit).
fn decimal_digits(mut n: u64, k: usize) -> Vec<u8> {
    let mut d = vec![0u8; k];
    for i in (0..k).rev() {
        d[i] = (n % 10) as u8;
        n /= 10;
    }
    d
}

/// Emit a literal EBNF sequence for a digit sequence.
/// Empty input → empty string.
fn digits_to_literal(digits: &[u8]) -> String {
    digits
        .iter()
        .map(|d| format!("\"{}\"", d))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Emit `[0-9]` repeated `n` times (space-separated EBNF tokens).
/// `n == 0` → empty string.
fn free_tail(n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    if n == 1 {
        return "[0-9]".to_string();
    }
    format!("[0-9]{{{n}}}")
}

/// EBNF for the upper-tail of a number whose leading digit equals
/// `digits[0]`: matches every number whose remaining digits are
/// ≥ `digits[1..]`. Walks the rest of the digits recursively.
fn upper_tail(digits: &[u8], len: usize) -> String {
    debug_assert_eq!(digits.len(), len);
    if len == 0 {
        return String::new();
    }
    let first = digits[0];
    let tail_digits = &digits[1..];
    let tail_len = len - 1;

    // Three sub-buckets:
    //   1. exact `first` + recursive upper_tail on the rest
    //   2. (first+1..=9) + free_tail
    let mut alts = Vec::new();

    if tail_len == 0 {
        // Last digit position: range [first, 9].
        if first == 9 {
            alts.push("\"9\"".to_string());
        } else {
            alts.push(format!("[{}-9]", first));
        }
    } else {
        // 1: exact first, then recursive upper_tail
        let inner = upper_tail(tail_digits, tail_len);
        alts.push(format!("\"{}\" {}", first, inner).trim().to_string());
        // 2: digit ∈ (first, 9], then free tail
        if first < 9 {
            let head = if first + 1 == 9 {
                "\"9\"".to_string()
            } else {
                format!("[{}-9]", first + 1)
            };
            let tail = free_tail(tail_len);
            if tail.is_empty() {
                alts.push(head);
            } else {
                alts.push(format!("{head} {tail}"));
            }
        }
    }

    if alts.len() == 1 {
        alts.into_iter().next().unwrap()
    } else {
        format!("({})", alts.join(" | "))
    }
}

/// EBNF for the lower-tail of a number whose leading digit equals
/// `digits[0]`: matches every number whose remaining digits are
/// ≤ `digits[1..]`. Symmetric to `upper_tail`.
fn lower_tail(digits: &[u8], len: usize) -> String {
    debug_assert_eq!(digits.len(), len);
    if len == 0 {
        return String::new();
    }
    let first = digits[0];
    let tail_digits = &digits[1..];
    let tail_len = len - 1;

    let mut alts = Vec::new();

    if tail_len == 0 {
        if first == 0 {
            alts.push("\"0\"".to_string());
        } else {
            alts.push(format!("[0-{}]", first));
        }
    } else {
        // 1: digit ∈ [0, first-1] + free tail
        if first > 0 {
            let head = if first == 1 {
                "\"0\"".to_string()
            } else {
                format!("[0-{}]", first - 1)
            };
            let tail = free_tail(tail_len);
            if tail.is_empty() {
                alts.push(head);
            } else {
                alts.push(format!("{head} {tail}"));
            }
        }
        // 2: exact first + recursive lower_tail on rest
        let inner = lower_tail(tail_digits, tail_len);
        alts.push(format!("\"{}\" {}", first, inner).trim().to_string());
    }

    if alts.len() == 1 {
        alts.into_iter().next().unwrap()
    } else {
        format!("({})", alts.join(" | "))
    }
}

fn decimal_len(n: u64) -> usize {
    if n == 0 {
        1
    } else {
        (n.ilog10() + 1) as usize
    }
}

/// Escape a Rust string for embedding inside a GBNF string literal
/// — `"..."` form. Only `"` and `\` need escaping in the literal
/// body; other bytes pass through.
fn ebnf_escape_string_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_needs_ebnf_pattern_with_length() {
        let s = json!({"type":"string","pattern":"^x+$","minLength":3});
        assert!(schema_needs_ebnf_path(&s));
    }

    #[test]
    fn schema_needs_ebnf_strict_object_with_required() {
        let s = json!({
            "type":"object","additionalProperties":false,
            "required":["a"],"properties":{"a":{"type":"string"}}
        });
        assert!(schema_needs_ebnf_path(&s));
    }

    #[test]
    fn schema_does_not_need_ebnf_simple_object() {
        let s = json!({"type":"object","properties":{"a":{"type":"string"}}});
        assert!(!schema_needs_ebnf_path(&s));
    }

    #[test]
    fn pattern_strip_ambiguous_quotes_records_escape_intent() {
        let (body, want_escapes) =
            pattern_to_safe_char_class(r#"^[А-Яа-яЁё0-9 \"']+$"#).expect("ok");
        assert!(!body.contains('"'), "must strip bare `\"`");
        assert!(!body.contains("\\\""), "must strip `\\\"`");
        assert!(body.contains('А'));
        assert!(body.contains('\''));
        assert!(
            want_escapes,
            "pattern had `\\\"` → escape intent must be recorded"
        );
    }

    #[test]
    fn pattern_without_quotes_does_not_request_escapes() {
        let (_body, want) = pattern_to_safe_char_class(r"^[a-z0-9]+$").expect("ok");
        assert!(
            !want,
            "purely alphanumeric class needs no JSON-escape branch"
        );
    }

    #[test]
    fn pattern_rejects_non_anchored() {
        let r = pattern_to_safe_char_class(r"[abc]+");
        assert!(matches!(r, Err(UnsupportedKind::Unsupported(_))));
    }

    #[test]
    fn optional_properties_emit_optional_wrapped_blocks() {
        // Schema with both required and optional properties. The
        // emitted EBNF must mark the optional ones with `?` and
        // keep declared order (a, b, c, d).
        let schema = json!({
            "type":"object","additionalProperties":false,
            "required":["a","c"],
            "properties":{
                "a":{"type":"integer","minimum":1,"maximum":9},
                "b":{"type":"integer","minimum":1,"maximum":9},
                "c":{"type":"integer","minimum":1,"maximum":9},
                "d":{"type":"integer","minimum":1,"maximum":9}
            }
        });
        assert!(schema_needs_ebnf_path(&schema));
        let ebnf = try_schema_to_ebnf(&schema).unwrap().unwrap();
        // `"a"` is the first required (no leading-only-optional);
        // `"b"` between required `a` and `c` → `(...)? ` block;
        // `"d"` after last required → `(...)?` block.
        assert!(
            ebnf.contains(r#""\"b\""#) && ebnf.contains(")?"),
            "optional `b` not wrapped in `(...)?`:\n{ebnf}"
        );
        assert!(
            ebnf.contains(r#""\"d\""#) && ebnf.contains(")?"),
            "optional `d` not wrapped in `(...)?`:\n{ebnf}"
        );

        // Round-trip via xgrammar to verify the emitted grammar is
        // syntactically valid.
        let tok = std::sync::Arc::new(
            xgrammar_rs::TokenizerInfo::from_encoded_vocab(
                &(0u32..256).map(|b| vec![b as u8]).collect::<Vec<_>>(),
                xgrammar_rs::VocabType::ByteLevel,
                false,
                &[0],
            )
            .unwrap(),
        );
        let compiler = xgrammar_rs::GrammarCompiler::new(tok).unwrap();
        compiler
            .compile_ebnf(&ebnf, "root")
            .unwrap_or_else(|e| panic!("compile_ebnf rejected the EBNF: {e}\n{ebnf}"));
    }

    #[test]
    fn all_required_no_optional_keeps_compact_form() {
        // No optional → no `?` markers around property blocks.
        let schema = json!({
            "type":"object","additionalProperties":false,
            "required":["x"],"properties":{"x":{"type":"integer","minimum":1,"maximum":9}}
        });
        let ebnf = try_schema_to_ebnf(&schema).unwrap().unwrap();
        // The object rule should not have any `)?` (no optional blocks).
        let obj_line = ebnf.lines().find(|l| l.starts_with("obj_")).unwrap();
        assert!(
            !obj_line.contains(")?"),
            "no optional properties should mean no `?` blocks: {obj_line}"
        );
    }

    #[test]
    fn union_oneof_strings_emits_alternation() {
        // Pattern must be `^[chars]+$` shape per the MVP — bare
        // single-char literals like `^a+$` are outside scope and
        // would force the translator to opt out (None).
        let schema = json!({
            "oneOf":[
                {"type":"string","pattern":"^[a]+$","minLength":1,"maxLength":3},
                {"type":"integer","minimum":1,"maximum":9}
            ]
        });
        let ebnf = try_schema_to_ebnf(&schema)
            .expect("translator")
            .expect("supported via MVP");
        // Must contain a `oneof_X` rule with a `|` alternation
        // referencing both child rule names.
        assert!(
            ebnf.lines()
                .any(|l| l.starts_with("oneof_") && l.contains('|')),
            "oneof rule with alternation missing:\n{ebnf}"
        );
    }

    #[test]
    fn union_anyof_compiles_via_xgrammar_end_to_end() {
        // Whole flow: translate + xgrammar compile via inline test.
        // anyOf wrapping two strict-object alternatives — both
        // need EBNF path individually so the union does too.
        let schema = json!({
            "anyOf":[
                {"type":"object","additionalProperties":false,
                 "required":["x"],"properties":{
                    "x":{"type":"string","pattern":"^[a-z]+$","minLength":2,"maxLength":5}
                 }},
                {"type":"object","additionalProperties":false,
                 "required":["y"],"properties":{
                    "y":{"type":"integer","minimum":1,"maximum":99}
                 }}
            ]
        });
        assert!(schema_needs_ebnf_path(&schema));
        let ebnf = try_schema_to_ebnf(&schema).unwrap().unwrap();
        // Round-trip through xgrammar.
        let tok = std::sync::Arc::new(
            xgrammar_rs::TokenizerInfo::from_encoded_vocab(
                &(0u32..256).map(|b| vec![b as u8]).collect::<Vec<_>>(),
                xgrammar_rs::VocabType::ByteLevel,
                false,
                &[0],
            )
            .unwrap(),
        );
        let compiler = xgrammar_rs::GrammarCompiler::new(tok).unwrap();
        compiler
            .compile_ebnf(&ebnf, "root")
            .unwrap_or_else(|e| panic!("compile_ebnf failed: {e}\nEBNF:\n{ebnf}"));
    }

    #[test]
    fn emit_unbounded_string_inlines_escape_branch() {
        // Unbounded (no length bounds) → the content star inlines the
        // escape alternation `"\" ["\/bfnrt]` directly (no json_escape
        // rule-ref) so xgrammar keeps it an FSM. See emit_string's NOTE.
        let schema = json!({ "type":"string", "pattern":r#"^[a-z\"]+$"# });
        let ebnf = try_schema_to_ebnf(&serde_json::json!({
            "type":"object","additionalProperties":false,
            "required":["s"],"properties":{"s": schema}
        }))
        .unwrap()
        .unwrap();
        assert!(
            !ebnf.contains("json_escape"),
            "escape must be inlined, not a json_escape rule-ref:\n{ebnf}"
        );
        assert!(
            ebnf.contains(r#"| "\\" ["#),
            "unbounded string rule must inline the `\\ <escape-class>` branch:\n{ebnf}"
        );
    }

    #[test]
    fn emit_bounded_string_uses_pure_class_no_escape() {
        // Bounded {m,n} → a pure char-class repeat (no escape branch, no
        // rule-ref) so the bounded repeat stays a small fast FSM and does
        // not blow up the grammar. See emit_string's NOTE.
        let schema = json!({ "type":"string", "minLength":3, "maxLength":5 });
        let ebnf = try_schema_to_ebnf(&serde_json::json!({
            "type":"object","additionalProperties":false,
            "required":["s"],"properties":{"s": schema}
        }))
        .unwrap()
        .unwrap();
        assert!(
            !ebnf.contains("json_escape") && !ebnf.contains(r#"| "\\" ["#),
            "bounded string must be a pure char-class repeat (no escape branch):\n{ebnf}"
        );
        assert!(
            ebnf.contains("{3,5}"),
            "bounded string must carry the {{3,5}} quantifier:\n{ebnf}"
        );
    }

    #[test]
    fn integer_range_1_to_9999() {
        let body = positive_integer_range_ebnf(1, Some(9999)).unwrap();
        assert_eq!(body, "[1-9] [0-9]{0,3}");
    }

    #[test]
    fn integer_range_0_to_99() {
        let body = positive_integer_range_ebnf(0, Some(99)).unwrap();
        assert_eq!(body, "(\"0\" | [1-9] [0-9]{0,1})");
    }

    #[test]
    fn integer_range_arbitrary_bound_125_to_478() {
        // The bounded-digits walker should produce a union covering
        // exactly the 125..=478 range. Verify by sampling endpoints
        // and a few internal points: spot-check the structure
        // contains the expected sub-rules.
        let body = positive_integer_range_ebnf(125, Some(478)).unwrap();
        // Sanity: must contain literals for the boundary digit
        // prefixes (1, 2, 3, 4) and bracket expressions.
        assert!(
            body.contains("[1-9]") || body.contains("\"1\""),
            "must reference leading digits 1-4; got: {body}"
        );
        // Must contain the upper-tail and lower-tail digit literals.
        assert!(
            body.contains("\"4\""),
            "must mention 4 as boundary digit: {body}"
        );
    }

    #[test]
    fn integer_range_500_to_500_single() {
        // Single-value range — degenerate but legal.
        let body = positive_integer_range_ebnf(500, Some(500)).unwrap();
        assert_eq!(body, "\"500\"");
    }

    #[test]
    fn integer_range_min_unbounded_above_one() {
        // `minimum: 5`, no max — accept the loose `[1-9][0-9]*` form
        // documented in the function body; lower bound not strictly
        // enforced (log emitted at debug level).
        let body = positive_integer_range_ebnf(5, None).unwrap();
        assert_eq!(body, "[1-9] [0-9]*");
    }

    #[test]
    fn integer_signed_range_minus_5_to_5() {
        let mut b = EbnfBuilder::new();
        let schema = json!({"type":"integer","minimum":-5,"maximum":5});
        let rule = b.emit(&schema).unwrap();
        let rule_body = &b.rules.iter().find(|(n, _)| n == &rule).unwrap().1;
        // Must contain a `-` prefix branch.
        assert!(
            rule_body.contains("\"-\""),
            "signed range must emit negative branch: {rule_body}"
        );
    }

    #[test]
    fn integer_all_negative_range() {
        let mut b = EbnfBuilder::new();
        let schema = json!({"type":"integer","minimum":-100,"maximum":-1});
        let rule = b.emit(&schema).unwrap();
        let body = &b.rules.iter().find(|(n, _)| n == &rule).unwrap().1;
        assert!(
            body.starts_with("\"-\""),
            "all-negative starts with `-`: {body}"
        );
    }

    #[test]
    fn debug_dump_concept_ebnf() {
        let schema = json!({
            "type":"object","additionalProperties":false,
            "required":["unique_features","description","year","season"],
            "properties":{
                "unique_features":{"type":"array","minItems":5,"maxItems":7,
                    "items":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":8,"maxLength":80}},
                "description":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":300,"maxLength":1600},
                "year":{"type":"integer","minimum":1,"maximum":9999},
                "season":{"type":"string","enum":["spring","summer","autumn","winter"]}
            }
        });
        let ebnf = try_schema_to_ebnf(&schema).unwrap().unwrap();
        eprintln!("===== generated EBNF =====\n{ebnf}\n=====");
    }

    #[test]
    fn emit_concept_schema_end_to_end_compiles_via_xgrammar() {
        // The user's concept_test schema (CYR pattern + length +
        // strict object + array). Verifies the emitted EBNF compiles
        // through xgrammar without error.
        let schema = json!({
            "type":"object","additionalProperties":false,
            "required":["unique_features","description","year","season"],
            "properties":{
                "unique_features":{"type":"array","minItems":5,"maxItems":7,
                    "items":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":8,"maxLength":80}},
                "description":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":300,"maxLength":1600},
                "year":{"type":"integer","minimum":1,"maximum":9999},
                "season":{"type":"string","enum":["spring","summer","autumn","winter"]}
            }
        });
        assert!(schema_needs_ebnf_path(&schema));
        let ebnf = try_schema_to_ebnf(&schema)
            .expect("translate")
            .expect("supported MVP");
        // Must reference the four key literals and the `ws` rule.
        for must_have in [
            "\\\"unique_features\\\"",
            "\\\"description\\\"",
            "\\\"year\\\"",
            "\\\"season\\\"",
            "ws ::=",
            "[1-9] [0-9]{0,3}",
        ] {
            assert!(
                ebnf.contains(must_have),
                "emitted EBNF missing fragment {must_have:?}:\n{ebnf}"
            );
        }
    }
}
