//! Reasoning / thinking token parser infrastructure.
//!
//! Models like DeepSeek-R1, Qwen3, and others emit chain-of-thought reasoning
//! within special markers (e.g., `<think>...</think>`). This module extracts
//! reasoning content from model output, splitting it into `reasoning` and
//! `content` fields for the OpenAI-compatible API response.
//!
//! # Supported Parsers
//!
//! - **DeepSeek R1**: `<think>`/`</think>` — handles missing start token
//! - **DeepSeek V3**: Delegates to R1 or Identity based on thinking mode
//! - **Qwen3**: `<think>`/`</think>` — strict (both tags required)
//! - **Mistral**: `[THINK]`/`[/THINK]` control tokens
//! - **Step3**: `</think>` end-only (all content before is reasoning)
//! - **Step3p5**: `<think>`/`</think>` with newline stripping
//! - **Ernie45**: `<think>`/`</think>` + `<response>` wrapper
//! - **Granite**: "Here is my thought process:" / "Here is my response:"
//! - **Olmo3**: `<think>`/`</think>` (string-space with overlap detection)
//! - **Seed-OSS**: `<seed:think>`/`</seed:think>`
//! - **MiniMax M2**: `<think>`/`</think>` (no start token from model)
//! - **MiniMax M2 Append-Think**: passthrough that prepends `<think>` to content
//! - **HunYuan A13B**: `<think>`/`</think>` with state machine
//! - **Identity**: No-op (all output → content)
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::reasoning::{create_reasoning_parser, ReasoningParser};
//!
//! let parser = create_reasoning_parser("deepseek_r1");
//! let output = "<think>Let me work this out...</think>The answer is 42.";
//! let result = parser.extract_reasoning(output);
//! assert_eq!(result.reasoning.as_deref(), Some("Let me work this out..."));
//! assert_eq!(result.content.as_deref(), Some("The answer is 42."));
//! ```

/// Result of extracting reasoning from model output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningOutput {
    /// The chain-of-thought reasoning text, if any.
    pub reasoning: Option<String>,
    /// The final content/answer text, if any.
    pub content: Option<String>,
}

/// Streaming delta with either reasoning or content (mutually exclusive per delta).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    /// Reasoning text in this delta, if currently in the thinking block.
    pub reasoning: Option<String>,
    /// Content text in this delta, if outside the thinking block.
    pub content: Option<String>,
}

/// Trait for parsing reasoning/thinking tokens from model output.
///
/// Different models use different formats for emitting chain-of-thought reasoning.
/// This trait provides a common interface for extracting reasoning from both
/// complete outputs (non-streaming) and incremental deltas (streaming).
pub trait ReasoningParser: Send + Sync {
    /// Extract reasoning and content from a complete model output (non-streaming).
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput;

    /// Extract reasoning/content from a streaming delta.
    ///
    /// `previous_text` is all text accumulated before this delta.
    /// `delta_text` is the new text being added.
    ///
    /// Returns a `ReasoningDelta` with either `reasoning` or `content` set
    /// (mutually exclusive — a single delta carries one or the other, never both).
    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta;
}

// ─── Tag-based parser (shared logic for <think>/<\think> and similar) ─────

/// A reasoning parser that uses simple start/end tag pairs.
///
/// Handles the common pattern where reasoning is wrapped in matching tags
/// like `<think>...</think>`. Supports multiple behaviors via `TagParserMode`.
struct TagReasoningParser {
    start_tag: &'static str,
    end_tag: &'static str,
    mode: TagParserMode,
}

/// Behavioral modes for tag-based reasoning parsers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagParserMode {
    /// Standard: requires start tag. Content before start tag is content.
    /// DeepSeek R1 behavior: also handles missing start tag.
    DeepSeekR1,
    /// Strict: requires BOTH start and end tags present.
    /// If either is missing, entire output is treated as content.
    Strict,
    /// No-start: model doesn't generate start tag.
    /// Everything before end tag is reasoning.
    NoStartTag,
    /// End-only: only end tag matters (no start tag concept).
    /// Everything before end tag is reasoning.
    EndOnly,
    /// Newline-stripping: like standard but strips \n before end tag
    /// and \n\n after end tag.
    StripNewlines,
}

impl TagReasoningParser {
    const fn new(start_tag: &'static str, end_tag: &'static str, mode: TagParserMode) -> Self {
        Self {
            start_tag,
            end_tag,
            mode,
        }
    }

    fn non_empty(s: &str) -> Option<String> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }
}

impl ReasoningParser for TagReasoningParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        match self.mode {
            TagParserMode::Strict => {
                // Both tags must be present, otherwise all content
                let has_start = output.contains(self.start_tag);
                let has_end = output.contains(self.end_tag);
                if !has_start || !has_end {
                    return ReasoningOutput {
                        reasoning: None,
                        content: Self::non_empty(output),
                    };
                }
                if let Some(start_pos) = output.find(self.start_tag) {
                    let after_start = start_pos + self.start_tag.len();
                    if let Some(end_pos) = output[after_start..].find(self.end_tag) {
                        let reasoning = &output[after_start..after_start + end_pos];
                        let content = &output[after_start + end_pos + self.end_tag.len()..];
                        return ReasoningOutput {
                            reasoning: Self::non_empty(reasoning),
                            content: Self::non_empty(content),
                        };
                    }
                }
                ReasoningOutput {
                    reasoning: None,
                    content: Self::non_empty(output),
                }
            }
            TagParserMode::DeepSeekR1 => {
                // Find start tag
                if let Some(start_pos) = output.find(self.start_tag) {
                    let after_start = start_pos + self.start_tag.len();
                    if let Some(end_pos) = output[after_start..].find(self.end_tag) {
                        let reasoning = &output[after_start..after_start + end_pos];
                        let content = &output[after_start + end_pos + self.end_tag.len()..];
                        return ReasoningOutput {
                            reasoning: Self::non_empty(reasoning),
                            content: Self::non_empty(content),
                        };
                    }
                    // Start tag but no end tag — everything after start is reasoning
                    let reasoning = &output[after_start..];
                    return ReasoningOutput {
                        reasoning: Self::non_empty(reasoning),
                        content: None,
                    };
                }
                // No start tag — check for end tag (model skipped start)
                if let Some(end_pos) = output.find(self.end_tag) {
                    let reasoning = &output[..end_pos];
                    let content = &output[end_pos + self.end_tag.len()..];
                    return ReasoningOutput {
                        reasoning: Self::non_empty(reasoning),
                        content: Self::non_empty(content),
                    };
                }
                // No tags at all — all content
                ReasoningOutput {
                    reasoning: None,
                    content: Self::non_empty(output),
                }
            }
            TagParserMode::NoStartTag | TagParserMode::EndOnly => {
                // Everything before end tag is reasoning
                if let Some(end_pos) = output.find(self.end_tag) {
                    let reasoning = &output[..end_pos];
                    let content = &output[end_pos + self.end_tag.len()..];
                    ReasoningOutput {
                        reasoning: Self::non_empty(reasoning),
                        content: Self::non_empty(content),
                    }
                } else {
                    // No end tag — all reasoning (still thinking)
                    ReasoningOutput {
                        reasoning: Self::non_empty(output),
                        content: None,
                    }
                }
            }
            TagParserMode::StripNewlines => {
                if let Some(start_pos) = output.find(self.start_tag) {
                    let after_start = start_pos + self.start_tag.len();
                    if let Some(end_pos) = output[after_start..].find(self.end_tag) {
                        let mut reasoning = output[after_start..after_start + end_pos].to_string();
                        // Strip trailing newlines from reasoning
                        while reasoning.ends_with('\n') {
                            reasoning.pop();
                        }
                        let content = output[after_start + end_pos + self.end_tag.len()..]
                            .trim_start_matches('\n');
                        return ReasoningOutput {
                            reasoning: Self::non_empty(&reasoning),
                            content: Self::non_empty(content),
                        };
                    }
                    let reasoning = &output[after_start..];
                    return ReasoningOutput {
                        reasoning: Self::non_empty(reasoning),
                        content: None,
                    };
                }
                // No start tag — check for end tag
                if let Some(end_pos) = output.find(self.end_tag) {
                    let reasoning = &output[..end_pos];
                    let content = output[end_pos + self.end_tag.len()..].trim_start_matches('\n');
                    return ReasoningOutput {
                        reasoning: Self::non_empty(reasoning),
                        content: Self::non_empty(content),
                    };
                }
                ReasoningOutput {
                    reasoning: None,
                    content: Self::non_empty(output),
                }
            }
        }
    }

    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta {
        let current_text_len = previous_text.len() + delta_text.len();
        // Build current_text only when needed (for tag search)
        let has_start_in_prev = previous_text.contains(self.start_tag);
        let has_end_in_prev = previous_text.contains(self.end_tag);

        match self.mode {
            TagParserMode::EndOnly | TagParserMode::NoStartTag => {
                // No start tag concept — track only end tag
                if has_end_in_prev {
                    // Already past end tag — all delta is content
                    return ReasoningDelta {
                        reasoning: None,
                        content: Some(delta_text.to_string()),
                    };
                }
                // Check if end tag spans previous+delta
                let search_start = previous_text.len().saturating_sub(self.end_tag.len());
                let combined = format!("{}{}", &previous_text[search_start..], delta_text);
                if let Some(end_pos_in_combined) = combined.find(self.end_tag) {
                    // End tag found in the boundary region
                    let abs_end = search_start + end_pos_in_combined;
                    if abs_end >= previous_text.len() {
                        // End tag starts in delta
                        let reasoning_part = &delta_text[..abs_end - previous_text.len()];
                        let content_part =
                            &delta_text[abs_end - previous_text.len() + self.end_tag.len()..];
                        if !reasoning_part.is_empty() && !content_part.is_empty() {
                            // Split delta: reasoning before end tag, content after
                            return ReasoningDelta {
                                reasoning: None,
                                content: Some(content_part.to_string()),
                            };
                        }
                        if !content_part.is_empty() {
                            return ReasoningDelta {
                                reasoning: None,
                                content: Some(content_part.to_string()),
                            };
                        }
                    }
                    // End tag fully in previous text's tail — emit remaining delta as content
                    let after_end_in_combined = end_pos_in_combined + self.end_tag.len();
                    if after_end_in_combined > combined.len() - delta_text.len() {
                        // Some delta text is after end tag
                        let delta_after_end =
                            after_end_in_combined - (combined.len() - delta_text.len());
                        let content_part = &delta_text[delta_after_end..];
                        if !content_part.is_empty() {
                            return ReasoningDelta {
                                reasoning: None,
                                content: Some(content_part.to_string()),
                            };
                        }
                    }
                    // End tag consumed the delta
                    return ReasoningDelta {
                        reasoning: None,
                        content: None,
                    };
                }
                // No end tag yet — all delta is reasoning
                // But check for potential partial end tag at the boundary
                if could_be_partial_tag(delta_text, self.end_tag) {
                    // Hold back — but for simplicity, emit as reasoning
                    // (the server will handle the rare edge case)
                    return ReasoningDelta {
                        reasoning: Some(delta_text.to_string()),
                        content: None,
                    };
                }
                ReasoningDelta {
                    reasoning: Some(delta_text.to_string()),
                    content: None,
                }
            }
            _ => {
                // Modes with start tag: DeepSeekR1, Strict, StripNewlines
                if has_end_in_prev {
                    // Already past end tag — all delta is content
                    let content = if self.mode == TagParserMode::StripNewlines {
                        // Check if we're right after end tag
                        let end_pos = previous_text.rfind(self.end_tag).unwrap();
                        let after_end = end_pos + self.end_tag.len();
                        if after_end == previous_text.len() {
                            // Delta is right after end tag — strip leading newlines
                            delta_text.trim_start_matches('\n').to_string()
                        } else {
                            delta_text.to_string()
                        }
                    } else {
                        delta_text.to_string()
                    };
                    return ReasoningDelta {
                        reasoning: None,
                        content: if content.is_empty() {
                            None
                        } else {
                            Some(content)
                        },
                    };
                }

                if has_start_in_prev {
                    // We're inside reasoning block — check if end tag appears in delta
                    // Check boundary between previous and delta
                    let search_start = previous_text.len().saturating_sub(self.end_tag.len());
                    let tail = &previous_text[search_start..];
                    let combined = format!("{}{}", tail, delta_text);
                    if let Some(end_pos) = combined.find(self.end_tag) {
                        let abs_end = search_start + end_pos;
                        // Split: reasoning before end tag, content after
                        if abs_end < previous_text.len() {
                            // End tag starts in previous text tail — delta is content
                            let content_start_in_combined = end_pos + self.end_tag.len();
                            let delta_start_in_combined = combined.len() - delta_text.len();
                            if content_start_in_combined > delta_start_in_combined {
                                let content = &delta_text
                                    [content_start_in_combined - delta_start_in_combined..];
                                return ReasoningDelta {
                                    reasoning: None,
                                    content: if content.is_empty() {
                                        None
                                    } else {
                                        Some(content.to_string())
                                    },
                                };
                            }
                            return ReasoningDelta {
                                reasoning: None,
                                content: None,
                            };
                        } else {
                            // End tag starts in delta
                            let delta_offset = abs_end - previous_text.len();
                            let reasoning_part = &delta_text[..delta_offset];
                            let content_part = &delta_text[delta_offset + self.end_tag.len()..];
                            // In StripNewlines mode, strip newlines at the boundary
                            let content_part = if self.mode == TagParserMode::StripNewlines {
                                content_part.trim_start_matches('\n')
                            } else {
                                content_part
                            };
                            if !reasoning_part.is_empty() && !content_part.is_empty() {
                                // Both reasoning and content in this delta — emit content
                                // (reasoning part was implicit continuation)
                                return ReasoningDelta {
                                    reasoning: None,
                                    content: Some(content_part.to_string()),
                                };
                            }
                            if !content_part.is_empty() {
                                return ReasoningDelta {
                                    reasoning: None,
                                    content: Some(content_part.to_string()),
                                };
                            }
                            if !reasoning_part.is_empty() {
                                return ReasoningDelta {
                                    reasoning: Some(reasoning_part.to_string()),
                                    content: None,
                                };
                            }
                            // End tag consumed entire delta
                            return ReasoningDelta {
                                reasoning: None,
                                content: None,
                            };
                        }
                    }
                    // No end tag — all delta is reasoning
                    return ReasoningDelta {
                        reasoning: Some(delta_text.to_string()),
                        content: None,
                    };
                }

                // No start tag in previous text yet
                if self.mode == TagParserMode::Strict {
                    // Check if start tag appears in delta
                    if let Some(start_pos) = delta_text.find(self.start_tag) {
                        let reasoning_part = &delta_text[start_pos + self.start_tag.len()..];
                        if !reasoning_part.is_empty() {
                            return ReasoningDelta {
                                reasoning: Some(reasoning_part.to_string()),
                                content: None,
                            };
                        }
                        return ReasoningDelta {
                            reasoning: None,
                            content: None,
                        };
                    }
                    // No start tag — all content
                    return ReasoningDelta {
                        reasoning: None,
                        content: Some(delta_text.to_string()),
                    };
                }

                // DeepSeekR1 / StripNewlines mode — check delta for start tag
                // Also check boundary for split start tag
                let search_start = previous_text.len().saturating_sub(self.start_tag.len());
                let tail = &previous_text[search_start..];
                let combined = format!("{}{}", tail, delta_text);
                if let Some(start_pos) = combined.find(self.start_tag) {
                    let abs_start = search_start + start_pos;
                    let after_start_in_combined = start_pos + self.start_tag.len();
                    let delta_start_in_combined = combined.len() - delta_text.len();

                    if after_start_in_combined > delta_start_in_combined {
                        let reasoning_part =
                            &delta_text[after_start_in_combined - delta_start_in_combined..];
                        if !reasoning_part.is_empty() {
                            // Check if end tag is also in this reasoning part
                            if let Some(end_in_reasoning) = reasoning_part.find(self.end_tag) {
                                let r = &reasoning_part[..end_in_reasoning];
                                let c = &reasoning_part[end_in_reasoning + self.end_tag.len()..];
                                if !c.is_empty() {
                                    return ReasoningDelta {
                                        reasoning: None,
                                        content: Some(c.to_string()),
                                    };
                                }
                                if !r.is_empty() {
                                    return ReasoningDelta {
                                        reasoning: Some(r.to_string()),
                                        content: None,
                                    };
                                }
                                return ReasoningDelta {
                                    reasoning: None,
                                    content: None,
                                };
                            }
                            return ReasoningDelta {
                                reasoning: Some(reasoning_part.to_string()),
                                content: None,
                            };
                        }
                        // Start tag consumed the delta
                        return ReasoningDelta {
                            reasoning: None,
                            content: None,
                        };
                    }
                    // Start tag entirely in previous text's tail region but wasn't
                    // detected before — shouldn't happen, but handle gracefully
                    if abs_start < previous_text.len() {
                        return ReasoningDelta {
                            reasoning: Some(delta_text.to_string()),
                            content: None,
                        };
                    }
                }

                // Check if end tag appears (DeepSeekR1: model skipped start)
                if self.mode == TagParserMode::DeepSeekR1 {
                    let end_combined = format!("{}{}", tail, delta_text);
                    if let Some(end_pos) = end_combined.find(self.end_tag) {
                        let abs_end = search_start + end_pos;
                        if abs_end >= previous_text.len() {
                            let delta_offset = abs_end - previous_text.len();
                            let reasoning_part = &delta_text[..delta_offset];
                            let content_part = &delta_text[delta_offset + self.end_tag.len()..];
                            if !content_part.is_empty() {
                                return ReasoningDelta {
                                    reasoning: None,
                                    content: Some(content_part.to_string()),
                                };
                            }
                            if !reasoning_part.is_empty() {
                                return ReasoningDelta {
                                    reasoning: Some(reasoning_part.to_string()),
                                    content: None,
                                };
                            }
                            return ReasoningDelta {
                                reasoning: None,
                                content: None,
                            };
                        }
                    }
                }

                // No tags found — in DeepSeekR1 mode with no start tag,
                // we can't know if this is reasoning or content yet.
                // Default: treat as content (will be corrected when end tag appears)
                if previous_text.is_empty() && current_text_len <= self.start_tag.len() {
                    // Very beginning — could be the start tag being built up
                    return ReasoningDelta {
                        reasoning: None,
                        content: None,
                    };
                }
                ReasoningDelta {
                    reasoning: None,
                    content: Some(delta_text.to_string()),
                }
            }
        }
    }
}

/// Check if the end of `text` could be the beginning of `tag`.
fn could_be_partial_tag(text: &str, tag: &str) -> bool {
    if text.is_empty() || tag.is_empty() {
        return false;
    }
    let max_overlap = text.len().min(tag.len() - 1);
    for len in (1..=max_overlap).rev() {
        if text.ends_with(&tag[..len]) {
            return true;
        }
    }
    false
}

// ─── Granite parser (regex/text-based) ────────────────────────────────────

struct GraniteReasoningParser;

impl ReasoningParser for GraniteReasoningParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        let thought_start = find_granite_marker(output, true);
        let response_start = find_granite_marker(output, false);

        match (thought_start, response_start) {
            (Some(ts), Some(rs)) => {
                let reasoning = &output[ts.end..rs.start];
                let content = &output[rs.end..];
                ReasoningOutput {
                    reasoning: TagReasoningParser::non_empty(reasoning),
                    content: TagReasoningParser::non_empty(content),
                }
            }
            (Some(ts), None) => {
                let reasoning = &output[ts.end..];
                ReasoningOutput {
                    reasoning: TagReasoningParser::non_empty(reasoning),
                    content: None,
                }
            }
            (None, Some(rs)) => {
                let content = &output[rs.end..];
                ReasoningOutput {
                    reasoning: None,
                    content: TagReasoningParser::non_empty(content),
                }
            }
            (None, None) => ReasoningOutput {
                reasoning: None,
                content: TagReasoningParser::non_empty(output),
            },
        }
    }

    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta {
        let current_text = format!("{}{}", previous_text, delta_text);

        let has_thought_prev = find_granite_marker(previous_text, true).is_some();
        let has_response_prev = find_granite_marker(previous_text, false).is_some();
        let has_response_curr = find_granite_marker(&current_text, false).is_some();

        if has_response_prev || has_response_curr {
            // We're in content mode
            ReasoningDelta {
                reasoning: None,
                content: if has_response_prev {
                    Some(delta_text.to_string())
                } else {
                    // Response marker just appeared — extract content after it
                    if let Some(rs) = find_granite_marker(&current_text, false) {
                        let _content = &current_text[rs.end..];
                        let prev_end = previous_text.len();
                        if rs.end > prev_end {
                            let in_delta = &delta_text[rs.end - prev_end..];
                            if in_delta.is_empty() {
                                None
                            } else {
                                Some(in_delta.to_string())
                            }
                        } else {
                            Some(delta_text.to_string())
                        }
                    } else {
                        Some(delta_text.to_string())
                    }
                },
            }
        } else if has_thought_prev {
            // We're in reasoning mode
            ReasoningDelta {
                reasoning: Some(delta_text.to_string()),
                content: None,
            }
        } else {
            // Check if thought marker just appeared
            if find_granite_marker(&current_text, true).is_some() {
                // Thought just started — delta might contain the marker
                ReasoningDelta {
                    reasoning: None,
                    content: None,
                }
            } else {
                // No markers yet — buffer (emit as content for now)
                ReasoningDelta {
                    reasoning: None,
                    content: Some(delta_text.to_string()),
                }
            }
        }
    }
}

struct MarkerPos {
    #[allow(dead_code)]
    start: usize,
    end: usize,
}

/// Find "Here is my thought process:" or "Here is my response:" markers.
fn find_granite_marker(text: &str, is_thought: bool) -> Option<MarkerPos> {
    let patterns: &[&str] = if is_thought {
        &["Here is my thought process:", "Here's my thought process:"]
    } else {
        &["Here is my response:", "Here's my response:"]
    };

    for pattern in patterns {
        if let Some(start) = text.find(pattern) {
            return Some(MarkerPos {
                start,
                end: start + pattern.len(),
            });
        }
    }
    None
}

// ─── Ernie45 parser (think + response tags) ───────────────────────────────

struct Ernie45ReasoningParser;

impl ReasoningParser for Ernie45ReasoningParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        // Ernie format: <think>reasoning\n</think>\n\n<response>\ncontent\n</response>\n
        let think_start = output.find("<think>");
        let think_end = output.find("</think>");

        match (think_start, think_end) {
            (Some(ts), Some(te)) => {
                let reasoning = &output[ts + "<think>".len()..te];
                let after_think = &output[te + "</think>".len()..];
                // Strip <response> wrapper if present
                let content = strip_response_tags(after_think);
                ReasoningOutput {
                    reasoning: TagReasoningParser::non_empty(reasoning),
                    content: TagReasoningParser::non_empty(content),
                }
            }
            (None, Some(te)) => {
                // No start tag — content before end is reasoning
                let reasoning = &output[..te];
                let after_think = &output[te + "</think>".len()..];
                let content = strip_response_tags(after_think);
                ReasoningOutput {
                    reasoning: TagReasoningParser::non_empty(reasoning),
                    content: TagReasoningParser::non_empty(content),
                }
            }
            (Some(ts), None) => {
                let reasoning = &output[ts + "<think>".len()..];
                ReasoningOutput {
                    reasoning: TagReasoningParser::non_empty(reasoning),
                    content: None,
                }
            }
            (None, None) => ReasoningOutput {
                reasoning: None,
                content: TagReasoningParser::non_empty(output),
            },
        }
    }

    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta {
        let has_end_in_prev = previous_text.contains("</think>");

        if has_end_in_prev {
            // Past thinking — all delta is content (strip response tags inline)
            let clean = delta_text
                .replace("<response>", "")
                .replace("</response>", "");
            let clean = clean.trim_start_matches('\n');
            ReasoningDelta {
                reasoning: None,
                content: if clean.is_empty() {
                    None
                } else {
                    Some(clean.to_string())
                },
            }
        } else {
            // Check if </think> appears in the boundary
            let search_start = previous_text.len().saturating_sub("</think>".len());
            let combined = format!("{}{}", &previous_text[search_start..], delta_text);
            if let Some(end_pos) = combined.find("</think>") {
                let abs_end = search_start + end_pos;
                if abs_end >= previous_text.len() {
                    let delta_offset = abs_end - previous_text.len();
                    let reasoning_part = &delta_text[..delta_offset];
                    let content_part = &delta_text[delta_offset + "</think>".len()..];
                    let content_part = strip_response_tags(content_part);
                    let content_part = content_part.trim_start_matches('\n');
                    if !content_part.is_empty() {
                        return ReasoningDelta {
                            reasoning: None,
                            content: Some(content_part.to_string()),
                        };
                    }
                    if !reasoning_part.is_empty() {
                        return ReasoningDelta {
                            reasoning: Some(reasoning_part.to_string()),
                            content: None,
                        };
                    }
                    return ReasoningDelta {
                        reasoning: None,
                        content: None,
                    };
                }
                // End tag in previous text
                return ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        strip_response_tags(delta_text)
                            .trim_start_matches('\n')
                            .to_string(),
                    ),
                };
            }
            // No end tag — check if we're past start tag
            if previous_text.contains("<think>") || delta_text.contains("<think>") {
                // In reasoning mode — strip the start tag from delta if present
                let delta_clean = delta_text.replace("<think>", "");
                if delta_clean.is_empty() {
                    ReasoningDelta {
                        reasoning: None,
                        content: None,
                    }
                } else {
                    ReasoningDelta {
                        reasoning: Some(delta_clean),
                        content: None,
                    }
                }
            } else {
                // Before any tags — treat as content
                ReasoningDelta {
                    reasoning: None,
                    content: Some(delta_text.to_string()),
                }
            }
        }
    }
}

fn strip_response_tags(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("<response>").unwrap_or(s);
    let s = s.strip_suffix("</response>").unwrap_or(s);
    s.trim()
}

// ─── Identity parser (no-op) ──────────────────────────────────────────────

struct IdentityReasoningParser;

impl ReasoningParser for IdentityReasoningParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        ReasoningOutput {
            reasoning: None,
            content: if output.is_empty() {
                None
            } else {
                Some(output.to_string())
            },
        }
    }

    fn extract_reasoning_streaming(
        &self,
        _previous_text: &str,
        delta_text: &str,
    ) -> ReasoningDelta {
        ReasoningDelta {
            reasoning: None,
            content: if delta_text.is_empty() {
                None
            } else {
                Some(delta_text.to_string())
            },
        }
    }
}

// ─── MiniMax M2 Append-Think Parser ──────────────────────────────────────

/// MiniMax M2 "append think" mode: wraps all output in `<think>` as content.
///
/// Unlike the standard MiniMax M2 parser which extracts reasoning into a
/// separate field, this variant prepends `<think>` to the output and puts
/// everything in `content` — making the thinking visible inline.
struct MiniMaxM2AppendThinkParser;

impl ReasoningParser for MiniMaxM2AppendThinkParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        if output.is_empty() {
            return ReasoningOutput::default();
        }
        ReasoningOutput {
            reasoning: None,
            content: Some(format!("<think>{output}")),
        }
    }

    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta {
        if delta_text.is_empty() {
            return ReasoningDelta::default();
        }
        if previous_text.is_empty() {
            ReasoningDelta {
                reasoning: None,
                content: Some(format!("<think>{delta_text}")),
            }
        } else {
            ReasoningDelta {
                reasoning: None,
                content: Some(delta_text.to_string()),
            }
        }
    }
}

// ─── GPT-OSS Reasoning Parser ────────────────────────────────────────────────

/// Reasoning parser for GPT-OSS models (harmony-format channel protocol).
///
/// GPT-OSS structures output into named channels:
/// ```text
/// <|channel|>analysis<|message|>{reasoning}<|end|>
/// <|channel|>final<|message|>{content}<|end|>
/// ```
///
/// Up to ~200 bytes of special tokens may appear between `<|channel|>final` and
/// `<|message|>` (equivalent to the "up to 20 token IDs" in the Python reference).
/// These are silently skipped.
///
/// Streaming mirrors the Python `parse_chat_output()` approach: re-parse the full
/// accumulated text on each delta and return the incremental diff.
struct GptOssReasoningParser;

impl GptOssReasoningParser {
    const ANALYSIS_BEGIN: &'static str = "<|channel|>analysis<|message|>";
    const FINAL_CHANNEL: &'static str = "<|channel|>final";
    const MESSAGE_SEP: &'static str = "<|message|>";
    const CHANNEL_END: &'static str = "<|end|>";
    /// Byte budget between `<|channel|>final` and `<|message|>`.
    /// Corresponds to the Python `reasoning_max_num_between_tokens = 20`.
    /// 20 tokens × ≤ 10 bytes each = 200 bytes (generous upper bound).
    const MAX_INTER_BYTES: usize = 200;

    fn parse(text: &str) -> ReasoningOutput {
        // Reasoning: text between ANALYSIS_BEGIN and CHANNEL_END (or end of string).
        let reasoning = if let Some(a_pos) = text.find(Self::ANALYSIS_BEGIN) {
            let after = a_pos + Self::ANALYSIS_BEGIN.len();
            let body = match text[after..].find(Self::CHANNEL_END) {
                Some(end_off) => &text[after..after + end_off],
                None => &text[after..],
            };
            let s = body.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        } else {
            None
        };

        // Content: after `<|channel|>final` + up to MAX_INTER_BYTES + `<|message|>`.
        let content = if let Some(f_pos) = text.find(Self::FINAL_CHANNEL) {
            let search_from = f_pos + Self::FINAL_CHANNEL.len();
            let search_end = (search_from + Self::MAX_INTER_BYTES).min(text.len());
            if let Some(msg_off) = text[search_from..search_end].find(Self::MESSAGE_SEP) {
                let content_start = search_from + msg_off + Self::MESSAGE_SEP.len();
                let body = match text[content_start..].find(Self::CHANNEL_END) {
                    Some(end_off) => &text[content_start..content_start + end_off],
                    None => &text[content_start..],
                };
                let s = body.trim();
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            } else {
                None
            }
        } else {
            None
        };

        ReasoningOutput { reasoning, content }
    }

    /// Compute the incremental delta between two versions of an accumulated string.
    ///
    /// If `cur` starts with `prev`, returns only the appended suffix.
    /// If `cur` is `None`, returns `None`. If `prev` is `None`, returns all of `cur`.
    fn delta(prev: Option<&str>, cur: Option<&str>) -> Option<String> {
        let cur = cur?;
        match prev {
            None => Some(cur.to_string()),
            Some(p) if cur.starts_with(p) => {
                let tail = &cur[p.len()..];
                if tail.is_empty() {
                    None
                } else {
                    Some(tail.to_string())
                }
            }
            // Edge case: text was truncated/re-emitted — return full cur.
            Some(_) => Some(cur.to_string()),
        }
    }
}

impl ReasoningParser for GptOssReasoningParser {
    fn extract_reasoning(&self, output: &str) -> ReasoningOutput {
        Self::parse(output)
    }

    /// Re-parse the full accumulated text and return the incremental diff.
    ///
    /// This mirrors the Python streaming implementation which calls
    /// `parse_chat_output(prev_ids)` and `parse_chat_output(cur_ids)` and diffs.
    fn extract_reasoning_streaming(&self, previous_text: &str, delta_text: &str) -> ReasoningDelta {
        let current = format!("{previous_text}{delta_text}");
        let prev = Self::parse(previous_text);
        let cur = Self::parse(&current);

        ReasoningDelta {
            reasoning: Self::delta(prev.reasoning.as_deref(), cur.reasoning.as_deref()),
            content: Self::delta(prev.content.as_deref(), cur.content.as_deref()),
        }
    }
}

// ─── Factory ──────────────────────────────────────────────────────────────

/// Create a reasoning parser by name.
///
/// Supported names:
/// - `deepseek_r1` — DeepSeek-R1 (`<think>`/`</think>`, handles missing start)
/// - `deepseek_v3` — DeepSeek V3 (same as `deepseek_r1`)
/// - `qwen3` — Qwen3 (`<think>`/`</think>`, strict mode)
/// - `mistral` — Mistral (`[THINK]`/`[/THINK]`)
/// - `step3` — Step3 (`</think>` end-only)
/// - `step3p5` — Step3.5 (`<think>`/`</think>` with newline stripping)
/// - `ernie45` — ERNIE 4.5 (`<think>` + `<response>` tags)
/// - `granite` — Granite ("Here is my thought process:")
/// - `olmo3` — OLMo3 (`<think>`/`</think>`)
/// - `seed_oss` — Seed-OSS (`<seed:think>`/`</seed:think>`)
/// - `minimax_m2` — MiniMax M2 (`<think>`/`</think>`, no start from model)
/// - `minimax_m2_append_think` — MiniMax M2 passthrough (prepends `<think>` to content)
/// - `hunyuan_a13b` — HunYuan A13B (`<think>`/`</think>`)
/// - `glm45`, `holo2`, `kimi_k2` — aliases for `deepseek_r1`
/// - `gpt_oss` — GPT-OSS harmony-format channels (`<|channel|>analysis` / `<|channel|>final`)
/// - `identity` — no-op (all output → content)
///
/// Unknown names default to `identity`.
pub fn create_reasoning_parser(name: &str) -> Box<dyn ReasoningParser> {
    match name {
        "deepseek_r1" | "deepseek-r1" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        // DeepSeek V3 with thinking enabled acts like R1
        "deepseek_v3" | "deepseek-v3" | "deepseek_v3_thinking" => Box::new(
            TagReasoningParser::new("<think>", "</think>", TagParserMode::DeepSeekR1),
        ),
        "qwen3" | "qwen-3" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::Strict,
        )),
        "mistral" => Box::new(TagReasoningParser::new(
            "[THINK]",
            "[/THINK]",
            TagParserMode::DeepSeekR1,
        )),
        "step3" | "step-3" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::EndOnly,
        )),
        "step3p5" | "step-3.5" | "step3.5" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::StripNewlines,
        )),
        "ernie45" | "ernie-4.5" | "ernie_45" => Box::new(Ernie45ReasoningParser),
        "granite" => Box::new(GraniteReasoningParser),
        "olmo3" | "olmo-3" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        "seed_oss" | "seed-oss" => Box::new(TagReasoningParser::new(
            "<seed:think>",
            "</seed:think>",
            TagParserMode::DeepSeekR1,
        )),
        "minimax_m2" | "minimax-m2" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::NoStartTag,
        )),
        "minimax_m2_append_think" | "minimax-m2-append-think" => {
            Box::new(MiniMaxM2AppendThinkParser)
        }
        "hunyuan_a13b" | "hunyuan-a13b" | "hunyuan" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        // Aliases — all use <think>/</ think> with DeepSeek R1 behavior
        "glm45" | "glm-45" | "glm4.5" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        "holo2" | "holo-2" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        "kimi_k2" | "kimi-k2" => Box::new(TagReasoningParser::new(
            "<think>",
            "</think>",
            TagParserMode::DeepSeekR1,
        )),
        "gpt_oss" | "gpt-oss" | "gptoss" => Box::new(GptOssReasoningParser),
        // No-op: all output is content
        "identity" | "" => Box::new(IdentityReasoningParser),
        unknown => {
            tracing::warn!("Unknown reasoning parser '{unknown}', defaulting to identity");
            Box::new(IdentityReasoningParser)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── DeepSeek R1 ──────────────────────────────────────────────────

    #[test]
    fn deepseek_r1_basic() {
        let parser = create_reasoning_parser("deepseek_r1");
        let output = "<think>Let me think...</think>The answer is 42.";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Let me think..."));
        assert_eq!(result.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn deepseek_r1_no_start_tag() {
        let parser = create_reasoning_parser("deepseek_r1");
        let output = "I need to think about this...</think>The answer is 42.";
        let result = parser.extract_reasoning(output);
        assert_eq!(
            result.reasoning.as_deref(),
            Some("I need to think about this...")
        );
        assert_eq!(result.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn deepseek_r1_no_end_tag() {
        let parser = create_reasoning_parser("deepseek_r1");
        let output = "<think>Still thinking...";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Still thinking..."));
        assert!(result.content.is_none());
    }

    #[test]
    fn deepseek_r1_no_tags() {
        let parser = create_reasoning_parser("deepseek_r1");
        let output = "Just a normal response";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(result.content.as_deref(), Some("Just a normal response"));
    }

    #[test]
    fn deepseek_r1_empty_reasoning() {
        let parser = create_reasoning_parser("deepseek_r1");
        let output = "<think></think>The answer is 42.";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(result.content.as_deref(), Some("The answer is 42."));
    }

    // ─── Qwen3 (strict mode) ─────────────────────────────────────────

    #[test]
    fn qwen3_basic() {
        let parser = create_reasoning_parser("qwen3");
        let output = "<think>Thinking...</think>Answer.";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Thinking..."));
        assert_eq!(result.content.as_deref(), Some("Answer."));
    }

    #[test]
    fn qwen3_missing_start_tag() {
        let parser = create_reasoning_parser("qwen3");
        let output = "No thinking tags here</think>Answer.";
        let result = parser.extract_reasoning(output);
        // Strict mode: missing start → all content
        assert!(result.reasoning.is_none());
        assert!(result.content.is_some());
    }

    #[test]
    fn qwen3_missing_end_tag() {
        let parser = create_reasoning_parser("qwen3");
        let output = "<think>Still thinking without end tag";
        let result = parser.extract_reasoning(output);
        // Strict mode: missing end → all content
        assert!(result.reasoning.is_none());
        assert!(result.content.is_some());
    }

    #[test]
    fn qwen3_no_tags() {
        let parser = create_reasoning_parser("qwen3");
        let output = "Normal response without tags";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(
            result.content.as_deref(),
            Some("Normal response without tags")
        );
    }

    // ─── Mistral ──────────────────────────────────────────────────────

    #[test]
    fn mistral_basic() {
        let parser = create_reasoning_parser("mistral");
        let output = "[THINK]Let me reason...[/THINK]The answer.";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Let me reason..."));
        assert_eq!(result.content.as_deref(), Some("The answer."));
    }

    // ─── Step3 (end-only) ─────────────────────────────────────────────

    #[test]
    fn step3_basic() {
        let parser = create_reasoning_parser("step3");
        let output = "My reasoning process...</think>The answer.";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("My reasoning process..."));
        assert_eq!(result.content.as_deref(), Some("The answer."));
    }

    #[test]
    fn step3_no_end_tag() {
        let parser = create_reasoning_parser("step3");
        let output = "Still reasoning...";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Still reasoning..."));
        assert!(result.content.is_none());
    }

    // ─── Step3p5 (newline stripping) ──────────────────────────────────

    #[test]
    fn step3p5_basic() {
        let parser = create_reasoning_parser("step3p5");
        let output = "<think>Reasoning\n\n</think>\n\nContent here";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Reasoning"));
        assert_eq!(result.content.as_deref(), Some("Content here"));
    }

    // ─── Ernie45 ──────────────────────────────────────────────────────

    #[test]
    fn ernie45_basic() {
        let parser = create_reasoning_parser("ernie45");
        let output = "<think>My reasoning</think>\n\n<response>\nThe answer\n</response>\n";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("My reasoning"));
        assert_eq!(result.content.as_deref(), Some("The answer"));
    }

    #[test]
    fn ernie45_no_response_tags() {
        let parser = create_reasoning_parser("ernie45");
        let output = "<think>Reasoning</think>Direct content";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Reasoning"));
        assert_eq!(result.content.as_deref(), Some("Direct content"));
    }

    // ─── Granite ──────────────────────────────────────────────────────

    #[test]
    fn granite_basic() {
        let parser = create_reasoning_parser("granite");
        let output = "Here is my thought process:\nI think about this...\nHere is my response:\nThe answer is 42.";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_some());
        assert!(result.content.is_some());
        assert!(result.reasoning.as_deref().unwrap().contains("I think"));
        assert!(result.content.as_deref().unwrap().contains("42"));
    }

    #[test]
    fn granite_heres_variant() {
        let parser = create_reasoning_parser("granite");
        let output = "Here's my thought process:\nThinking...\nHere's my response:\nDone.";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_some());
        assert!(result.content.is_some());
    }

    #[test]
    fn granite_no_markers() {
        let parser = create_reasoning_parser("granite");
        let output = "Just a normal response.";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(result.content.as_deref(), Some("Just a normal response."));
    }

    // ─── Seed-OSS ─────────────────────────────────────────────────────

    #[test]
    fn seed_oss_basic() {
        let parser = create_reasoning_parser("seed_oss");
        let output = "<seed:think>Reasoning here</seed:think>The answer.";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("Reasoning here"));
        assert_eq!(result.content.as_deref(), Some("The answer."));
    }

    // ─── MiniMax M2 (no start tag) ───────────────────────────────────

    #[test]
    fn minimax_m2_basic() {
        let parser = create_reasoning_parser("minimax_m2");
        let output = "Model reasoning without start tag</think>The answer.";
        let result = parser.extract_reasoning(output);
        assert_eq!(
            result.reasoning.as_deref(),
            Some("Model reasoning without start tag")
        );
        assert_eq!(result.content.as_deref(), Some("The answer."));
    }

    #[test]
    fn minimax_m2_no_end_tag() {
        let parser = create_reasoning_parser("minimax_m2");
        let output = "All reasoning, no end tag";
        let result = parser.extract_reasoning(output);
        assert_eq!(
            result.reasoning.as_deref(),
            Some("All reasoning, no end tag")
        );
        assert!(result.content.is_none());
    }

    // ─── MiniMax M2 Append-Think ──────────────────────────────────────

    #[test]
    fn minimax_m2_append_think_basic() {
        let parser = create_reasoning_parser("minimax_m2_append_think");
        let output = "reasoning here</think>answer here";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(
            result.content.as_deref(),
            Some("<think>reasoning here</think>answer here")
        );
    }

    #[test]
    fn minimax_m2_append_think_streaming() {
        let parser = create_reasoning_parser("minimax_m2_append_think");

        // First delta — should prepend <think>
        let d1 = parser.extract_reasoning_streaming("", "Hello");
        assert!(d1.reasoning.is_none());
        assert_eq!(d1.content.as_deref(), Some("<think>Hello"));

        // Second delta — plain passthrough
        let d2 = parser.extract_reasoning_streaming("Hello", " world");
        assert!(d2.reasoning.is_none());
        assert_eq!(d2.content.as_deref(), Some(" world"));
    }

    #[test]
    fn minimax_m2_append_think_empty_delta() {
        let parser = create_reasoning_parser("minimax_m2_append_think");
        let d = parser.extract_reasoning_streaming("", "");
        assert!(d.reasoning.is_none());
        assert!(d.content.is_none());
    }

    // ─── Identity ─────────────────────────────────────────────────────

    #[test]
    fn identity_basic() {
        let parser = create_reasoning_parser("identity");
        let output = "Everything is content.";
        let result = parser.extract_reasoning(output);
        assert!(result.reasoning.is_none());
        assert_eq!(result.content.as_deref(), Some("Everything is content."));
    }

    #[test]
    fn identity_empty() {
        let parser = create_reasoning_parser("identity");
        let result = parser.extract_reasoning("");
        assert!(result.reasoning.is_none());
        assert!(result.content.is_none());
    }

    // ─── Factory ──────────────────────────────────────────────────────

    #[test]
    fn factory_known_parsers() {
        let names = [
            "deepseek_r1",
            "deepseek_v3",
            "qwen3",
            "mistral",
            "step3",
            "step3p5",
            "ernie45",
            "granite",
            "olmo3",
            "seed_oss",
            "minimax_m2",
            "minimax_m2_append_think",
            "hunyuan_a13b",
            "glm45",
            "holo2",
            "kimi_k2",
            "gpt_oss",
            "identity",
            "",
        ];
        for name in &names {
            let parser = create_reasoning_parser(name);
            // Every parser should handle empty input without panicking
            let result = parser.extract_reasoning("");
            assert!(
                result.reasoning.is_none() && result.content.is_none(),
                "parser '{name}' should return empty for empty input"
            );
        }
    }

    #[test]
    fn factory_unknown_defaults_to_identity() {
        let parser = create_reasoning_parser("nonexistent_parser");
        let result = parser.extract_reasoning("<think>test</think>content");
        // Identity parser: no reasoning extraction
        assert!(result.reasoning.is_none());
        assert!(result.content.is_some());
    }

    // ─── Streaming tests ──────────────────────────────────────────────

    #[test]
    fn streaming_deepseek_r1_simple() {
        let parser = create_reasoning_parser("deepseek_r1");

        // Token 1: "<think>"
        let d1 = parser.extract_reasoning_streaming("", "<think>");
        assert!(d1.reasoning.is_none() || d1.content.is_none());

        // Token 2: "thinking"
        let d2 = parser.extract_reasoning_streaming("<think>", "thinking");
        assert_eq!(d2.reasoning.as_deref(), Some("thinking"));
        assert!(d2.content.is_none());

        // Token 3: "</think>"
        let d3 = parser.extract_reasoning_streaming("<think>thinking", "</think>");
        assert!(d3.content.is_none()); // End tag consumed the delta

        // Token 4: "answer"
        let d4 = parser.extract_reasoning_streaming("<think>thinking</think>", "answer");
        assert!(d4.reasoning.is_none());
        assert_eq!(d4.content.as_deref(), Some("answer"));
    }

    #[test]
    fn streaming_qwen3_strict() {
        let parser = create_reasoning_parser("qwen3");

        // Before start tag — content
        let d1 = parser.extract_reasoning_streaming("", "Hello ");
        assert_eq!(d1.content.as_deref(), Some("Hello "));
    }

    #[test]
    fn streaming_identity() {
        let parser = create_reasoning_parser("identity");
        let d1 = parser.extract_reasoning_streaming("", "hello");
        assert!(d1.reasoning.is_none());
        assert_eq!(d1.content.as_deref(), Some("hello"));

        let d2 = parser.extract_reasoning_streaming("hello", " world");
        assert!(d2.reasoning.is_none());
        assert_eq!(d2.content.as_deref(), Some(" world"));
    }

    #[test]
    fn streaming_step3_end_only() {
        let parser = create_reasoning_parser("step3");

        // Before end tag — all reasoning
        let d1 = parser.extract_reasoning_streaming("", "thinking");
        assert_eq!(d1.reasoning.as_deref(), Some("thinking"));
        assert!(d1.content.is_none());

        // After end tag — content
        let d2 = parser.extract_reasoning_streaming("thinking</think>", "answer");
        assert!(d2.reasoning.is_none());
        assert_eq!(d2.content.as_deref(), Some("answer"));
    }

    #[test]
    fn streaming_minimax_no_start() {
        let parser = create_reasoning_parser("minimax_m2");

        let d1 = parser.extract_reasoning_streaming("", "reasoning");
        assert_eq!(d1.reasoning.as_deref(), Some("reasoning"));

        let d2 = parser.extract_reasoning_streaming("reasoning</think>", "answer");
        assert_eq!(d2.content.as_deref(), Some("answer"));
    }

    // ─── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn could_be_partial_tag_works() {
        assert!(could_be_partial_tag("abc<thi", "<think>"));
        assert!(could_be_partial_tag("abc<", "<think>"));
        assert!(!could_be_partial_tag("abc", "<think>"));
        assert!(could_be_partial_tag("</thin", "</think>"));
        assert!(!could_be_partial_tag("", "</think>"));
    }

    #[test]
    fn ernie45_streaming_basic() {
        let parser = create_reasoning_parser("ernie45");

        // In reasoning mode
        let d1 = parser.extract_reasoning_streaming("<think>", "reasoning");
        assert_eq!(d1.reasoning.as_deref(), Some("reasoning"));

        // After end tag — strips response tags
        let d2 = parser.extract_reasoning_streaming("<think>reasoning</think>", "\n<response>\n");
        assert!(d2.reasoning.is_none());
    }

    // ─── GPT-OSS ──────────────────────────────────────────────────────

    #[test]
    fn gptoss_basic_both_channels() {
        let parser = create_reasoning_parser("gpt_oss");
        let output =
            "<|channel|>analysis<|message|>thinking here<|end|><|channel|>final<|message|>answer<|end|>";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("thinking here"));
        assert_eq!(result.content.as_deref(), Some("answer"));
    }

    #[test]
    fn gptoss_analysis_only_no_final() {
        let parser = create_reasoning_parser("gpt_oss");
        let output = "<|channel|>analysis<|message|>still reasoning";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("still reasoning"));
        assert!(result.content.is_none());
    }

    #[test]
    fn gptoss_final_channel_inter_tokens() {
        // Special tokens may appear between <|channel|>final and <|message|>.
        let parser = create_reasoning_parser("gpt_oss");
        let output =
            "<|channel|>analysis<|message|>reasoning<|end|><|channel|>final<|action|>none<|message|>final answer<|end|>";
        let result = parser.extract_reasoning(output);
        assert_eq!(result.reasoning.as_deref(), Some("reasoning"));
        assert_eq!(result.content.as_deref(), Some("final answer"));
    }

    #[test]
    fn gptoss_streaming_progressive() {
        let parser = create_reasoning_parser("gpt_oss");

        // Delta arrives inside the analysis channel
        let d1 = parser.extract_reasoning_streaming("<|channel|>analysis<|message|>", "thinking");
        assert_eq!(d1.reasoning.as_deref(), Some("thinking"));
        assert!(d1.content.is_none());

        // More reasoning
        let d2 =
            parser.extract_reasoning_streaming("<|channel|>analysis<|message|>thinking", " deeply");
        assert_eq!(d2.reasoning.as_deref(), Some(" deeply"));
        assert!(d2.content.is_none());

        // Transition: end of analysis + start of final content
        let prev =
            "<|channel|>analysis<|message|>thinking deeply<|end|><|channel|>final<|message|>";
        let d3 = parser.extract_reasoning_streaming(prev, "answer");
        assert!(d3.reasoning.is_none());
        assert_eq!(d3.content.as_deref(), Some("answer"));
    }

    #[test]
    fn gptoss_factory_aliases() {
        for name in &["gpt_oss", "gpt-oss", "gptoss"] {
            let parser = create_reasoning_parser(name);
            let out = "<|channel|>analysis<|message|>r<|end|><|channel|>final<|message|>c<|end|>";
            let result = parser.extract_reasoning(out);
            assert_eq!(
                result.reasoning.as_deref(),
                Some("r"),
                "alias '{name}' failed"
            );
            assert_eq!(
                result.content.as_deref(),
                Some("c"),
                "alias '{name}' failed"
            );
        }
    }
}
