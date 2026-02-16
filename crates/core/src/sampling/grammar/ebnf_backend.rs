//! EBNF/GBNF grammar backend.
//!
//! Two strategies:
//! 1. **Regular grammars** (no recursion): convert to regex, delegate to `RegexDfaGrammar`
//! 2. **Recursive grammars**: pushdown automaton with stack-based state tracking

use std::collections::HashMap;
use std::sync::Arc;

use super::bitmask::PackedBitmask;
use super::ebnf_parser::{
    CharClass, CharRange, GbnfElement, GbnfGrammar, GbnfRule, GbnfSequence, RepeatKind,
};
use super::regex_backend::RegexDfaGrammar;
use super::vocabulary::VocabularyIndex;
use super::StructuredOutputGrammar;

/// Create a grammar from a GBNF string.
///
/// Automatically selects the appropriate backend:
/// - Regular grammars → compiled to regex → DFA bitmasks
/// - Recursive grammars → pushdown automaton
pub fn ebnf_to_grammar(
    input: &str,
    vocab_index: Arc<VocabularyIndex>,
) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
    let parsed = super::ebnf_parser::parse_gbnf(input)?;

    if parsed.is_regular() {
        // Convert to regex and use DFA backend
        let regex = grammar_to_regex(&parsed)?;
        let dfa_grammar = RegexDfaGrammar::new(&regex, vocab_index)?;
        Ok(Box::new(dfa_grammar))
    } else {
        // Use pushdown automaton
        let pda = PdaGrammar::new(parsed, vocab_index)?;
        Ok(Box::new(pda))
    }
}

/// Convert a regular (non-recursive) GBNF grammar to a regex string.
fn grammar_to_regex(grammar: &GbnfGrammar) -> anyhow::Result<String> {
    let rule_map: HashMap<&str, &GbnfRule> =
        grammar.rules.iter().map(|r| (r.name.as_str(), r)).collect();

    let root = grammar
        .root_rule()
        .ok_or_else(|| anyhow::anyhow!("grammar has no rules"))?;

    rule_to_regex(root, &rule_map)
}

fn rule_to_regex(rule: &GbnfRule, rules: &HashMap<&str, &GbnfRule>) -> anyhow::Result<String> {
    let alts: Result<Vec<String>, _> = rule
        .alternatives
        .iter()
        .map(|seq| sequence_to_regex(seq, rules))
        .collect();
    let alts = alts?;

    if alts.len() == 1 {
        Ok(alts.into_iter().next().unwrap())
    } else {
        Ok(format!("({})", alts.join("|")))
    }
}

fn sequence_to_regex(
    seq: &GbnfSequence,
    rules: &HashMap<&str, &GbnfRule>,
) -> anyhow::Result<String> {
    let parts: Result<Vec<String>, _> = seq
        .elements
        .iter()
        .map(|e| element_to_regex(e, rules))
        .collect();
    Ok(parts?.join(""))
}

fn element_to_regex(
    elem: &GbnfElement,
    rules: &HashMap<&str, &GbnfRule>,
) -> anyhow::Result<String> {
    match elem {
        GbnfElement::Literal(s) => Ok(regex::escape(s)),
        GbnfElement::CharClass(cc) => Ok(char_class_to_regex(cc)),
        GbnfElement::RuleRef(name) => {
            let rule = rules
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("undefined rule: {name}"))?;
            let inner = rule_to_regex(rule, rules)?;
            Ok(format!("(?:{inner})"))
        }
        GbnfElement::Group(alts) => {
            let parts: Result<Vec<String>, _> = alts
                .iter()
                .map(|seq| sequence_to_regex(seq, rules))
                .collect();
            let parts = parts?;
            if parts.len() == 1 {
                Ok(format!("(?:{})", parts[0]))
            } else {
                Ok(format!("(?:{})", parts.join("|")))
            }
        }
        GbnfElement::Repeat(inner, kind) => {
            let inner_regex = element_to_regex(inner, rules)?;
            let suffix = match kind {
                RepeatKind::ZeroOrMore => "*",
                RepeatKind::OneOrMore => "+",
                RepeatKind::Optional => "?",
            };
            Ok(format!("(?:{inner_regex}){suffix}"))
        }
    }
}

fn char_class_to_regex(cc: &CharClass) -> String {
    let mut result = String::from("[");
    if cc.negated {
        result.push('^');
    }
    for range in &cc.ranges {
        match range {
            CharRange::Single(c) => {
                // Escape special regex chars inside character class
                match c {
                    ']' | '\\' | '^' | '-' => {
                        result.push('\\');
                        result.push(*c);
                    }
                    _ => result.push(*c),
                }
            }
            CharRange::Range(start, end) => {
                result.push(*start);
                result.push('-');
                result.push(*end);
            }
        }
    }
    result.push(']');
    result
}

// ─── Pushdown Automaton Grammar ──────────────────────────────────────────

/// PDA-based grammar for recursive GBNF grammars.
///
/// Uses a stack-based state machine to handle recursive rule references.
/// Bitmask computation is done on-demand per (rule, position) state by
/// checking which tokens are valid continuations.
struct PdaGrammar {
    grammar: GbnfGrammar,
    vocab_index: Arc<VocabularyIndex>,
    /// Current position in the grammar: stack of (rule_name, alt_index, elem_index)
    stack: Vec<PdaFrame>,
    /// Generated bytes so far (for matching)
    generated: Vec<u8>,
    /// History for rollback: each entry is (stack_snapshot, generated_len)
    history: Vec<(Vec<PdaFrame>, usize)>,
    /// Whether the grammar has been fully matched
    terminated: bool,
}

#[derive(Debug, Clone)]
struct PdaFrame {
    _rule_name: String,
}

impl PdaGrammar {
    fn new(grammar: GbnfGrammar, vocab_index: Arc<VocabularyIndex>) -> anyhow::Result<Self> {
        let root_name = grammar
            .root_rule()
            .ok_or_else(|| anyhow::anyhow!("grammar has no rules"))?
            .name
            .clone();

        let stack = vec![PdaFrame {
            _rule_name: root_name,
        }];

        Ok(Self {
            grammar,
            vocab_index,
            stack,
            generated: Vec::new(),
            history: Vec::new(),
            terminated: false,
        })
    }

    /// Check if a token's bytes are valid at the current grammar position.
    ///
    /// Uses a trial-and-error approach: tries appending the token's bytes
    /// and checking if the result is still a valid prefix of the grammar.
    fn is_token_valid(&self, token_bytes: &[u8]) -> bool {
        if token_bytes.is_empty() {
            return false;
        }

        // Build candidate byte string
        let mut candidate = self.generated.clone();
        candidate.extend_from_slice(token_bytes);

        // Check if the candidate is a valid prefix of any derivation
        self.matches_prefix(&candidate)
    }

    /// Check if a byte sequence is a valid prefix of the grammar.
    fn matches_prefix(&self, bytes: &[u8]) -> bool {
        let text = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };

        let root = match self.grammar.root_rule() {
            Some(r) => r,
            None => return false,
        };

        self.rule_matches_prefix(root, text).is_some()
    }

    /// Check if a rule can match a prefix, returning remaining text if so.
    fn rule_matches_prefix<'a>(&self, rule: &GbnfRule, text: &'a str) -> Option<&'a str> {
        for alt in &rule.alternatives {
            if let Some(remaining) = self.sequence_matches_prefix(alt, text) {
                return Some(remaining);
            }
        }
        None
    }

    fn sequence_matches_prefix<'a>(
        &self,
        seq: &GbnfSequence,
        mut text: &'a str,
    ) -> Option<&'a str> {
        for elem in &seq.elements {
            match self.element_matches_prefix(elem, text) {
                Some(remaining) => text = remaining,
                None => return None,
            }
        }
        Some(text)
    }

    fn element_matches_prefix<'a>(&self, elem: &GbnfElement, text: &'a str) -> Option<&'a str> {
        match elem {
            GbnfElement::Literal(lit) => {
                if text.starts_with(lit.as_str()) {
                    Some(&text[lit.len()..])
                } else if lit.starts_with(text) {
                    // text is a prefix of the literal
                    Some("")
                } else {
                    None
                }
            }
            GbnfElement::CharClass(cc) => {
                if text.is_empty() {
                    return Some(text);
                }
                let ch = text.chars().next()?;
                if char_matches(cc, ch) {
                    Some(&text[ch.len_utf8()..])
                } else {
                    None
                }
            }
            GbnfElement::RuleRef(name) => {
                let rule = self.grammar.find_rule(name)?;
                self.rule_matches_prefix(rule, text)
            }
            GbnfElement::Group(alts) => {
                for alt in alts {
                    if let Some(remaining) = self.sequence_matches_prefix(alt, text) {
                        return Some(remaining);
                    }
                }
                None
            }
            GbnfElement::Repeat(inner, kind) => match kind {
                RepeatKind::ZeroOrMore | RepeatKind::Optional => {
                    // Try matching, fall back to not matching
                    if let Some(remaining) = self.element_matches_prefix(inner, text) {
                        if *kind == RepeatKind::ZeroOrMore && remaining != text {
                            // Try to match more
                            if let Some(r) = self.element_matches_prefix(elem, remaining) {
                                return Some(r);
                            }
                        }
                        Some(remaining)
                    } else {
                        Some(text) // Zero matches
                    }
                }
                RepeatKind::OneOrMore => {
                    // Must match at least once
                    let remaining = self.element_matches_prefix(inner, text)?;
                    // Try to match more (zero or more additional)
                    let zero_or_more = GbnfElement::Repeat(inner.clone(), RepeatKind::ZeroOrMore);
                    self.element_matches_prefix(&zero_or_more, remaining)
                }
            },
        }
    }
}

fn char_matches(cc: &CharClass, ch: char) -> bool {
    let in_class = cc.ranges.iter().any(|range| match range {
        CharRange::Single(c) => ch == *c,
        CharRange::Range(start, end) => ch >= *start && ch <= *end,
    });

    if cc.negated {
        !in_class
    } else {
        in_class
    }
}

impl StructuredOutputGrammar for PdaGrammar {
    fn accept_tokens(&mut self, tokens: &[u32]) -> bool {
        for &token_id in tokens {
            let token_bytes = self.vocab_index.token_bytes(token_id);
            if token_bytes.is_empty() {
                return false;
            }

            // Save state for rollback
            self.history
                .push((self.stack.clone(), self.generated.len()));

            if !self.is_token_valid(token_bytes) {
                // Undo the history push
                self.history.pop();
                return false;
            }

            self.generated.extend_from_slice(token_bytes);

            // Check if we've reached a complete match
            if let Ok(text) = std::str::from_utf8(&self.generated) {
                if let Some(root) = self.grammar.root_rule() {
                    if let Some(remaining) = self.rule_matches_prefix(root, text) {
                        if remaining.is_empty() {
                            // Could be terminated (full match)
                            self.terminated = true;
                        }
                    }
                }
            }
        }
        true
    }

    fn fill_bitmask(&self, bitmask: &mut PackedBitmask, batch_index: usize) {
        // Check each token against the current grammar state
        for (token_id, token_bytes) in self.vocab_index.iter() {
            if !token_bytes.is_empty() && self.is_token_valid(token_bytes) {
                bitmask.set_bit(batch_index, token_id as usize);
            }
        }
    }

    fn rollback(&mut self, num_tokens: usize) {
        for _ in 0..num_tokens {
            if let Some((stack, gen_len)) = self.history.pop() {
                self.stack = stack;
                self.generated.truncate(gen_len);
                self.terminated = false;
            }
        }
    }

    fn is_terminated(&self) -> bool {
        self.terminated
    }

    fn reset(&mut self) {
        let root_name = self
            .grammar
            .root_rule()
            .map(|r| r.name.clone())
            .unwrap_or_default();

        self.stack = vec![PdaFrame {
            _rule_name: root_name,
        }];
        self.generated.clear();
        self.history.clear();
        self.terminated = false;
    }
}

impl std::fmt::Debug for PdaGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdaGrammar")
            .field("stack_depth", &self.stack.len())
            .field("generated_len", &self.generated.len())
            .field("terminated", &self.terminated)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenizerWrapper;

    fn make_vocab_index(size: usize) -> Arc<VocabularyIndex> {
        let tok = TokenizerWrapper::for_testing(size);
        Arc::new(VocabularyIndex::from_tokenizer(&tok))
    }

    #[test]
    fn regular_grammar_to_regex() {
        let parsed = super::super::ebnf_parser::parse_gbnf(r#"root ::= "hello""#).unwrap();
        assert!(parsed.is_regular());
        let regex = grammar_to_regex(&parsed).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("world"));
    }

    #[test]
    fn regular_grammar_with_alternatives() {
        let parsed = super::super::ebnf_parser::parse_gbnf(r#"root ::= "yes" | "no""#).unwrap();
        let regex = grammar_to_regex(&parsed).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("yes"));
        assert!(re.is_match("no"));
        assert!(!re.is_match("maybe"));
    }

    #[test]
    fn regular_grammar_with_char_class() {
        let parsed = super::super::ebnf_parser::parse_gbnf("root ::= [a-z]+").unwrap();
        let regex = grammar_to_regex(&parsed).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("hello"));
        assert!(!re.is_match("Hello"));
        assert!(!re.is_match(""));
    }

    #[test]
    fn regular_grammar_with_rule_ref() {
        let parsed = super::super::ebnf_parser::parse_gbnf(
            r#"root ::= greeting
greeting ::= "hello""#,
        )
        .unwrap();
        let regex = grammar_to_regex(&parsed).unwrap();
        let re = regex::Regex::new(&format!("^(?:{regex})$")).unwrap();
        assert!(re.is_match("hello"));
    }

    #[test]
    fn ebnf_to_grammar_regular() {
        let vi = make_vocab_index(100);
        let result = ebnf_to_grammar(r#"root ::= "test""#, vi);
        assert!(result.is_ok());
    }

    #[test]
    fn ebnf_to_grammar_recursive() {
        let vi = make_vocab_index(100);
        let result = ebnf_to_grammar(r#"root ::= "(" root ")" | "x""#, vi);
        assert!(result.is_ok());
    }

    #[test]
    fn char_class_regex_conversion() {
        let cc = CharClass {
            negated: false,
            ranges: vec![CharRange::Range('a', 'z'), CharRange::Single('_')],
        };
        let regex = char_class_to_regex(&cc);
        assert_eq!(regex, "[a-z_]");
    }

    #[test]
    fn negated_char_class_regex_conversion() {
        let cc = CharClass {
            negated: true,
            ranges: vec![CharRange::Range('0', '9')],
        };
        let regex = char_class_to_regex(&cc);
        assert_eq!(regex, "[^0-9]");
    }
}
