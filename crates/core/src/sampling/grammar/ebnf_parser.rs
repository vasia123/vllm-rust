//! GBNF (GGML BNF) grammar parser using nom.
//!
//! Parses GBNF grammars into an AST of rules, which can then be
//! compiled into either regex (for regular grammars) or a pushdown
//! automaton (for recursive grammars).

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, multispace0, none_of, satisfy},
    combinator::{map, opt, value},
    multi::{many0, many1, separated_list1},
    sequence::{delimited, preceded},
    IResult,
};

/// A complete GBNF grammar consisting of named rules.
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    pub rules: Vec<GbnfRule>,
}

impl GbnfGrammar {
    /// Get the root rule (first rule, conventionally named "root").
    pub fn root_rule(&self) -> Option<&GbnfRule> {
        self.rules.first()
    }

    /// Find a rule by name.
    pub fn find_rule(&self, name: &str) -> Option<&GbnfRule> {
        self.rules.iter().find(|r| r.name == name)
    }

    /// Check if the grammar is regular (no recursive rule references).
    ///
    /// A grammar is regular if no rule transitively references itself.
    pub fn is_regular(&self) -> bool {
        use std::collections::{HashMap, HashSet};

        let rule_map: HashMap<&str, &GbnfRule> =
            self.rules.iter().map(|r| (r.name.as_str(), r)).collect();

        for rule in &self.rules {
            let mut visited = HashSet::new();
            if self.has_recursion(&rule.name, &rule_map, &mut visited) {
                return false;
            }
        }
        true
    }

    fn has_recursion(
        &self,
        name: &str,
        rules: &std::collections::HashMap<&str, &GbnfRule>,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        if !visited.insert(name.to_string()) {
            return true;
        }

        if let Some(rule) = rules.get(name) {
            for alt in &rule.alternatives {
                for elem in &alt.elements {
                    if let GbnfElement::RuleRef(ref_name) = elem {
                        if self.has_recursion(ref_name, rules, visited) {
                            return true;
                        }
                    }
                    // Also check inside groups
                    if let GbnfElement::Group(inner_alts) = elem {
                        for inner_alt in inner_alts {
                            for inner_elem in &inner_alt.elements {
                                if let GbnfElement::RuleRef(ref_name) = inner_elem {
                                    if self.has_recursion(ref_name, rules, visited) {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        visited.remove(name);
        false
    }
}

/// A single GBNF rule: `name ::= alt1 | alt2 | ...`
#[derive(Debug, Clone)]
pub struct GbnfRule {
    pub name: String,
    pub alternatives: Vec<GbnfSequence>,
}

/// A sequence of elements within one alternative.
#[derive(Debug, Clone)]
pub struct GbnfSequence {
    pub elements: Vec<GbnfElement>,
}

/// A single element in a GBNF sequence.
#[derive(Debug, Clone)]
pub enum GbnfElement {
    /// Literal string: `"hello"`
    Literal(String),
    /// Character class: `[a-z0-9]` or `[^a-z]`
    CharClass(CharClass),
    /// Reference to another rule: `rule_name`
    RuleRef(String),
    /// Grouped alternatives: `(alt1 | alt2)`
    Group(Vec<GbnfSequence>),
    /// Repeated element with kind: `elem*`, `elem+`, `elem?`
    Repeat(Box<GbnfElement>, RepeatKind),
}

/// Character class definition.
#[derive(Debug, Clone)]
pub struct CharClass {
    pub negated: bool,
    pub ranges: Vec<CharRange>,
}

/// A range of characters: single char or range (e.g., `a-z`).
#[derive(Debug, Clone)]
pub enum CharRange {
    Single(char),
    Range(char, char),
}

/// Repetition kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatKind {
    /// `*` — zero or more
    ZeroOrMore,
    /// `+` — one or more
    OneOrMore,
    /// `?` — zero or one
    Optional,
}

/// Parse a GBNF grammar string into a `GbnfGrammar`.
pub fn parse_gbnf(input: &str) -> anyhow::Result<GbnfGrammar> {
    // Strip comments (lines starting with #)
    let cleaned: String = input
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");

    let (remaining, rules) = many1(preceded(multispace0, parse_rule))(&cleaned)
        .map_err(|e| anyhow::anyhow!("GBNF parse error: {e}"))?;

    let remaining = remaining.trim();
    if !remaining.is_empty() {
        anyhow::bail!("unexpected trailing content: {remaining:?}");
    }

    Ok(GbnfGrammar { rules })
}

/// Parse a single rule: `name ::= alternatives`
fn parse_rule(input: &str) -> IResult<&str, GbnfRule> {
    let (input, name) = parse_identifier(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("::=")(input)?;
    let (input, _) = multispace0(input)?;
    let (input, alternatives) = parse_alternatives(input)?;
    let (input, _) = opt(char('\n'))(input)?;

    Ok((
        input,
        GbnfRule {
            name: name.to_string(),
            alternatives,
        },
    ))
}

/// Parse pipe-separated alternatives.
fn parse_alternatives(input: &str) -> IResult<&str, Vec<GbnfSequence>> {
    separated_list1(
        delimited(multispace0, char('|'), multispace0),
        parse_sequence,
    )(input)
}

/// Parse a sequence of elements.
fn parse_sequence(input: &str) -> IResult<&str, GbnfSequence> {
    let (input, elements) = many1(preceded(multispace0, parse_element_with_repeat))(input)?;
    Ok((input, GbnfSequence { elements }))
}

/// Parse an element with optional repetition suffix.
fn parse_element_with_repeat(input: &str) -> IResult<&str, GbnfElement> {
    let (input, base) = parse_base_element(input)?;
    let (input, repeat) = opt(parse_repeat_kind)(input)?;

    Ok((
        input,
        match repeat {
            Some(kind) => GbnfElement::Repeat(Box::new(base), kind),
            None => base,
        },
    ))
}

/// Parse repetition suffix: `*`, `+`, or `?`
fn parse_repeat_kind(input: &str) -> IResult<&str, RepeatKind> {
    alt((
        value(RepeatKind::ZeroOrMore, char('*')),
        value(RepeatKind::OneOrMore, char('+')),
        value(RepeatKind::Optional, char('?')),
    ))(input)
}

/// Parse a base element (without repetition).
fn parse_base_element(input: &str) -> IResult<&str, GbnfElement> {
    alt((parse_literal, parse_char_class, parse_group, parse_rule_ref))(input)
}

/// Parse a quoted literal: `"text"` or `'text'`
fn parse_literal(input: &str) -> IResult<&str, GbnfElement> {
    alt((
        map(
            delimited(char('"'), many0(parse_string_char('"')), char('"')),
            |chars| GbnfElement::Literal(chars.into_iter().collect()),
        ),
        map(
            delimited(char('\''), many0(parse_string_char('\'')), char('\'')),
            |chars| GbnfElement::Literal(chars.into_iter().collect()),
        ),
    ))(input)
}

/// Parse a single character inside a string literal, handling escapes.
fn parse_string_char(quote: char) -> impl Fn(&str) -> IResult<&str, char> {
    move |input: &str| {
        alt((
            preceded(
                char('\\'),
                alt((
                    value('\\', char('\\')),
                    value('"', char('"')),
                    value('\'', char('\'')),
                    value('\n', char('n')),
                    value('\r', char('r')),
                    value('\t', char('t')),
                )),
            ),
            none_of(&[quote, '\\'][..]),
        ))(input)
    }
}

/// Parse a character class: `[a-z0-9]` or `[^a-z]`
fn parse_char_class(input: &str) -> IResult<&str, GbnfElement> {
    let (input, _) = char('[')(input)?;
    let (input, negated) = opt(char('^'))(input)?;
    let (input, ranges) = many1(parse_char_range)(input)?;
    let (input, _) = char(']')(input)?;

    Ok((
        input,
        GbnfElement::CharClass(CharClass {
            negated: negated.is_some(),
            ranges,
        }),
    ))
}

/// Parse a character range within a character class.
fn parse_char_range(input: &str) -> IResult<&str, CharRange> {
    let (input, start) = parse_class_char(input)?;
    let (input, range_end) = opt(preceded(char('-'), parse_class_char))(input)?;

    Ok((
        input,
        match range_end {
            Some(end) => CharRange::Range(start, end),
            None => CharRange::Single(start),
        },
    ))
}

/// Parse a single character inside a character class.
fn parse_class_char(input: &str) -> IResult<&str, char> {
    alt((
        preceded(
            char('\\'),
            alt((
                value('\\', char('\\')),
                value(']', char(']')),
                value('[', char('[')),
                value('-', char('-')),
                value('\n', char('n')),
                value('\r', char('r')),
                value('\t', char('t')),
            )),
        ),
        satisfy(|c| c != ']' && c != '\\'),
    ))(input)
}

/// Parse a grouped expression: `( alternatives )`
fn parse_group(input: &str) -> IResult<&str, GbnfElement> {
    let (input, _) = char('(')(input)?;
    let (input, _) = multispace0(input)?;
    let (input, alts) = parse_alternatives(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(')')(input)?;

    Ok((input, GbnfElement::Group(alts)))
}

/// Parse a rule reference (identifier), but not if it's followed by `::=`
/// (which would indicate the start of a new rule definition).
fn parse_rule_ref(input: &str) -> IResult<&str, GbnfElement> {
    let (remaining, name) = parse_identifier(input)?;
    // Peek ahead: if `::=` follows, this is a new rule definition, not a reference
    let after_ws = remaining.trim_start();
    if after_ws.starts_with("::=") {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Verify,
        )));
    }
    Ok((remaining, GbnfElement::RuleRef(name.to_string())))
}

/// Parse an identifier: `[a-zA-Z_][a-zA-Z0-9_-]*`
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    let (input, _) = take_while(|c: char| c == ' ' || c == '\t')(input)?;
    let start = input;
    let (input, _first) = satisfy(|c| c.is_ascii_alphabetic() || c == '_')(input)?;
    let (input, _rest) =
        take_while(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-')(input)?;
    let ident = &start[..start.len() - input.len()];
    Ok((input, ident))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_literal_rule() {
        let grammar = parse_gbnf(r#"root ::= "hello""#).unwrap();
        assert_eq!(grammar.rules.len(), 1);
        assert_eq!(grammar.rules[0].name, "root");
        assert_eq!(grammar.rules[0].alternatives.len(), 1);
        match &grammar.rules[0].alternatives[0].elements[0] {
            GbnfElement::Literal(s) => assert_eq!(s, "hello"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }

    #[test]
    fn parse_alternatives() {
        let grammar = parse_gbnf(r#"root ::= "yes" | "no""#).unwrap();
        assert_eq!(grammar.rules[0].alternatives.len(), 2);
    }

    #[test]
    fn parse_char_class() {
        let grammar = parse_gbnf("root ::= [a-z]+").unwrap();
        let elem = &grammar.rules[0].alternatives[0].elements[0];
        match elem {
            GbnfElement::Repeat(inner, RepeatKind::OneOrMore) => match inner.as_ref() {
                GbnfElement::CharClass(cc) => {
                    assert!(!cc.negated);
                    assert_eq!(cc.ranges.len(), 1);
                }
                other => panic!("expected CharClass, got {other:?}"),
            },
            other => panic!("expected Repeat, got {other:?}"),
        }
    }

    #[test]
    fn parse_negated_char_class() {
        let grammar = parse_gbnf("root ::= [^a-z]").unwrap();
        let elem = &grammar.rules[0].alternatives[0].elements[0];
        match elem {
            GbnfElement::CharClass(cc) => assert!(cc.negated),
            other => panic!("expected CharClass, got {other:?}"),
        }
    }

    #[test]
    fn parse_rule_reference() {
        let grammar = parse_gbnf(
            r#"root ::= greeting
greeting ::= "hello""#,
        )
        .unwrap();
        assert_eq!(grammar.rules.len(), 2);
        match &grammar.rules[0].alternatives[0].elements[0] {
            GbnfElement::RuleRef(name) => assert_eq!(name, "greeting"),
            other => panic!("expected RuleRef, got {other:?}"),
        }
    }

    #[test]
    fn parse_repetition() {
        let grammar = parse_gbnf(r#"root ::= "a"* "b"+ "c"?"#).unwrap();
        let elems = &grammar.rules[0].alternatives[0].elements;
        assert_eq!(elems.len(), 3);
        assert!(matches!(
            &elems[0],
            GbnfElement::Repeat(_, RepeatKind::ZeroOrMore)
        ));
        assert!(matches!(
            &elems[1],
            GbnfElement::Repeat(_, RepeatKind::OneOrMore)
        ));
        assert!(matches!(
            &elems[2],
            GbnfElement::Repeat(_, RepeatKind::Optional)
        ));
    }

    #[test]
    fn parse_group() {
        let grammar = parse_gbnf(r#"root ::= ("a" | "b")+"#).unwrap();
        let elem = &grammar.rules[0].alternatives[0].elements[0];
        match elem {
            GbnfElement::Repeat(inner, RepeatKind::OneOrMore) => {
                assert!(matches!(inner.as_ref(), GbnfElement::Group(_)));
            }
            other => panic!("expected Repeat(Group), got {other:?}"),
        }
    }

    #[test]
    fn parse_comments_stripped() {
        let grammar = parse_gbnf(
            r#"# This is a comment
root ::= "hello"
# Another comment"#,
        )
        .unwrap();
        assert_eq!(grammar.rules.len(), 1);
    }

    #[test]
    fn is_regular_non_recursive() {
        let grammar = parse_gbnf(
            r#"root ::= greeting
greeting ::= "hello""#,
        )
        .unwrap();
        assert!(grammar.is_regular());
    }

    #[test]
    fn is_not_regular_recursive() {
        let grammar = parse_gbnf(r#"root ::= "(" root ")" | "x""#).unwrap();
        assert!(!grammar.is_regular());
    }

    #[test]
    fn escape_sequences_in_literal() {
        let grammar = parse_gbnf(r#"root ::= "hello\nworld""#).unwrap();
        match &grammar.rules[0].alternatives[0].elements[0] {
            GbnfElement::Literal(s) => assert_eq!(s, "hello\nworld"),
            other => panic!("expected Literal, got {other:?}"),
        }
    }
}
