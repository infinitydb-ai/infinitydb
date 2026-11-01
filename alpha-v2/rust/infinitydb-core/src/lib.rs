//! Dependency-free primitives for the InfinityDB v2 alpha.

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate<'a> {
    pub id: &'a str,
    pub content: &'a str,
    pub lexical_score: f32,
    pub semantic_score: f32,
    pub recency_score: f32,
    pub confidence: f32,
}

/// Weighted score fusion. The caller owns score normalization and weight policy.
pub fn fused_score(candidate: &Candidate<'_>, lexical: f32, semantic: f32, recency: f32) -> f32 {
    let total = lexical + semantic + recency;
    if total <= 0.0 { return 0.0; }
    let score = (candidate.lexical_score * lexical
        + candidate.semantic_score * semantic
        + candidate.recency_score * recency) / total;
    score.clamp(0.0, 1.0) * candidate.confidence.clamp(0.0, 1.0)
}

/// Conservative character-based estimate, not a model-specific tokenizer.
pub fn estimate_tokens(text: &str) -> usize {
    ((text.chars().count() + 3) / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_respects_confidence() {
        let candidate = Candidate { id: "x", content: "hello", lexical_score: 1.0,
            semantic_score: 0.5, recency_score: 0.5, confidence: 0.5 };
        assert!((fused_score(&candidate, 1.0, 1.0, 0.0) - 0.375).abs() < 0.0001);
    }

    #[test]
    fn empty_token_estimate_is_nonzero() {
        assert_eq!(estimate_tokens(""), 1);
    }
}
