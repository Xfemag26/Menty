// speculative.rs - Zero-overhead Speculative Decoding (Prompt Lookup + Kangaroo Draft)

pub struct PromptLookupSpeculator {
    pub ngram_size: usize,
    pub max_draft_tokens: usize,
}

impl PromptLookupSpeculator {
    pub fn new(ngram_size: usize, max_draft_tokens: usize) -> Self {
        PromptLookupSpeculator {
            ngram_size,
            max_draft_tokens,
        }
    }

    /// Finds matching n-gram suffix in historical context and drafts continuation tokens
    pub fn draft_tokens(&self, context: &[u32]) -> Vec<u32> {
        if context.len() < self.ngram_size + 1 {
            return Vec::new();
        }

        let query = &context[context.len() - self.ngram_size..];
        let search_len = context.len() - self.ngram_size;

        // Search backward for best matching ngram prefix
        for i in (0..search_len).rev() {
            if &context[i..i + self.ngram_size] == query {
                let match_end = i + self.ngram_size;
                let available = context.len() - 1 - match_end;
                let draft_len = available.min(self.max_draft_tokens);

                if draft_len > 0 {
                    return context[match_end..match_end + draft_len].to_vec();
                }
            }
        }

        Vec::new()
    }

    /// Verifies draft tokens against target model predictions
    /// Returns (accepted_tokens, next_token)
    pub fn verify(&self, draft_tokens: &[u32], model_predictions: &[u32]) -> (Vec<u32>, u32) {
        let mut accepted = Vec::new();
        for (i, &draft) in draft_tokens.iter().enumerate() {
            if i < model_predictions.len() && draft == model_predictions[i] {
                accepted.push(draft);
            } else {
                let next = if i < model_predictions.len() {
                    model_predictions[i]
                } else {
                    draft
                };
                return (accepted, next);
            }
        }

        let next = *model_predictions.last().unwrap_or(&0);
        (accepted, next)
    }
}
