// tokenizer.rs - tokenizer implementations for GGUF SentencePiece and GPT-2 BPE
use std::collections::HashMap;

use crate::gguf::{GGUFFile, MetadataValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerKind {
    Gpt2Bpe,
    SentencePiece,
}

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    kind: TokenizerKind,
    merges: HashMap<(String, String), usize>,
    special_tokens: Vec<(String, u32)>,
    byte_decoder: HashMap<char, u8>,
    add_bos: bool,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, String> {
        let tokens_val = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or("No tokenizer.ggml.tokens metadata found in GGUF")?;
        let vocab_arr = match tokens_val {
            MetadataValue::Array(arr) => arr,
            _ => return Err("tokenizer.ggml.tokens is not an array".into()),
        };

        let mut vocab = Vec::with_capacity(vocab_arr.len());
        let mut token_to_id = HashMap::with_capacity(vocab_arr.len());
        for (id, val) in vocab_arr.iter().enumerate() {
            let token = match val {
                MetadataValue::String(s) => s.clone(),
                _ => return Err(format!("tokenizer vocabulary entry {id} is not a string")),
            };
            token_to_id.insert(token.clone(), id as u32);
            vocab.push(token);
        }

        let model_name = gguf.get_str_metadata("tokenizer.ggml.model").unwrap_or("");
        let kind = if model_name.eq_ignore_ascii_case("gpt2")
            || model_name.eq_ignore_ascii_case("qwen2")
        {
            TokenizerKind::Gpt2Bpe
        } else {
            TokenizerKind::SentencePiece
        };

        let mut merges = HashMap::new();
        if let Some(MetadataValue::Array(items)) = gguf.metadata.get("tokenizer.ggml.merges") {
            for (rank, item) in items.iter().enumerate() {
                if let MetadataValue::String(merge) = item {
                    if let Some((left, right)) = merge.split_once(' ') {
                        merges.insert((left.to_owned(), right.to_owned()), rank);
                    }
                }
            }
        }
        if kind == TokenizerKind::Gpt2Bpe && merges.is_empty() {
            return Err("GPT-2-style tokenizer has no tokenizer.ggml.merges table".into());
        }

        let bos_id = gguf
            .get_u32_metadata("tokenizer.ggml.bos_token_id")
            .unwrap_or(1);
        let eos_id = gguf
            .get_u32_metadata("tokenizer.ggml.eos_token_id")
            .unwrap_or(2);
        let add_bos = match gguf.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(MetadataValue::Bool(value)) => *value,
            _ => kind == TokenizerKind::SentencePiece,
        };

        let special_tokens = vocab
            .iter()
            .enumerate()
            .filter(|(_, token)| is_special_token(token))
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<Vec<_>>();
        let mut special_tokens = special_tokens;
        special_tokens.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));

        Ok(Self {
            vocab,
            token_to_id,
            bos_id,
            eos_id,
            kind,
            merges,
            special_tokens,
            byte_decoder: build_byte_decoder(),
            add_bos,
        })
    }

    /// Encode user text according to the tokenizer declared in the GGUF metadata.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_impl(text, self.add_bos)
    }

    /// Tokenize text without control tokens for retrieval-vector pooling.
    pub fn encode_retrieval(&self, text: &str) -> Vec<u32> {
        self.encode_impl(text, false)
    }

    fn encode_impl(&self, text: &str, include_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if include_bos && self.add_bos {
            out.push(self.bos_id);
        }

        let mut rest = text;
        while !rest.is_empty() {
            if let Some((special, id)) = self
                .special_tokens
                .iter()
                .find(|(special, _)| rest.starts_with(special))
            {
                out.push(*id);
                rest = &rest[special.len()..];
                continue;
            }

            let next_special = self
                .special_tokens
                .iter()
                .filter_map(|(special, _)| rest.find(special))
                .min()
                .unwrap_or(rest.len());
            if next_special == 0 {
                // A malformed/overlapping special-token entry must not stall encoding.
                if let Some(ch) = rest.chars().next() {
                    self.encode_plain(&rest[..ch.len_utf8()], &mut out);
                    rest = &rest[ch.len_utf8()..];
                }
            } else {
                self.encode_plain(&rest[..next_special], &mut out);
                rest = &rest[next_special..];
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        match self.kind {
            TokenizerKind::Gpt2Bpe => {
                for piece in gpt2_pretokens(text) {
                    let encoded: String = piece
                        .as_bytes()
                        .iter()
                        .map(|byte| byte_to_unicode(*byte))
                        .collect();
                    let mut symbols = encoded.chars().map(|c| c.to_string()).collect::<Vec<_>>();
                    self.apply_bpe(&mut symbols);
                    for symbol in symbols {
                        if let Some(id) = self.token_to_id.get(&symbol) {
                            out.push(*id);
                        } else {
                            self.push_gpt2_symbol_fallback(&symbol, out);
                        }
                    }
                }
            }
            TokenizerKind::SentencePiece => {
                // GGUF SentencePiece vocabularies encode word boundaries as U+2581.
                let normalized = text.replace(' ', "▁");
                self.encode_sentencepiece(&normalized, out);
            }
        }
    }

    fn apply_bpe(&self, symbols: &mut Vec<String>) {
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    self.merges
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (index, *rank))
                })
                .min_by_key(|(_, rank)| *rank);
            let Some((index, _)) = best else { break };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols.splice(index..=index + 1, [merged]);
        }
    }

    fn encode_sentencepiece(&self, text: &str, out: &mut Vec<u32>) {
        let chars = text.chars().collect::<Vec<_>>();
        let mut offset = 0;
        while offset < chars.len() {
            let max_len = (chars.len() - offset).min(128);
            let mut found = None;
            for len in (1..=max_len).rev() {
                let piece = chars[offset..offset + len].iter().collect::<String>();
                if let Some(id) = self.token_to_id.get(&piece) {
                    found = Some((len, *id));
                    break;
                }
            }
            if let Some((len, id)) = found {
                out.push(id);
                offset += len;
            } else {
                let ch = chars[offset];
                let mut buf = [0u8; 4];
                self.push_byte_fallback(ch.encode_utf8(&mut buf).as_bytes(), out);
                offset += 1;
            }
        }
    }

    fn push_gpt2_symbol_fallback(&self, symbol: &str, out: &mut Vec<u32>) {
        for ch in symbol.chars() {
            if let Some(byte) = self.byte_decoder.get(&ch) {
                let fallback = format!("<0x{byte:02X}>");
                if let Some(id) = self.token_to_id.get(&fallback) {
                    out.push(*id);
                }
            }
        }
    }

    fn push_byte_fallback(&self, bytes: &[u8], out: &mut Vec<u32>) {
        for byte in bytes {
            let fallback = format!("<0x{byte:02X}>");
            if let Some(id) = self.token_to_id.get(&fallback) {
                out.push(*id);
            } else if self.kind == TokenizerKind::Gpt2Bpe {
                let piece = byte_to_unicode(*byte).to_string();
                if let Some(id) = self.token_to_id.get(&piece) {
                    out.push(*id);
                }
            }
        }
    }

    pub fn decode(&self, id: u32) -> &str {
        self.vocab
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// Decode complete output, replacing any invalid trailing byte sequence.
    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        let (bytes, sentencepiece) = self.decode_bytes(tokens);
        let decoded = String::from_utf8_lossy(&bytes).into_owned();
        if sentencepiece {
            decoded.replace('▁', " ")
        } else {
            decoded
        }
    }

    /// Decode only the UTF-8-complete prefix, so token streaming doesn't emit
    /// replacement characters for a multibyte code point still being generated.
    pub fn decode_tokens_stream(&self, tokens: &[u32]) -> String {
        let (bytes, sentencepiece) = self.decode_bytes(tokens);
        let valid_len = match std::str::from_utf8(&bytes) {
            Ok(_) => bytes.len(),
            Err(error) => error.valid_up_to(),
        };
        let decoded = String::from_utf8_lossy(&bytes[..valid_len]).into_owned();
        if sentencepiece {
            decoded.replace('▁', " ")
        } else {
            decoded
        }
    }

    fn decode_bytes(&self, tokens: &[u32]) -> (Vec<u8>, bool) {
        let mut bytes = Vec::new();
        for &id in tokens {
            let Some(piece) = self.vocab.get(id as usize) else {
                continue;
            };
            if self.kind == TokenizerKind::Gpt2Bpe {
                for ch in piece.chars() {
                    if let Some(byte) = self.byte_decoder.get(&ch) {
                        bytes.push(*byte);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            } else {
                let piece = piece.replace('▁', " ");
                bytes.extend_from_slice(piece.as_bytes());
            }
        }
        (bytes, self.kind == TokenizerKind::SentencePiece)
    }
}

fn is_special_token(token: &str) -> bool {
    (token.starts_with("<|") && token.ends_with("|>"))
        || matches!(token, "<s>" | "</s>" | "[INST]" | "[/INST]")
}

fn build_byte_decoder() -> HashMap<char, u8> {
    (0u16..=255)
        .map(|byte| (byte_to_unicode(byte as u8), byte as u8))
        .collect()
}

/// GPT-2's reversible byte-to-Unicode mapping used by Qwen2 tokenizers.
fn byte_to_unicode(byte: u8) -> char {
    static MAPPING: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    MAPPING.get_or_init(|| {
        let visible = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .map(|value| value as u16)
            .collect::<Vec<_>>();
        let mut mapping = ['\0'; 256];
        let mut extra = 0u16;
        for value in 0u16..=255 {
            let code = if visible.contains(&value) {
                value
            } else {
                let code = 256 + extra;
                extra += 1;
                code
            };
            mapping[value as usize] = char::from_u32(code as u32).unwrap_or('\u{FFFD}');
        }
        mapping
    })[byte as usize]
}

/// GPT-2/Qwen byte-level BPE pre-tokenization pattern, compiled once.
fn gpt2_pretokens(text: &str) -> Vec<String> {
    static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = PATTERN.get_or_init(|| regex::Regex::new(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
    ).expect("static GPT-2 tokenizer regex is valid"));
    re.find_iter(text).map(|m| m.as_str().to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> Tokenizer {
        let vocab = vec!["h", "e", "l", "o", "he", "hel", "hell", "hello"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let token_to_id = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
            .iter()
            .enumerate()
            .map(|(rank, (left, right))| ((left.to_string(), right.to_string()), rank))
            .collect();
        Tokenizer {
            vocab,
            token_to_id,
            bos_id: 0,
            eos_id: 0,
            kind: TokenizerKind::Gpt2Bpe,
            merges,
            special_tokens: Vec::new(),
            byte_decoder: build_byte_decoder(),
            add_bos: false,
        }
    }

    #[test]
    fn gpt2_bpe_applies_ranked_merges() {
        let tokenizer = test_tokenizer();
        assert_eq!(tokenizer.encode("hello"), vec![7]);
    }

    #[test]
    fn byte_mapping_round_trips_utf8() {
        let tokenizer = test_tokenizer();
        let encoded = "é"
            .as_bytes()
            .iter()
            .map(|b| byte_to_unicode(*b))
            .collect::<String>();
        let ids = encoded
            .chars()
            .filter_map(|c| tokenizer.token_to_id.get(&c.to_string()).copied())
            .collect::<Vec<_>>();
        // The test vocabulary doesn't contain the bytes for é; verify the table's inverse directly.
        let inverse = build_byte_decoder();
        let bytes = encoded.chars().map(|c| inverse[&c]).collect::<Vec<_>>();
        assert_eq!(bytes, "é".as_bytes());
        assert!(ids.is_empty());
    }
}
