use std::cmp::Ordering;
use std::fs::File;
use std::{array, io};
use std::io::{BufReader, Read, Result};
use std::path::PathBuf;

use crate::core::types::{TokenIndex, Tokenizer};

fn get_str_from_vocab(vocab: &[TokenIndex], idx: usize) -> &str {
    let item_vec: Vec<&str> = vocab.iter().filter(
        |v| v.id == idx
    ).map(|v| v.string.as_str()).collect();
    item_vec[0]
}

impl Ord for TokenIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.string.cmp(&other.string)
    }
}

impl PartialOrd for TokenIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Tokenizer {
    pub fn load_tokenizer(tokenizer_path: PathBuf, vocab_size: usize) -> Result<Self> {
        let file = File::open(tokenizer_path)?;
        let mut reader = BufReader::new(file);

        let mut max_token_length = [0u8; 4];
        reader.read_exact(&mut max_token_length)?;
        let max_token_length = u32::from_ne_bytes(max_token_length) as usize;

        let mut sorted_vocab: Vec<TokenIndex> = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for i in 0..vocab_size {
            let mut score_bytes = [0u8; 4];
            reader.read_exact(&mut score_bytes)?;
            let score = f32::from_ne_bytes(score_bytes);
            vocab_scores.push(score);

            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_ne_bytes(len_bytes) as usize;

            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let word = String::from_utf8(word)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            sorted_vocab.push(
                TokenIndex {
                    string: word,
                    id: i
                }
            )
        }
        sorted_vocab.sort();

        let byte_pieces: [u8; 256] = array::from_fn(|i| i as u8);

        Ok(Tokenizer {
            vocab_scores,
            max_token_length,
            vocab_size,
            byte_pieces,
            sorted_vocab
        })
    }
    
    pub fn decode(&self, prev_token: usize, token: usize) -> &str {
        let mut piece = get_str_from_vocab(&self.sorted_vocab, token);

        // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
        if prev_token == 1 && piece.starts_with(' ') {
            piece = &piece[1..];
        }

        // Check if the token designates a raw byte
        if let Some(byte_val) = Self::parse_byte_token(piece) {
            let start = byte_val as usize;
            piece = std::str::from_utf8(&self.byte_pieces[start..start + 1])
                .unwrap_or_else(|_| panic!("Invalid UTF-8 sequence in byte_pieces: {}: {}", piece, byte_val));
        }

        piece
    }

    fn parse_byte_token(token: &str) -> Option<u8> {
        if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
            let byte = u8::from_str_radix(&token[3..5], 16).unwrap();
            if byte <= 127 {
                Some(byte)
            } else {
                println!("{}", byte);
                None
            }
        } else {
            None
        }
    }

    pub fn safe_print(piece: &str) {
        // piece might be a raw byte token, and
        // we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if piece.is_empty() {
            return;
        }
        if piece.len() == 1 {
            let byte_val = piece.as_bytes()[0];
            if !byte_val.is_ascii_whitespace() && !byte_val.is_ascii_graphic() {
                return;
            }
        }
        print!("{}", piece);
    }

    fn str_lookup(&self, str: &String) -> Option<usize> {
        // efficiently find the perfect match for str in vocab,
        // return its index or None if not found
        self.sorted_vocab.binary_search_by(|token| token.string.cmp(str))
            .ok()
            .map(|index| self.sorted_vocab[index].id)
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {

        let mut tokens = Vec::new();

        // add optional BOS (=1) token, if desired
        if bos {
            tokens.push(1);
        }
        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text is non-empty
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.is_empty() {
            if let Some(dummy_prefix) = self.str_lookup(&" ".to_string()) {
                tokens.push(dummy_prefix);
            }
        }

        let mut str_buffer = Vec::with_capacity(4);

        for c in text.bytes() {
            // Reset buffer if the current byte is ASCII or a leading byte
            if (c & 0xC0) != 0x80 {
                str_buffer.clear();
            }

            // Append the current byte to the buffer
            str_buffer.push(c);

            // Continue if the next byte is a continuation byte and buffer isn't full
            if text.as_bytes().get(str_buffer.len()).map_or(false, |&next_c| (next_c & 0xC0) == 0x80)
                && str_buffer.len() < 4 {
                continue;
            }
            
            let buff_str = std::str::from_utf8(&str_buffer).unwrap().to_string();

            // We've read in a full codepoint
            if let Some(id) = self.str_lookup(&buff_str) {
                // We found this codepoint in vocab, add it as a token
                tokens.push(id);
            } else {
                // Byte fallback encoding: encode each byte as a token
                tokens.extend(str_buffer.iter().map(|&b| b as usize + 3));
            }

            str_buffer.clear(); // Protect against a sequence of stray UTF8 continuation bytes
        }

        if !str_buffer.is_empty() {
            let buff_str = std::str::from_utf8(&str_buffer).unwrap();
            for &byte in buff_str.as_bytes() {
                tokens.push(byte as usize + 3);
            }
        }

        // merge the best consecutive pair each iteration, 
        // according the scores in vocab_scores
        loop {
            let mut best_merge = None;

            for i in 0..tokens.len() - 1 {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                let merged = format!(
                    "{}{}",
                    get_str_from_vocab(&self.sorted_vocab, tokens[i]),
                    get_str_from_vocab(&self.sorted_vocab, tokens[i + 1])
                );
                if let Some(id) = self.str_lookup(&merged) {
                    let score = self.vocab_scores[id];
                    if best_merge.map_or(true, |(_, best_score, _)| score > best_score) {
                        // this merge pair exists in vocab! record its score and position
                        best_merge = Some((i, score, id));
                    }
                }
            }

            match best_merge {
                Some((idx, _, id)) => {
                    // merge the consecutive pair (idx, idx+1) into new token
                    tokens[idx] = id;
                    // delete token at position idx+1, shift the entire sequence back 1
                    tokens.remove(idx + 1);
                }
                // we couldn't find any more pairs to merge, so we're done
                None => break,
            }
        }

        // add optional EOS (=2) token, if desired
        if eos {
            tokens.push(2);
        }
        
        tokens
    }
}
