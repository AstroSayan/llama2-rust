use std::env;
use sentencepiece::SentencePieceProcessor;
use crate::core::types::Tokenizer;

#[test]
fn test_tokenizer() {
    let current_dir = env::current_dir().unwrap();
    let model_path = current_dir.join("models");
    let tokenizer = Tokenizer::load_tokenizer(model_path.join("tokenizer.bin"), 32000).unwrap();
    let spp = SentencePieceProcessor::open(model_path.join("tokenizer.model")).unwrap();
    let mut pieces = spp.encode("I saw a girl with a telescope.").unwrap()
        .into_iter().map(|p| p.piece).collect::<Vec<_>>();
    println!("{:?}", pieces);
    pieces.clear();
    let pieces_ids = tokenizer.encode("I saw a girl with a telescope.", true, false);
    let mut prev_token = pieces_ids[0];
    pieces_ids.iter()
        .for_each(|v| {
            pieces.push(tokenizer.decode(prev_token, *v).to_string());
            prev_token = *v
        });
    println!("{:?}", pieces);
}