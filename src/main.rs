use std::env;

use core::types::{Sampler, Tokenizer, Transformer};
mod core;
#[cfg(test)]
mod test;

fn main() {
    let current_dir = env::current_dir().unwrap();
    let model_path = current_dir.join("models");
    println!("{:?}", current_dir);
    let mut transformer = Transformer::load_transformer_model(
        model_path.join("stories15M.bin")
    ).expect("TODO: Can not load");
    let vocab_size = transformer.config.vocab_size as usize;
    let tokenizer = Tokenizer::load_tokenizer(model_path.join("tokenizer.bin"), vocab_size).unwrap();
    let mut sampler = Sampler::new(vocab_size, 0.1, 0.5);
    let prompt = Some("One day, Lily met a Shoggoth");
    // let prompt = None;
    transformer.generate(&tokenizer, &mut sampler, prompt, 250);
}
