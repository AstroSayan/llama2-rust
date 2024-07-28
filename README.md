# Llama2-rust

This project is the Rust implementation of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Project Status

- At very initial stage.
- Able to generate at ~180 tk/s rate using `stories15M.bin` model.

## Needs fix
- Producing gibberish tokens till now.
- Failing intermittently due to decoding a non-utf8 token.

## Pending Tasks

- Yet to implement `chat` functionality.
- Yet to implement CLI functionalities.

### To run the application
- Make sure you are in project base directory.
- Then run `cargo run src\main.rs`
