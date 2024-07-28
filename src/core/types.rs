// Transformer related structs
#[derive(Debug)]
pub struct Config {
    pub dim: i32,         // transformer dimension
    pub hidden_dim: i32,  // for FFN layers
    pub n_layers: i32,    // number of layers
    pub n_heads: i32,     // number of query heads
    pub n_kv_heads: i32,  // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: i32,  // vocabulary size, usually 256 (byte-level)
    pub seq_len: i32      // max sequence length
}

pub struct TransformerWeights {
    // token embedding table
    pub token_embedding_table: Vec<f32>,  // (vocab_size, dim)
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>,         // (layer, dim) rmsnorm weights
    pub rms_ffn_weight: Vec<f32>,         // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<f32>,                     // (layer, dim, n_heads * head_size)
    pub wk: Vec<f32>,                     // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<f32>,                     // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<f32>,                     // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<f32>,                     // (layer, hidden_dim, dim)
    pub w2: Vec<f32>,                     // (layer, dim, hidden_dim)
    pub w3: Vec<f32>,                     // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>,       // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Vec<f32>
}

pub struct RunState {
    // current wave of activations
    pub x: Vec<f32>,            // activation at current time stamp (dim,)
    pub xb: Vec<f32>,           // same, but inside a residual branch (dim,)
    pub xb2: Vec<f32>,          // an additional buffer just for convenience (dim,)
    pub hb: Vec<f32>,           // buffer for hidden dimension in the ffn (hidden_dim,)
    pub hb2: Vec<f32>,          // buffer for hidden dimension in the ffn (hidden_dim,)
    pub q: Vec<f32>,            // query (dim,)
    pub k: Vec<f32>,            // key (dim,)
    pub v: Vec<f32>,            // value (dim,)
    pub att: Vec<f32>,          // buffer for scores/attention values (n_heads, seq_len)
    pub logits: Vec<f32>,       // output logits
    // kv cache
    pub key_cache: Vec<f32>,    // (layer, seq_len, dim)
    pub value_cache: Vec<f32>,  // (layer, seq_len, dim)
}

pub struct Transformer {
    pub config: Config,               // the hyperparameters of the architecture (the blueprint)
    pub weights: TransformerWeights,  // the weights of the model
    pub state: RunState,              // buffers for the "wave" of activations in the forward pass
    // pub data: Vec<f32>,            // memory mapped data pointer
    pub file_size: u64                // size of the checkpoint file in bytes
}

// -------------------------------------------------------------------------------------------------
// Tokenizer related structs
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
#[derive(Clone, Eq, PartialEq)]
pub struct TokenIndex {
    pub string: String,
    pub id: usize
}

pub struct Tokenizer {
    // pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub sorted_vocab: Vec<TokenIndex>,
    pub vocab_size: usize,
    pub max_token_length: usize,
    pub byte_pieces: [u8; 256]  // stores all single-byte strings
}

// -------------------------------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// struct used when sorting probabilities during top-p sampling
#[derive(Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct ProbIndex {
    pub prob: f32,
    pub index: usize
}

pub struct Sampler {
    pub vocab_size: usize,
    pub prob_index: Vec<ProbIndex>,  // buffer used in top-p sampling
    pub temperature: f32,
    pub top_p: f32
}
