use std::fs::File;
use std::io;
use std::io::{Read, Result, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::time::Instant;

use crate::core::nnblocks::*;

use super::types::{Config, RunState, Sampler, Tokenizer, Transformer, TransformerWeights};

fn convert_byte_to_int(byte_array: &[u8], end_index: &mut i32) -> Result<i32> {
    let start_index = *end_index as usize;
    *end_index += 4;
    Ok(i32::from_ne_bytes(
        byte_array[start_index..*end_index as usize].try_into().unwrap()
    ))
}

fn get_config(buffer: &[u8]) -> Result<Config> {
    let mut end_index: i32 = 0;
    Ok(Config {
        dim: convert_byte_to_int(buffer, &mut end_index)?,
        hidden_dim: convert_byte_to_int(buffer, &mut end_index)?,
        n_layers: convert_byte_to_int(buffer, &mut end_index)?,
        n_heads: convert_byte_to_int(buffer, &mut end_index)?,
        n_kv_heads: convert_byte_to_int(buffer, &mut end_index)?,
        vocab_size: convert_byte_to_int(buffer, &mut end_index)?,
        seq_len: convert_byte_to_int(buffer, &mut end_index)?,
    })
}

fn get_transformer_weights(
    config: Config, raw_data: Vec<u8>,
    shared_weights: bool) -> Result<(TransformerWeights, Config)> {
    let num_f32_values = raw_data.len() / 4;
    let data: Vec<f32> = (0..num_f32_values).map(
        |i| f32::from_ne_bytes(raw_data[i * 4..(i + 1) * 4].try_into().unwrap())
    ).collect();
    drop(raw_data);
    // defining TransformerWeights
    let head_size = config.dim as i64 / config.n_heads as i64;
    // Making sure the multiplications below are done in 64bit
    // to fit the parameter counts of 13B+ models
    let n_layers: i64 = i64::from(config.n_layers);
    let mut end_index: usize = 0;
    let end_index_ref = &mut end_index;
    let mut transformer_weights = TransformerWeights {
        token_embedding_table: data[
            *end_index_ref..*end_index_ref + (config.vocab_size as i64 * config.dim as i64) as usize
            ].to_vec(),
        rms_att_weight: data[
            *end_index_ref..*end_index_ref + (n_layers * config.dim as i64) as usize
            ].to_vec(),
        wq: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.dim as i64 * (config.n_heads as i64 * head_size)
            ) as usize
            ].to_vec(),
        wk: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.dim as i64 * (config.n_kv_heads as i64 * head_size)
            ) as usize
            ].to_vec(),
        wv: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.dim as i64 * (config.n_kv_heads as i64 * head_size)
            ) as usize
            ].to_vec(),
        wo: data[
            *end_index_ref..*end_index_ref + (
                n_layers * (config.n_heads as i64 * head_size) * config.dim as i64
            ) as usize
            ].to_vec(),
        rms_ffn_weight: data[
            *end_index_ref..*end_index_ref + (n_layers * config.dim as i64) as usize
            ].to_vec(),
        w1: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.dim as i64 * config.hidden_dim as i64
            ) as usize
            ].to_vec(),
        w2: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.hidden_dim as i64 * config.dim as i64
            ) as usize
            ].to_vec(),
        w3: data[
            *end_index_ref..*end_index_ref + (
                n_layers * config.dim as i64 * config.hidden_dim as i64
            ) as usize
            ].to_vec(),
        rms_final_weight: data[
            *end_index_ref..*end_index_ref + (config.dim as i64) as usize
            ].to_vec(),
        wcls: vec![]
    };
    transformer_weights.wcls = if shared_weights {
        transformer_weights.token_embedding_table.clone()
    } else {
        data[
            *end_index_ref..*end_index_ref
                // skip what used to be freq_cis_real (for RoPE)
                + (config.seq_len as i64 * head_size / 2) as usize
                // skip what used to be freq_cis_imag (for RoPE)
                + (config.seq_len as i64 * head_size / 2) as usize
            ].to_vec()
    };
    Ok((transformer_weights, config))
}

fn init_runstate(config: &Config) -> Result<RunState> {
    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    Ok(RunState {
        x: vec![0f32; config.dim as usize],
        xb: vec![0f32; config.dim as usize],
        xb2: vec![0f32; config.dim as usize],
        hb: vec![0f32; config.hidden_dim as usize],
        hb2: vec![0f32; config.hidden_dim as usize],
        q: vec![0f32; config.dim as usize],
        k: vec![0f32; config.dim as usize],
        v: vec![0f32; config.dim as usize],
        att: vec![0f32; (config.n_heads * config.seq_len) as usize],
        logits: vec![0f32; config.vocab_size as usize],
        key_cache: vec![0f32; (config.n_layers * config.seq_len * kv_dim) as usize],
        value_cache: vec![0f32; (config.n_layers * config.seq_len * kv_dim) as usize],
    })
}

fn read_checkpoint(filepath: PathBuf) -> Result<(Config, TransformerWeights, u64)> {
    let mut file = File::open(filepath)?;
    let mut buffer = [0u8; size_of::<Config>()];
    file.read_exact(&mut buffer)?;

    let mut config: Config = get_config(&buffer)?;
    println!("{:?}", config);
    let shared_weights = config.vocab_size > 0;
    config.vocab_size = config.vocab_size.abs();
    let file_size = file.metadata().unwrap().len();

    // let raw_data: Mmap = unsafe {
    //   Mmap::map(&file)?
    // };

    let mut raw_data = Vec::new();
    file.read_to_end(&mut raw_data)?;
    drop(file);
    // defining TransformerWeights
    let (transformer_weights, config) = get_transformer_weights(
        config, raw_data, shared_weights)?;
    Ok((config, transformer_weights, file_size))
}

impl Transformer {
    pub fn load_transformer_model(filepath: PathBuf) -> Result<Self> {
        let (config, weights, file_size) = read_checkpoint(filepath)?;
        let state = init_runstate(&config)?;
        
        Ok(Self {
            config,
            weights,
            state,
            file_size
        })
    }
    
    pub fn forward(&mut self, token: usize, pos: usize) -> &mut [f32] {
        let p = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let dim = p.dim as usize;
        let kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        let kv_mul = (p.n_heads / p.n_kv_heads) as usize;
        let hidden_dim = p.hidden_dim;
        let head_size = dim / p.n_heads as usize;

        // Copy the token embedding into x
        s.x.copy_from_slice(&w.token_embedding_table[
            token * dim..(token + 1) * dim]
        );

        // Forward all the layers
        for l in 0..p.n_layers as usize{
            // Attention rms normalization
            // X_b = rmsnorm(X, W_att_rms)
            rmsnorm(&mut s.xb, &s.x, &w.rms_att_weight[
                l * dim..(l + 1) * dim]
            );

            // Key and value point to the kv cache
            // kv cache layer offset for convenience
            let loff = l * (p.seq_len * kv_dim) as usize;
            let k_start = loff + (pos * kv_dim as usize);
            let v_start = k_start;

            // qkv matmuls for this position
            // Q = X_b @ W_q
            matmul(&mut s.q, &s.xb, &w.wq[
                l * dim * dim..(l + 1) * dim * dim
                ], dim, dim
            );
            // K_c = X_b @ W_k
            matmul(&mut s.key_cache[k_start..k_start + kv_dim as usize],
                   &s.xb, &w.wk[l * dim * kv_dim as usize..(l + 1) * dim * kv_dim as usize],
                   dim, kv_dim as usize
            );
            // V_c = X_b @ W_v
            matmul(&mut s.value_cache[v_start..v_start + kv_dim as usize],
                   &s.xb, &w.wv[l * dim * kv_dim as usize..(l + 1) * dim * kv_dim as usize],
                   dim, kv_dim as usize
            );

            // RoPE (relative positional encoding)
            // complex-valued rotate q and k in each head
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / 10000f32.powf(head_dim as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let (fcr, fci) = val.sin_cos();
                let rotn = if i < kv_dim as usize { 2 } else { 1 };  // 2 = q & k, 1 = q only
                for v in 0..rotn {
                    // the vector to rotate (query or key)
                    let vec = if v == 0 { &mut s.q } else {
                        &mut s.key_cache[k_start..k_start + kv_dim as usize] 
                    };
                    let v0 = vec[i];
                    let v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // Multi-head attention: iterate over all heads
            // Attention = softmax(Q.K).V
            (0..p.n_heads as usize).for_each(|h| {
                // Q for head
                let q = &s.q[h * head_size..(h + 1) * head_size];
                // Attention scores for head
                let att = &mut s.att[
                    h * p.seq_len as usize..(h + 1) * p.seq_len as usize];

                for (t, att_t) in att[..=pos].iter_mut().enumerate() {
                    // K for kv head
                    let k = &s.key_cache[
                        loff + t * kv_dim as usize + (h / kv_mul) * head_size..];
                    // Q.K
                    *att_t = q.iter()
                        .zip(k.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f32>() / (head_size as f32).sqrt();
                }
                
                // A = softmax(Q.K) 
                softmax(&mut att[0..=pos]);

                let xb = &mut s.xb[h * head_size..(h + 1) * head_size];
                xb.fill(0.0);
                // X_b = A.V
                for (t, &a) in att[..=pos].iter().enumerate() {
                    // V for kv head
                    let v = &s.value_cache[
                        loff + t * kv_dim as usize + (h / kv_mul) * head_size..];
                    for (i, xb_i) in xb.iter_mut().enumerate().take(head_size) {
                        *xb_i += a * v[i];
                    }
                }
            });

            // Final matmul to get the output of the attention
            // X_b2 = X_b @ W_o
            matmul(&mut s.xb2, &s.xb, 
                   &w.wo[l * dim * dim..(l + 1) * dim * dim],
                   dim, dim
            );

            // Residual connection back into x
            // X += X_b2
            for i in 0..dim {
                s.x[i] += s.xb2[i];
            }
            
            // Feed Forward Network
            // FFN rmsnorm
            // X_b = rmsnorm(X, W_ffn_rms)
            rmsnorm(&mut s.xb, &s.x, &w.rms_ffn_weight[l * dim..(l + 1) * dim]);

            // H_b = X_b @ W_1
            matmul(&mut s.hb, &s.xb, 
                   &w.w1[l * dim * hidden_dim as usize..(l + 1) * dim * hidden_dim as usize], 
                   dim, hidden_dim as usize
            );
            // H_b2 = X_b @ W_3
            matmul(&mut s.hb2, &s.xb, 
                   &w.w3[l * dim * hidden_dim as usize..(l + 1) * dim * hidden_dim as usize],
                   dim, hidden_dim as usize
            );

            // SwiGLU non-linearity
            // H_b = SwiGLU(H_b, H_b2)
            for i in 0..hidden_dim as usize {
                let val = s.hb[i];
                s.hb[i] = val / (1.0 + (-val).exp()) * s.hb2[i];
            }

            // Final matmul to get the output of the ffn
            // X_b = H_b @ W_2
            matmul(&mut s.xb, &s.hb, 
                   &w.w2[l * dim * hidden_dim as usize..(l + 1) * dim * hidden_dim as usize], 
                   hidden_dim as usize, dim
            );

            // Residual connection
            // X += X_b
            for i in 0..dim {
                s.x[i] += s.xb[i];
            }
        }

        // Final rmsnorm
        // X = rmsnorm(X, W_fw_rms)
        let mut final_rmsnorm: Vec<f32> = vec![0f32; s.x.len()];
        rmsnorm(&mut final_rmsnorm, &s.x, &w.rms_final_weight);
        s.x = final_rmsnorm;

        // Classifier into logits
        // logits = X @ W_cls
        matmul(&mut s.logits, &s.x, &w.wcls, p.dim as usize, p.vocab_size as usize);

        &mut s.logits
    }

    pub fn generate(&mut self, tokenizer: &Tokenizer, sampler: &mut Sampler, prompt: Option<&str>, steps: usize) {
        let prompt = prompt.unwrap_or("");
        let prompt_tokens = tokenizer.encode(prompt, true, false);
        if prompt_tokens.is_empty() {
            eprintln!("something is wrong, expected at least 1 prompt token");
            std::process::exit(1);
        }

        let mut start = None;
        let mut token = prompt_tokens[0];
        for pos in 0..steps {
            let logits = self.forward(token, pos);
            let next = if pos < prompt_tokens.len() - 1 {
                prompt_tokens[pos + 1]
            } else {
                sampler.sample(logits)
            };

            if next == 1 {
                break;
            }
            let piece = Tokenizer::decode(tokenizer, token, next);
            Tokenizer::safe_print(piece);
            io::stdout().flush().unwrap();
            token = next;

            if start.is_none() {
                start = Some(Instant::now());
            }
        }
        println!();

        if let Some(start) = start {
            let duration = start.elapsed();
            eprintln!("achieved tok/s: {:.2}", (steps as f64 - 1.0) / duration.as_secs_f64());
        }
    }
}
