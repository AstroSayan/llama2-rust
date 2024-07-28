use std::mem::size_of;

use chrono::Utc;

use crate::core::nnblocks::softmax;
use crate::core::types::{ProbIndex, Sampler};

fn random_u32(mut state: u64) -> u32 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    ((state.wrapping_mul(0x2545F4914F6CDD1D)) >> 32) as u32
}

fn random_f32(state: u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16_777_216.0
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, top_p: f32) -> Self {
        let prob_index = Vec::with_capacity(
            vocab_size * size_of::<ProbIndex>()
        );
        Sampler {
            vocab_size,
            temperature,
            top_p,
            prob_index
        }
    }

    fn sample_argmax(probabilities: &[f32]) -> usize {
        // return the index that has the highest probability
        probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1)
        probabilities
            .iter()
            .scan(0.0, |cdf, &p| {
                *cdf += p;
                Some(*cdf)
            })
            .position(|cdf| coin < cdf)
            .unwrap_or(probabilities.len() - 1)  // in case of rounding errors
    }

    fn sample_top_p(&mut self, probabilities: &[f32], top_p: f32, coin: f32) -> usize {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1)
        let n = probabilities.len();
        
        // values smaller than (1 - top_p) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        let cutoff = (1.0 - top_p) / (n - 1) as f32;
        
        self.prob_index = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= cutoff)
            .map(|(i, &p)| ProbIndex { prob: p, index: i })
            .collect();
        
        // sort indices in descending order of probabilities
        self.prob_index.sort_unstable_by(
            |a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal)
        );

        // truncate the list where cumulative probability exceeds top_p
        let mut cumulative_prob = 0.0;
        let last_idx = self.prob_index
            .iter()
            .take_while(|pi| {
                cumulative_prob += pi.prob;
                cumulative_prob <= top_p
            })
            .count();
        
        // sample from the truncated list
        let r = coin * cumulative_prob;

        let mut cdf = 0.0;
        for pi in &self.prob_index[..=last_idx] {
            cdf += pi.prob;
            if r < cdf {
                return pi.index;
            }
        }

        // In case of rounding errors
        self.prob_index[last_idx].index
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            Self::sample_argmax(logits)
        } else {
            for logit in logits.iter_mut() {
                *logit /= self.temperature;
            }
            softmax(logits);
            let coin: f32 = random_f32(Utc::now().timestamp() as u64);
            if self.top_p <= 0.0 || self.top_p >= 1.0 {
                Self::sample_mult(logits, coin)
            } else {
                self.sample_top_p(logits, self.top_p, coin)
            }
        }
    }
}
