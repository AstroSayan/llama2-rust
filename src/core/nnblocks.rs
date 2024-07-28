use rayon::prelude::*;

pub fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    assert_eq!(o.len(), x.len(), "o and x must have the same length");
    assert_eq!(o.len(), weight.len(), "o and weight must have the same length");

    let size = x.len();

    // Calculate sum of squares
    let sum_squares: f32 = x.iter().map(|&val| val * val).sum();

    let ss = (sum_squares / size as f32) + 1e-5;
    let scale = 1.0 / ss.sqrt();

    // Normalize and scale
    o.iter_mut()
        .zip(x.iter())
        .zip(weight.iter())
        .for_each(|((o, &x), &w)| {
            *o = w * (scale * x);
        });
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Find max value (for numerical stability)
    let max_val = x.iter().fold(x[0], |max, &val| max.max(val));

    // Exp and sum
    let sum: f32 = x.iter_mut()
        .map(|val| {
            *val = (*val - max_val).exp();
            *val
        })
        .sum();

    // Normalize
    x.iter_mut().for_each(|val| *val /= sum);
}

// Matrix multiplication using parallel execution
pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    assert_eq!(xout.len(), d, "xout must have length d");
    assert_eq!(x.len(), n, "x must have length n");
    assert_eq!(w.len(), d * n, "w must have length d * n");
    xout.par_iter_mut()
        .enumerate()
        .for_each(|(i, val)| {
            *val = w[i * n..(i + 1) * n]
                .iter()
                .zip(x.iter())
                .map(|(&w, &x)| w * x)
                .sum();
        });
}
