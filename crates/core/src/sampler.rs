//! Host-side quality sampling (temperature / top-k / top-p / repetition penalty).
//!
//! Greedy benchmarks stay on the GPU argmax path and never call into this module.

use std::collections::HashSet;

/// Sampling knobs. `temperature <= 0` means greedy (caller should skip this path).
#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 256,
            repetition_penalty: 1.0,
        }
    }
}

/// xorshift64* style PRNG step; returns U[0, 1).
pub fn next_random(state: &mut u64) -> f64 {
    let mut x = (*state).max(1);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f64) / (u64::MAX as f64 + 1.0)
}

/// Sample a token id from `(id, logit)` candidates.
pub fn sample_logits(
    mut logits: Vec<(u32, f32)>,
    history: &[u32],
    opts: SamplingOptions,
    rng: &mut u64,
) -> u32 {
    let penalty = opts.repetition_penalty.max(1.0);
    if penalty > 1.0 {
        let seen: HashSet<u32> = history.iter().copied().collect();
        for (id, logit) in &mut logits {
            if seen.contains(id) {
                *logit = if *logit < 0.0 {
                    *logit * penalty
                } else {
                    *logit / penalty
                };
            }
        }
    }
    let temperature = opts.temperature.max(1e-5);
    let candidate_count = opts.top_k.max(1).min(logits.len());
    if candidate_count < logits.len() {
        logits.select_nth_unstable_by(candidate_count, |a, b| b.1.total_cmp(&a.1));
        logits.truncate(candidate_count);
    }
    logits.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let max_logit = logits.first().map_or(0.0, |x| x.1);
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|(_, x)| (((*x - max_logit) / temperature) as f64).exp())
        .collect();
    let total: f64 = probs.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    for p in &mut probs {
        *p /= total;
    }
    let top_p = opts.top_p.clamp(1e-6, 1.0) as f64;
    let mut keep = probs.len();
    let mut cumulative = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumulative += *p;
        if cumulative >= top_p {
            keep = i + 1;
            break;
        }
    }
    let kept_total: f64 = probs[..keep].iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let target = next_random(rng) * kept_total;
    let mut running = 0.0;
    for (i, p) in probs[..keep].iter().enumerate() {
        running += *p;
        if running >= target {
            return logits[i].0;
        }
    }
    logits[keep.saturating_sub(1)].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_can_change_the_choice() {
        let mut seed = 1;
        let id = sample_logits(
            vec![(1, 10.0), (2, 9.9)],
            &[1],
            SamplingOptions {
                temperature: 0.01,
                top_p: 1.0,
                top_k: 2,
                repetition_penalty: 1.2,
            },
            &mut seed,
        );
        assert_eq!(id, 2);
    }

    #[test]
    fn top_k_one_always_picks_max() {
        let mut seed = 42;
        let id = sample_logits(
            vec![(7, 3.0), (8, 9.0), (9, 1.0)],
            &[],
            SamplingOptions {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 1,
                repetition_penalty: 1.0,
            },
            &mut seed,
        );
        assert_eq!(id, 8);
    }

    #[test]
    fn top_p_can_exclude_low_mass_tail() {
        // Strong peak on id 1; with tiny top_p only the peak remains.
        let mut seed = 99;
        let id = sample_logits(
            vec![(1, 20.0), (2, 0.0), (3, 0.0)],
            &[],
            SamplingOptions {
                temperature: 0.1,
                top_p: 0.1,
                top_k: 10,
                repetition_penalty: 1.0,
            },
            &mut seed,
        );
        assert_eq!(id, 1);
    }

    #[test]
    fn next_random_is_deterministic() {
        let mut a = 12345u64;
        let mut b = 12345u64;
        assert_eq!(next_random(&mut a), next_random(&mut b));
        assert_ne!(next_random(&mut a), next_random(&mut a)); // advances
    }
}
