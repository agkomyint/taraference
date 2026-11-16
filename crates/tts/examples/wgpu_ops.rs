//! Ablation benchmark of one F5 DiT block on Burn's wgpu backend (random weights, fp16).
//! Each line drops or swaps one part of the block; times are per block.
use burn::{
    backend::{
        wgpu::{self, WgpuDevice},
        Wgpu,
    },
    tensor::{backend::Backend, f16, Distribution, Int, Tensor, TensorData},
};
use std::time::Instant;

type B = Wgpu<f16, i32>;
type T = Tensor<B, 3>;
type T4 = Tensor<B, 4>;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Full,
    NoNorm,
    NoRope,
    NoAttention,
    NoMax,
    NoExp,
    NoFeedForward,
    NoGelu,
    NoQkv,
    MatmulOnly,
}

struct Block {
    n: usize,
    w: Vec<T>,
    wide: T,
    narrow: T,
    ada: T,
    cos: T,
    sin: T,
    swap: Tensor<B, 1, Int>,
    ones: T4,
}

fn norm(x: T) -> T {
    let centered = x.clone() - x.mean_dim(2);
    let centered = centered.clone() / (centered.abs().max_dim(2) + 1e-4);
    let variance = (centered.clone() * centered.clone()).mean_dim(2);
    centered / (variance + 1e-6).sqrt()
}

impl Block {
    fn rope(&self, x: T, mode: Mode) -> T4 {
        let n = self.n;
        let x = if mode == Mode::NoRope || mode == Mode::MatmulOnly {
            x
        } else {
            x.clone() * self.cos.clone() + x.select(2, self.swap.clone()) * self.sin.clone()
        };
        x.reshape([2, n, 16, 64]).swap_dims(1, 2)
    }

    fn run(&self, mut h: T, mode: Mode) -> T {
        let n = self.n;
        let plain = mode == Mode::MatmulOnly;
        let part = |j: usize| self.ada.clone().slice([0..1, 0..1, j * 1024..(j + 1) * 1024]);
        let y = if mode == Mode::NoNorm || plain { h.clone() } else { norm(h.clone()) * (part(1) + 1.0) + part(0) };
        let (q, k, v) = if mode == Mode::NoQkv {
            (y.clone(), y.clone(), y)
        } else {
            (y.clone().matmul(self.w[0].clone()), y.clone().matmul(self.w[1].clone()), y.matmul(self.w[2].clone()))
        };
        let att = if mode == Mode::NoAttention {
            v
        } else {
            let q = self.rope(q, mode);
            let k = self.rope(k, mode).swap_dims(2, 3);
            let v = Tensor::cat(vec![v.reshape([2, n, 16, 64]).swap_dims(1, 2), self.ones.clone()], 3);
            let scores = q.matmul(k);
            let weights = match mode {
                Mode::NoMax => (scores - 4.0).exp(),
                Mode::NoExp | Mode::MatmulOnly => scores,
                _ => (scores.clone() - scores.max_dim(3)).exp(),
            };
            let mixed = weights.matmul(v);
            let out = if plain {
                mixed.slice([0..2, 0..16, 0..n, 0..64])
            } else {
                mixed.clone().slice([0..2, 0..16, 0..n, 0..64]) / mixed.slice([0..2, 0..16, 0..n, 64..65])
            };
            out.swap_dims(1, 2).reshape([2, n, 1024])
        };
        h = if plain { h + att.matmul(self.w[3].clone()) } else { h + part(2) * att.matmul(self.w[3].clone()) };
        if mode == Mode::NoFeedForward {
            return h;
        }
        let y = if mode == Mode::NoNorm || plain { h.clone() } else { norm(h.clone()) * (part(4) + 1.0) + part(3) };
        let y = y.matmul(self.wide.clone());
        let y = if mode == Mode::NoGelu || plain {
            y
        } else {
            let cube = y.clone() * y.clone() * y.clone();
            y.clone() * 0.5 * (((y + cube * 0.044715) * 0.797_884_6).tanh() + 1.0)
        };
        if plain { h + y.matmul(self.narrow.clone()) } else { h + part(5) * y.matmul(self.narrow.clone()) }
    }
}

fn main() {
    let device = WgpuDevice::default();
    let options = wgpu::RuntimeOptions { memory_config: wgpu::MemoryConfiguration::ExclusivePages, ..Default::default() };
    let setup = wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, options);
    println!("{:?}", setup.adapter.get_info());
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(1125);
    let r = |shape: [usize; 3], std: f64| T::random(shape, Distribution::Normal(0.0, std), &device);
    let block = Block {
        n,
        w: (0..4).map(|_| r([1, 1024, 1024], 0.02)).collect(),
        wide: r([1, 1024, 2048], 0.02),
        narrow: r([1, 2048, 1024], 0.02),
        ada: r([1, 1, 6144], 0.1),
        cos: r([1, n, 1024], 0.5),
        sin: r([1, n, 1024], 0.5),
        swap: Tensor::from_data(TensorData::new((0..1024).map(|c| c ^ 1).collect::<Vec<i32>>(), [1024]), &device),
        ones: T4::ones([2, 16, n, 1], &device),
    };
    let h = r([2, n, 1024], 1.0);
    let time = |label: &str, run: &mut dyn FnMut() -> T| {
        let _ = run().slice([0..1, 0..1, 0..1]).into_data();
        let started = Instant::now();
        let mut last = None;
        for _ in 0..20 {
            last = Some(run());
        }
        let _ = last.unwrap().slice([0..1, 0..1, 0..1]).into_data();
        B::sync(&device);
        println!("{label:<44} {:7.2} ms", started.elapsed().as_secs_f64() * 50.0);
    };
    let scores = T4::random([2, 16, n, n], Distribution::Normal(0.0, 1.0), &device);
    let flat = |x: T4| { let [a, b, c, d] = x.dims(); x.reshape([a * b, c, d]) };
    time("row max: max_dim(3)", &mut || flat(scores.clone().max_dim(3)));
    if n % 25 == 0 {
        time("row max: two stages of 25", &mut || flat(scores.clone().reshape([2, 16, n, n / 25, 25]).max_dim(4).max_dim(3).reshape([2, 16, n, 1])));
    }
    time("row max: max_dim(2) (columns)", &mut || flat(scores.clone().max_dim(2)));
    time("row sum: sum_dim(3)", &mut || flat(scores.clone().sum_dim(3)));
    time("exp(scores - 4)", &mut || flat((scores.clone() - 4.0).exp()));
    let x = r([2, n, 1024], 1.0);
    let pad = T::zeros([2, 15, 1024], &device);
    let wcat = T4::random([1, 16, 31 * 64, 64], Distribution::Normal(0.0, 0.02), &device);
    let wtap: Vec<T4> = (0..31).map(|_| T4::random([1, 16, 64, 64], Distribution::Normal(0.0, 0.02), &device)).collect();
    time("pos conv: stacked taps, one matmul", &mut || {
        let x = T::cat(vec![pad.clone(), x.clone(), pad.clone()], 1).reshape([2, n + 30, 16, 64]).swap_dims(1, 2);
        let taps = (0..31).map(|t| x.clone().slice([0..2, 0..16, t..t + n, 0..64])).collect();
        Tensor::cat(taps, 3).matmul(wcat.clone()).swap_dims(1, 2).reshape([2, n, 1024])
    });
    time("pos conv: 31 matmuls summed", &mut || {
        let x = T::cat(vec![pad.clone(), x.clone(), pad.clone()], 1).reshape([2, n + 30, 16, 64]).swap_dims(1, 2);
        let mut sum = x.clone().slice([0..2, 0..16, 0..n, 0..64]).matmul(wtap[0].clone());
        for t in 1..31 {
            sum = sum + x.clone().slice([0..2, 0..16, t..t + n, 0..64]).matmul(wtap[t].clone());
        }
        sum.swap_dims(1, 2).reshape([2, n, 1024])
    });
    let wfull = r([1, 1024, 1024], 0.02);
    time("pos conv: 31 dense 1024 matmuls (upper bound)", &mut || {
        let x = T::cat(vec![pad.clone(), x.clone(), pad.clone()], 1);
        let mut sum = x.clone().slice([0..2, 0..n, 0..1024]).matmul(wfull.clone());
        for t in 1..31 {
            sum = sum + x.clone().slice([0..2, t..t + n, 0..1024]).matmul(wfull.clone());
        }
        sum
    });
    time("layer norm (max-scaled)", &mut || norm(x.clone()));
    let mean_w = T::ones([1, 1024, 1], &device) / 1024.0;
    time("layer norm (means by matmul)", &mut || {
        let c = x.clone() - x.clone().matmul(mean_w.clone());
        let c = c.clone() / (c.abs().max_dim(2) + 1e-4);
        c.clone() / ((c.clone() * c).matmul(mean_w.clone()) + 1e-6).sqrt()
    });
    time("readback [2,n,100]", &mut || { let y = x.clone().slice([0..2, 0..n, 0..100]) * 2.0; let _ = y.clone().into_data(); y });
    let modes = [
        ("full block", Mode::Full),
        ("without layer norms", Mode::NoNorm),
        ("without rotary", Mode::NoRope),
        ("without q/k/v linears", Mode::NoQkv),
        ("without attention core", Mode::NoAttention),
        ("attention without row max", Mode::NoMax),
        ("attention without max and exp", Mode::NoExp),
        ("without feed-forward", Mode::NoFeedForward),
        ("without gelu", Mode::NoGelu),
        ("matmuls only", Mode::MatmulOnly),
    ];
    for (label, mode) in modes {
        let _ = block.run(h.clone(), mode).slice([0..1, 0..1, 0..1]).into_data();
        let started = Instant::now();
        let mut x = h.clone();
        for _ in 0..22 {
            x = block.run(x, mode) * 0.5;
        }
        let _ = x.slice([0..1, 0..1, 0..1]).into_data();
        B::sync(&device);
        println!("{label:<32} {:7.2} ms/block", started.elapsed().as_secs_f64() * 1000.0 / 22.0);
    }
}
