#!/usr/bin/env python3
"""Generate independent fixed-input parity fixtures from the existing ONNX exports.

Writes local raw little-endian f32/i64 files, never committed model weights.
Run the Rust `wgpu_parity` example against the output directory.
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models/tts/f5-myanmar"))
    p.add_argument("--out", type=Path, default=Path("target/f5-wgpu-fixtures"))
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=4, help="Euler steps for final-mel parity")
    a = p.parse_args()
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"] if a.provider == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    rng = np.random.default_rng(123)
    n = a.frames
    if n < 2:
        p.error("frames must be >= 2")
    x = rng.normal(size=(1, n, 100)).astype("<f4")
    cond = rng.normal(size=x.shape).astype("<f4")
    ids = rng.integers(1, 2000, size=(1, n), dtype=np.int64)
    ids[:, n // 2:] = 0
    a.out.mkdir(parents=True, exist_ok=True)
    feeds = {"x": x, "cond": cond, "text": ids, "time": np.array(0.3, dtype="<f4"), "cfg": np.array(2.0, dtype="<f4")}
    sess = ort.InferenceSession(str(a.model_dir / "dit.onnx"), providers=providers)
    velocity = sess.run(None, feeds)[0]
    for name, data in [("x", x), ("cond", cond), ("ids", ids.astype("<i8")), ("velocity", velocity)]:
        data.tofile(a.out / f"{name}.bin")
    if not 1 <= a.steps <= 128:
        p.error("steps must be in 1..=128")
    # Same sway schedule and Euler updates as the Rust frontend, independent ORT network.
    t = np.linspace(0.0, 1.0, a.steps + 1, dtype=np.float32)
    ts = t + np.float32(-1.0) * (np.cos(np.pi / 2.0 * t) - 1.0 + t)
    sample = x.copy()
    for i in range(a.steps):
        feeds["x"], feeds["time"] = sample, np.array(ts[i], dtype="<f4")
        v = sess.run(None, feeds)[0]
        sample = sample + (ts[i + 1] - ts[i]) * v
    sample.astype("<f4").tofile(a.out / "sample.bin")
    ts.astype("<f4").tofile(a.out / "timesteps.bin")
    del sess
    mel = rng.normal(size=(1, 100, n)).astype("<f4")
    sess = ort.InferenceSession(str(a.model_dir / "vocos.onnx"), providers=providers)
    re, im = sess.run(None, {"mel": mel})
    for name, data in [("mel", mel), ("re", re), ("im", im)]:
        data.tofile(a.out / f"{name}.bin")
    print(f"ONNX fixtures: {a.out}, {n} frames, provider {a.provider}")


if __name__ == "__main__":
    main()
