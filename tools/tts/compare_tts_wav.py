#!/usr/bin/env python3
"""Compare same-seed TTS WAVs using waveform and reconstructed log-mel correlation.

Independent NumPy implementation of the frontend's 24 kHz HTK-mel analysis.
Use identical text, reference audio, steps, guidance, speed and seed in both runs.
"""
import argparse
import wave
import numpy as np


def read(path):
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 24000):
            raise ValueError(f"{path}: expected mono 16-bit PCM at 24 kHz")
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    if len(x) < 1024 or np.sqrt(np.mean(x * x)) < 1e-5:
        raise ValueError(f"{path}: empty or silent speech output")
    return x


def log_mel(x):
    padded = np.pad(x, (512, 512), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 1024)[::256]
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(1024) / 1024)
    mag = np.abs(np.fft.rfft(frames * hann, axis=-1))
    points = 700.0 * (10.0 ** (np.linspace(0, 2595.0 * np.log10(1 + 12000.0 / 700), 102) / 2595.0) - 1)
    freqs = np.linspace(0, 12000.0, 513)
    down = (freqs[None] - points[:-2, None]) / (points[1:-1, None] - points[:-2, None])
    up = (points[2:, None] - freqs[None]) / (points[2:, None] - points[1:-1, None])
    filters = np.maximum(np.minimum(down, up), 0)
    return np.log(np.maximum(mag @ filters.T, 1e-5)).ravel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--min-mel-correlation", type=float, default=0.985)
    a = p.parse_args()
    reference, candidate = read(a.reference), read(a.candidate)
    if reference.shape != candidate.shape:
        p.error(f"sample count mismatch: {len(reference)} != {len(candidate)}")
    corr = np.corrcoef(log_mel(reference), log_mel(candidate))[0, 1]
    waveform = np.corrcoef(reference, candidate)[0, 1]
    print(f"{len(reference) / 24000:.6f} s, {len(reference)} samples, 24000 Hz")
    print(f"waveform correlation={waveform:.7f}, log-mel correlation={corr:.7f}")
    print(f"candidate RMS={np.sqrt(np.mean(candidate**2)):.7f}, peak={np.max(np.abs(candidate)):.7f}")
    if not np.isfinite(corr) or corr < a.min_mel_correlation:
        raise SystemExit("FAIL: log-mel correlation below threshold")
    print("PASS (numeric comparison; listening evaluation still required)")


if __name__ == "__main__":
    main()
