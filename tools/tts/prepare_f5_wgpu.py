#!/usr/bin/env python3
"""Prepare native F5 weights locally. No downloads or inference exports.

F5's original model.safetensors is copied unchanged. Vocos' upstream PyTorch
state dict is repacked losslessly as safetensors; Python is only used for setup.
Both checkpoints must already be obtained under their upstream licences.
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f5-dir", type=Path, required=True, help="Upstream F5 directory with model.safetensors, vocab.txt and ref_sample.wav")
    p.add_argument("--vocos", type=Path, required=True, help="Upstream charactr/vocos-mel-24khz pytorch_model.bin")
    p.add_argument("--out", type=Path, default=Path("models/tts/f5-myanmar"))
    a = p.parse_args()
    import torch
    from safetensors.torch import save_file
    for name in ("model.safetensors", "vocab.txt", "ref_sample.wav"):
        if not (a.f5_dir / name).is_file():
            p.error(f"missing {a.f5_dir / name}")
    state = torch.load(a.vocos, map_location="cpu", weights_only=True)
    state = {k: v.contiguous() for k, v in state.items() if k.startswith(("backbone.", "head.out."))}
    if tuple(state["head.out.weight"].shape) != (1026, 512):
        p.error("expected the 24 kHz Vocos checkpoint")
    a.out.mkdir(parents=True, exist_ok=True)
    save_file(state, a.out / "vocos.safetensors")
    for name in ("model.safetensors", "vocab.txt", "ref_sample.wav"):
        src, dst = a.f5_dir / name, a.out / name
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    print(f"Native F5-Myanmar-TTS v2 + Vocos weights ready: {a.out}")


if __name__ == "__main__":
    main()
