#!/usr/bin/env python3
"""Generate fixed-input parity fixtures for the native wgpu BiRefNet Lite port.

Runs the upstream PyTorch reference (ZhengPeng7/BiRefNet_lite, pinned revision,
downloaded with `--download`) and writes raw little-endian f32 files plus a
`manifest.txt` of `name dims...` lines. Nothing here is committed: the weights
live under `models/` and the fixtures under `target/`.

    python tools/image/birefnet_wgpu_fixtures.py --download
    python tools/image/birefnet_wgpu_fixtures.py [--image photo.jpg] [--device cuda]
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = "ZhengPeng7/BiRefNet_lite"
REVISION = "aa62cd87eafb9cc43056d08ef3615a14628b831d"
SHA256 = "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815"
FILES = ["BiRefNet_config.py", "birefnet.py", "config.json", "model.safetensors"]


def download(model_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name in FILES:
        hf_hub_download(REPO, name, revision=REVISION, local_dir=str(model_dir))


def verify(model_dir: Path) -> None:
    import hashlib

    digest = hashlib.sha256()
    with open(model_dir / "model.safetensors", "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        sys.exit(f"model.safetensors does not match the pinned SHA-256 {SHA256}")


def stub_optional_imports() -> None:
    """The reference imports timm and kornia for training-only helpers."""
    import torch
    import transformers  # noqa: F401  (probes optional packages; import before stubbing)
    from importlib.machinery import ModuleSpec

    def stub(name):
        module = types.ModuleType(name)
        module.__spec__ = ModuleSpec(name, None)
        return module

    if importlib.util.find_spec("timm") is None:
        layers = stub("timm.layers")

        class DropPath(torch.nn.Identity):
            def __init__(self, drop_prob=0.0, *args, **kwargs):
                super().__init__()

        layers.DropPath = DropPath
        layers.to_2tuple = lambda v: tuple(v) if isinstance(v, (tuple, list)) else (v, v)
        layers.trunc_normal_ = lambda tensor, *args, **kwargs: tensor
        timm = stub("timm")
        timm.layers = layers
        sys.modules["timm"], sys.modules["timm.layers"] = timm, layers
    if importlib.util.find_spec("kornia") is None:
        filters = stub("kornia.filters")
        filters.laplacian = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("kornia.filters.laplacian is training-only")
        )
        kornia = stub("kornia")
        kornia.filters = filters
        sys.modules["kornia"], sys.modules["kornia.filters"] = kornia, filters


def load_reference(model_dir: Path):
    import torch
    from safetensors.torch import load_file

    stub_optional_imports()
    package = types.ModuleType("birefnet_pt")
    package.__path__ = [str(model_dir)]
    sys.modules["birefnet_pt"] = package
    for name in ["BiRefNet_config", "birefnet"]:
        spec = importlib.util.spec_from_file_location(f"birefnet_pt.{name}", model_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    module = sys.modules["birefnet_pt.birefnet"]
    model = module.BiRefNet(config=sys.modules["birefnet_pt.BiRefNet_config"].BiRefNetConfig(bb_pretrained=False))
    state = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval(), torch


def preprocess(path: Path, size: int) -> np.ndarray:
    """Same 8-bit triangle resize and ImageNet normalisation as the Rust frontend."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((pixels - mean) / std).transpose(2, 0, 1)[None].astype("<f4")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=Path("models/background-removal/birefnet-lite-pt"))
    p.add_argument("--out", type=Path, default=Path("target/birefnet-wgpu-fixtures"))
    p.add_argument("--image", type=Path, help="RGB image; a seeded synthetic image is used otherwise")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--download", action="store_true", help="fetch the pinned upstream files and exit")
    p.add_argument("--onnx", type=Path, help="also store this ONNX model's logits as `onnx_logits`")
    a = p.parse_args()
    if a.download:
        download(a.model_dir)
        verify(a.model_dir)
        print(f"BiRefNet Lite reference: {a.model_dir} ({REPO}@{REVISION[:7]}, MIT)")
        return
    if a.size % 64 or not 64 <= a.size <= 1024:
        p.error("size must be a multiple of 64 in 64..=1024")
    verify(a.model_dir)
    model, torch = load_reference(a.model_dir)
    model = model.to(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(name: str, tensor) -> None:
        data = tensor.detach().float().cpu().numpy().astype("<f4") if hasattr(tensor, "detach") else tensor
        data.tofile(a.out / f"{name}.bin")
        manifest.append(f"{name} {' '.join(str(d) for d in data.shape)}")

    if a.image:
        image = preprocess(a.image, a.size)
    else:
        # Smooth blobs plus noise: exercises the same value range as a normalised photo.
        rng = np.random.default_rng(123)
        yy, xx = np.mgrid[0 : a.size, 0 : a.size].astype(np.float32) / a.size
        blob = np.exp(-(((xx - 0.45) ** 2 + (yy - 0.55) ** 2) / 0.05))
        base = np.stack([blob * 3 - 1.5, np.sin(xx * 9) * np.cos(yy * 7), 1.2 - 2.4 * yy])
        image = (base + 0.15 * rng.normal(size=base.shape)).astype("<f4")[None]
    x = torch.from_numpy(image).to(a.device)
    save("input", x)

    with torch.inference_mode():
        # 1. Deformable convolution in isolation (the decoder's first 3x3 ASPP branch).
        #    The batch norm is included because the native port folds it into the weights.
        branch = model.decoder.decoder_block4.dec_att.aspp_deforms[1]
        rng = np.random.default_rng(7)
        deform_in = torch.from_numpy(rng.normal(size=(1, 64, 24, 24)).astype("<f4")).to(a.device)
        save("deform_in", deform_in)
        save("deform_out", branch.bn(branch.atrous_conv(deform_in)))

        # 2. Backbone stages at full and half resolution.
        stages = model.bb(x)
        for index, stage in enumerate(stages, 1):
            save(f"bb_x{index}", stage)

        # 3. Encoder (two scales + context), squeeze block, decoder input.
        (x1, x2, x3, x4), _ = model.forward_enc(x)
        save("enc_x1", x1)
        save("enc_x4", x4)
        x4 = model.squeeze_module(x4)
        save("squeeze", x4)

        # 4. Final logits.
        logits = model(x)[-1]
        save("logits", logits)

    if a.onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
        feed = image.astype(np.float16) if "float16" in session.get_inputs()[0].type else image
        onnx_logits = session.run(None, {session.get_inputs()[0].name: feed})[-1].astype("<f4")
        save("onnx_logits", onnx_logits)
        ref = logits.float().cpu().numpy()
        a_mask, b_mask = ref > 0, onnx_logits.reshape(ref.shape) > 0
        union = np.logical_or(a_mask, b_mask).sum()
        print(f"PyTorch vs ONNX mask IoU: {np.logical_and(a_mask, b_mask).sum() / max(union, 1):.5f}")

    (a.out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"BiRefNet Lite fixtures: {a.out}, {a.size}x{a.size}, device {a.device}")
    for line in manifest:
        print("  " + line)


if __name__ == "__main__":
    main()
