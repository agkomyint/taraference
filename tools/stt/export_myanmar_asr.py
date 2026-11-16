"""Export freococo/myanmar_asr (Burmese Conformer CTC, Apache-2.0) to ONNX for `tarafer speech transcribe`.

    pip install torch safetensors onnx onnxruntime huggingface_hub
    python tools/stt/export_myanmar_asr.py [--source DIR] [--out models/stt/myanmar-asr]

Writes `model.onnx` (log-mel `[1, frames, 80]` -> CTC logits `[1, frames/4, 2566]`, with the
upstream CMVN folded in) and `tokens.txt` (one syllable per line, line number = token id).
Only the encoder and CTC head are exported; the attention decoder is unused at inference.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

REPO = "freococo/myanmar_asr"
REV = "e40d48e354f6ee1b8e15e4fc513cfba7d624e80a"
# model.py is executed, so every input is pinned by content.
SHA256 = {
    "model.py": "d731b82039b8ba818f2806f68e56f5a9edb1cabd7efc0d876b21f4b84ecaa400",
    "model.safetensors": "d4d6b5b69c804db6043f3fc322c4e53f1996b274a55a0f7b260d15db812c541a",
    "vocab.json": "5154e188dec1e665f0d6411f003ba91bbdf62757f45c5188f39d018e058a4e8e",
    "cmvn.json": "cfa0f98140520bebee2580fc440c7a72cb9149a20b0003546bf037540388de0a",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="directory holding the upstream files (default: download)")
    parser.add_argument("--out", default="models/stt/myanmar-asr")
    args = parser.parse_args()

    if args.source:
        source = pathlib.Path(args.source)
    else:
        from huggingface_hub import snapshot_download

        source = pathlib.Path(
            snapshot_download(REPO, revision=REV, allow_patterns=sorted(SHA256))
        )
    for name, expected in SHA256.items():
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"{source / name}: sha256 {digest}, expected {expected} ({REPO}@{REV})")

    import torch

    spec = importlib.util.spec_from_file_location("myanmar_asr_model", source / "model.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    vocab = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    assert sorted(vocab.values()) == list(range(len(tokens))), "vocab ids are not contiguous"
    assert tokens[upstream.BLANK_ID] == "<blank>" and all("\n" not in t for t in tokens)
    cmvn = json.loads((source / "cmvn.json").read_text(encoding="utf-8"))
    model = upstream.load_model(str(source / "model.safetensors"), device="cpu")

    class Ctc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor(cmvn["mean"], dtype=torch.float32))
            self.register_buffer("std", torch.tensor(cmvn["std"], dtype=torch.float32) + 1e-9)

        def forward(self, mel):
            x = self.model.frontend((mel - self.mean) / self.std)
            for block in self.model.blocks:
                x = block(x)
            return self.model.ctc_fc(self.model.enc_norm(x))

    wrapper = Ctc().eval()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.randn(1, 200, 80),),
        str(out / "model.onnx"),
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {1: "frames"}, "logits": {1: "steps"}},
        opset_version=17,
        dynamo=False,
    )
    (out / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")

    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    torch.manual_seed(0)
    for frames in (57, 200, 1501):
        mel = torch.randn(1, frames, 80) * 2 - 4
        with torch.inference_mode():
            want = wrapper(mel).numpy()
            # The upstream entry point (with its all-false padding mask) must agree too.
            upstream_logits, _ = model((mel - wrapper.mean) / wrapper.std)
        got = session.run(None, {"mel": mel.numpy()})[0]
        error = abs(got - want).max()
        agree = (got.argmax(-1) == upstream_logits.numpy().argmax(-1)).mean()
        print(f"{frames} frames -> {got.shape[1]} steps: max |onnx - torch| = {error:.2e}, argmax agreement {agree:.4f}")
        assert error < 1e-3 and agree == 1.0, "ONNX export does not match PyTorch"
    print(f"wrote {out / 'model.onnx'} and {out / 'tokens.txt'} ({len(tokens)} tokens)")


if __name__ == "__main__":
    main()
