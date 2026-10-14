#!/usr/bin/env python3
"""Export freococo/F5-Myanmar-TTS (CC BY-NC 4.0, non-commercial) to the ONNX layout used by
`tarafer tts`.

Output directory (default models/tts/f5-myanmar):
  dit.onnx        DiT denoiser with classifier-free guidance folded in (fp16 weights, fp32 I/O)
  vocos.onnx      Vocos mel -> complex spectrogram (ISTFT runs in Rust)
  vocab.txt       character vocabulary
  ref_sample.wav  default 24 kHz reference voice

Requirements:  pip install f5-myanmar-tts onnx onnxruntime
Run:           python tools/tts/export_f5_myanmar.py [--out DIR] [--fp32]

fp16 export needs a CUDA GPU; use --fp32 to export on CPU (1.3 GB dit.onnx).
"""
import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


class DitStep(torch.nn.Module):
    """One CFG-guided velocity prediction.

    text: [1, N] int64 token ids (vocab index + 1, 0 = filler) already padded to N mel frames.
    """

    def __init__(self, dit, half):
        super().__init__()
        self.m = dit
        self.half = half

    def _text(self, ids, drop):
        te = self.m.text_embed
        n = ids.shape[1]
        mask = (ids == 0).unsqueeze(-1)
        if drop:
            ids = torch.zeros_like(ids)
        e = te.text_embed(ids) + te.freqs_cis[:n, :]
        e = e.masked_fill(mask, 0.0)
        for block in te.text_blocks:
            e = block(e).masked_fill(mask, 0.0)
        return e

    def forward(self, x, cond, text, time, cfg):
        m = self.m
        if self.half:
            x, cond, time, cfg = x.half(), cond.half(), time.half(), cfg.half()
        n = x.shape[1]
        t = m.time_embed(time.reshape(1))
        xc = m.input_embed(x, cond, self._text(text, False))
        xu = m.input_embed(x, torch.zeros_like(cond), self._text(text, True))
        h = torch.cat((xc, xu), 0)
        t2 = torch.cat((t, t), 0)
        rope = m.rotary_embed.forward_from_seq_len(n)
        residual = h if m.long_skip_connection is not None else None
        for block in m.transformer_blocks:
            h = block(h, t2, mask=None, rope=rope)
        if m.long_skip_connection is not None:
            h = m.long_skip_connection(torch.cat((h, residual), dim=-1))
        out = m.proj_out(m.norm_out(h, t2))
        cond_pred, null_pred = out.chunk(2, 0)
        return (cond_pred + (cond_pred - null_pred) * cfg).float()


class VocosSpec(torch.nn.Module):
    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.proj = vocos.head.out

    def forward(self, mel):
        x = self.proj(self.backbone(mel)).transpose(1, 2)
        mag, phase = x.chunk(2, 1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/tts/f5-myanmar")
    ap.add_argument("--fp32", action="store_true", help="export fp32 weights (CPU friendly, 2x larger)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    half = not args.fp32
    device = "cuda" if half else "cpu"
    if half and not torch.cuda.is_available():
        raise SystemExit("fp16 export needs CUDA; re-run with --fp32 to export on CPU")

    from f5_myanmar_tts import MyanmarTTS

    tts = MyanmarTTS(device=device)
    dit = tts.model.transformer.eval()
    dit = dit.half() if half else dit.float()
    step = DitStep(dit, half).eval()

    n, nt = 300, 120
    x = torch.randn(1, n, 100, device=device)
    cond = torch.randn(1, n, 100, device=device)
    text = torch.cat([torch.randint(1, 90, (1, nt)), torch.zeros(1, n - nt, dtype=torch.long)], 1).to(device)
    t = torch.tensor(0.3, device=device)
    cfg = torch.tensor(2.0, device=device)
    print("exporting dit.onnx ...")
    torch.onnx.export(
        step, (x, cond, text, t, cfg), str(out / "dit.onnx"),
        input_names=["x", "cond", "text", "time", "cfg"], output_names=["v"],
        dynamic_axes={"x": {1: "n"}, "cond": {1: "n"}, "text": {1: "n"}, "v": {1: "n"}},
        opset_version=17, dynamo=False,
    )

    print("exporting vocos.onnx ...")
    vocos = VocosSpec(tts.vocoder).eval().float().cpu()
    mel = torch.randn(1, 100, 250)
    torch.onnx.export(
        vocos, (mel,), str(out / "vocos.onnx"),
        input_names=["mel"], output_names=["re", "im"],
        dynamic_axes={"mel": {2: "n"}, "re": {2: "n"}, "im": {2: "n"}},
        opset_version=17, dynamo=False,
    )

    shutil.copyfile(tts.vocab_path, out / "vocab.txt")
    shutil.copyfile(tts.default_ref_audio, out / "ref_sample.wav")

    # Sanity check at a length different from the traced one.
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if half else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(out / "dit.onnx"), providers=providers)
    n2, nt2 = 640, 150
    x = torch.randn(1, n2, 100, device=device)
    cond = torch.randn(1, n2, 100, device=device)
    text = torch.cat([torch.randint(1, 90, (1, nt2)), torch.zeros(1, n2 - nt2, dtype=torch.long)], 1).to(device)
    with torch.no_grad():
        ref = step(x, cond, text, t, cfg).cpu().numpy()
    got = sess.run(None, {
        "x": x.cpu().numpy(), "cond": cond.cpu().numpy(), "text": text.cpu().numpy(),
        "time": t.cpu().numpy(), "cfg": cfg.cpu().numpy(),
    })[0]
    err = float(np.abs(ref - got).max())
    scale = float(np.abs(ref).max())
    print(f"dit.onnx check: max|err|={err:.4f} (scale {scale:.1f})")
    if err > 0.02 * scale + 1e-3:
        raise SystemExit("ONNX output does not match PyTorch; do not use this export")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
