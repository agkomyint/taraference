#!/usr/bin/env python3
"""Independent fp32 PyTorch reference for the native Qwen2/Qwen2.5 wgpu path.

Developer validation only: the product does not depend on Python, PyTorch or gguf-py.
Weights are dequantized by gguf-py, one layer at a time; network math uses fp32
without TF32. K/V are rounded to fp16 after RoPE to match the runtime cache.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path, help="wgpu bench JSON including teacher-forced positions")
    p.add_argument("--model", type=Path, default=Path("models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"))
    p.add_argument("--gguf-lib", type=Path, help="optional local directory containing gguf-py")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.gguf_lib:
        sys.path.insert(0, str(a.gguf_lib))
    import gguf
    import torch
    import torch.nn.functional as F
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    report = json.loads(a.report.read_text(encoding="utf-8"))
    if report["backend"] != "wgpu":
        p.error("expected a wgpu report")
    reader = gguf.GGUFReader(a.model)
    tensors = {t.name: t for t in reader.tensors}
    meta = lambda key: reader.get_field("qwen2." + key).contents()
    width, heads, kvheads = [int(meta(k)) for k in ("embedding_length", "attention.head_count", "attention.head_count_kv")]
    hd = width // heads
    eps = float(meta("attention.layer_norm_rms_epsilon"))
    theta = float(meta("rope.freq_base"))
    blocks = int(meta("block_count"))
    ids = report["prompt_ids"] + report["verification"]["teacher_ids"]
    n = len(ids)

    def load(name):
        t = tensors[name]
        values = gguf.dequantize(t.data, t.tensor_type)
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(a.device)

    def linear(x, name):
        w = load(name + ".weight")
        b = load(name + ".bias") if name + ".bias" in tensors else None
        return F.linear(x, w, b)

    def rms(x, name):
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * load(name + ".weight")

    positions = torch.arange(n, device=a.device, dtype=torch.float32)
    freqs = theta ** (-torch.arange(0, hd, 2, device=a.device, dtype=torch.float32) / hd)
    angle = positions[:, None] * freqs[None, :]
    cos, sin = angle.cos()[None], angle.sin()[None]
    def rope(x, h):
        x = x.reshape(n, h, hd).transpose(0, 1)
        lo, hi = x.chunk(2, dim=-1)
        return torch.cat([lo*cos-hi*sin, lo*sin+hi*cos], dim=-1)

    embedding = tensors["token_embd.weight"]
    rows = gguf.dequantize(embedding.data[np.array(ids)], embedding.tensor_type)
    x = torch.from_numpy(np.array(rows, dtype=np.float32, copy=True)).to(a.device)
    causal_mask = torch.ones((n, n), device=a.device, dtype=torch.bool).triu(1)
    with torch.inference_mode():
        for i in range(blocks):
            prefix = f"blk.{i}"
            y = rms(x, prefix + ".attn_norm")
            q = rope(linear(y, prefix + ".attn_q"), heads)
            k = rope(linear(y, prefix + ".attn_k"), kvheads).half().float()
            v = linear(y, prefix + ".attn_v").reshape(n, kvheads, hd).transpose(0, 1).half().float()
            k, v = [z.repeat_interleave(heads // kvheads, dim=0) for z in (k, v)]
            score = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
            score.masked_fill_(causal_mask, -float("inf"))
            att = (score.softmax(dim=-1) @ v).transpose(0, 1).reshape(n, width)
            x = x + linear(att, prefix + ".attn_output")
            y = rms(x, prefix + ".ffn_norm")
            x = x + linear(F.silu(linear(y, prefix + ".ffn_gate")) * linear(y, prefix + ".ffn_up"), prefix + ".ffn_down")
            if (i + 1) % 4 == 0:
                print(f"reference: layer {i+1}/{blocks}", flush=True)
        selected = [v["context_tokens"] - 1 for v in report["verification"]["positions"]]
        x = rms(x[selected], "output_norm")
        output_name = "output" if "output.weight" in tensors else "token_embd"
        # Keep the 152k-token output matrix out of RAM/VRAM in expanded form.
        # Chunking also allows this developer reference to run on the 4 GiB GPU.
        output = tensors[output_name + ".weight"]
        chunks = []
        for start in range(0, output.data.shape[0], 4096):
            values = gguf.dequantize(output.data[start:start+4096], output.tensor_type)
            weight = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(a.device)
            chunks.append(F.linear(x, weight).cpu())
        reference = torch.cat(chunks, dim=-1).numpy().astype(np.float64)
    candidate = np.fromfile(report["verification"]["logits_file"], dtype="<f4").reshape(reference.shape).astype(np.float64)
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise SystemExit("FAIL: non-finite logits")
    results = []
    for i, (ref, got) in enumerate(zip(reference, candidate)):
        cosine = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
        relative_rmse = float(np.linalg.norm(got-ref) / np.linalg.norm(ref))
        results.append({"context_tokens":selected[i]+1, "cosine":cosine, "relative_rmse":relative_rmse, "max_abs":float(np.max(np.abs(got-ref))), "top1_match":bool(np.argmax(ref) == np.argmax(got))})
    passed = all(r["cosine"] >= 0.9999 and r["relative_rmse"] <= 0.005 and r["top1_match"] for r in results)
    result = {"model":report["model"], "reference":"gguf-py dequantization + PyTorch fp32, TF32 disabled, fp16-rounded KV", "torch_version":torch.__version__, "device":a.device, "positions":results, "pass":passed}
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    reference.astype("<f4").tofile(a.out.with_suffix(".logits.bin"))
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("FAIL: independent fp32 reference parity")


if __name__ == "__main__":
    main()
