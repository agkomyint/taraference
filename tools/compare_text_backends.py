#!/usr/bin/env python3
"""Compare native wgpu and CUDA reports from `tarafer-wgpu bench`.

Checks identical workloads and full-vocabulary teacher-forced logits before
reporting median warm throughput. No comparison is made between different models.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cuda", type=Path)
    p.add_argument("wgpu", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    cuda, wgpu = [json.loads(f.read_text(encoding="utf-8")) for f in (a.cuda, a.wgpu)]
    if cuda["backend"] != "cuda" or wgpu["backend"] != "wgpu":
        p.error("pass the CUDA report first and wgpu report second")
    for field in ("model", "model_bytes", "context", "prompt_ids", "full_vocabulary", "build_profile"):
        if cuda[field] != wgpu[field]:
            p.error(f"different {field}; this is not a comparable workload")
    for field in ("teacher_ids", "positions"):
        left, right = cuda["verification"][field], wgpu["verification"][field]
        if field == "positions":
            left, right = [[v["context_tokens"] for v in x] for x in (left, right)]
        if left != right:
            p.error(f"different verification {field}")
    if {r["generated_tokens"] for r in cuda["runs"]} != {r["generated_tokens"] for r in wgpu["runs"]}:
        p.error("different generated token counts")
    shape = (len(cuda["verification"]["positions"]), cuda["full_vocabulary"])
    reference, candidate = [np.fromfile(r["verification"]["logits_file"], dtype="<f4").reshape(shape).astype(np.float64) for r in (cuda, wgpu)]
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        p.error("non-finite verification logits")
    parity = []
    for i, position in enumerate(cuda["verification"]["positions"]):
        ref, got = reference[i], candidate[i]
        cosine = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
        relative_rmse = float(np.linalg.norm(got-ref) / np.linalg.norm(ref))
        top1 = bool(np.argmax(ref) == np.argmax(got))
        overlap = len(set(np.argpartition(ref, -5)[-5:]) & set(np.argpartition(got, -5)[-5:]))
        parity.append({"context_tokens":position["context_tokens"], "cosine":cosine, "relative_rmse":relative_rmse, "max_abs":float(np.max(np.abs(got-ref))), "top1_match":top1, "top5_overlap":overlap})
    metrics = {}
    for name, report in (("cuda", cuda), ("wgpu", wgpu)):
        metrics[name] = {field:statistics.median(r[field] for r in report["runs"]) for field in ("decode_tps", "prefill_tps", "ttft_ms")}
        metrics[name]["load_ms"] = report["load_ms"]
    first_cuda, first_wgpu = cuda["runs"][0]["token_ids"], wgpu["runs"][0]["token_ids"]
    prefix = next((i for i,(x,y) in enumerate(zip(first_cuda, first_wgpu)) if x != y), min(len(first_cuda),len(first_wgpu)))
    passed = all(x["cosine"] >= 0.99 and x["relative_rmse"] <= 0.05 for x in parity)
    result = {"model":cuda["model"],"cuda_adapter":cuda["adapter"],"wgpu_adapter":wgpu["adapter"],"build_profile":cuda["build_profile"],"runs":{"cuda":len(cuda["runs"]),"wgpu":len(wgpu["runs"])},"median":metrics,"cuda_decode_speedup":metrics["cuda"]["decode_tps"]/metrics["wgpu"]["decode_tps"],"teacher_forced_logit_parity":parity,"parity_pass":passed,"first_run_matching_generated_prefix":prefix,"generated_tokens_per_run":len(first_cuda),"cuda_graph_active":cuda["cuda_graph_active"]}
    if a.out:
        a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("FAIL: teacher-forced logit parity outside tolerance")


if __name__ == "__main__":
    main()
