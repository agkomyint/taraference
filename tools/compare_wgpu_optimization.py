#!/usr/bin/env python3
"""Compare paired wgpu optimization benchmarks and cached independent fp32 fixtures.

The reference-source report must be the exact original input used to produce the
PyTorch reference. Reports and binary logits stay under target/, outside commits.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("cuda", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (a.before, a.after, a.cuda, a.reference_source)]
    before, after, cuda, source = reports
    reference = json.loads(a.reference.read_text(encoding="utf-8"))
    if [r["backend"] for r in reports] != ["wgpu", "wgpu", "cuda", "wgpu"]:
        p.error("expected before/after wgpu, CUDA, and original wgpu reference-source reports")
    for field in ("model", "model_bytes", "context", "prompt_ids", "full_vocabulary", "build_profile"):
        if any(r[field] != after[field] for r in reports):
            p.error(f"mismatched {field}")
    for r in reports:
        if r["verification"]["teacher_ids"] != after["verification"]["teacher_ids"]:
            p.error("mismatched teacher-forced token IDs")
        if [v["context_tokens"] for v in r["verification"]["positions"]] != [v["context_tokens"] for v in after["verification"]["positions"]]:
            p.error("mismatched verification positions")
    if reference["model"] != after["model"] or [v["context_tokens"] for v in reference["positions"]] != [v["context_tokens"] for v in source["verification"]["positions"]]:
        p.error("cached reference does not match source report")
    if not reference["pass"]:
        p.error("the original independent reference check did not pass")
    counts = [{r["generated_tokens"] for r in report["runs"]} for report in (before, after, cuda)]
    if counts[0] != counts[1] or counts[1] != counts[2]:
        p.error("different output token counts")
    shape = (len(reference["positions"]), after["full_vocabulary"])
    ref = np.fromfile(a.reference.with_suffix(".logits.bin"), dtype="<f4").reshape(shape).astype(np.float64)
    actual = np.fromfile(after["verification"]["logits_file"], dtype="<f4").reshape(shape).astype(np.float64)
    if not (np.isfinite(ref).all() and np.isfinite(actual).all()):
        p.error("nonfinite reference or candidate")
    checks = []
    for i, (expected, got) in enumerate(zip(ref, actual)):
        checks.append({"context_tokens":reference["positions"][i]["context_tokens"],"relative_rmse":float(np.linalg.norm(expected-got)/np.linalg.norm(expected)),"cosine":float(np.dot(expected,got)/(np.linalg.norm(expected)*np.linalg.norm(got))),"top1_match":bool(np.argmax(expected)==np.argmax(got))})
    passed = all(c["relative_rmse"]<=0.005 and c["cosine"]>=0.9999 and c["top1_match"] for c in checks)
    metrics = {name:{field:statistics.median(r[field] for r in report["runs"]) for field in ("decode_tps","prefill_tps","ttft_ms")} for name,report in (("before",before),("after",after),("cuda",cuda))}
    old_tokens, new_tokens = before["runs"][0]["token_ids"], after["runs"][0]["token_ids"]
    prefix = next((i for i,(x,y) in enumerate(zip(old_tokens,new_tokens)) if x!=y),min(len(old_tokens),len(new_tokens)))
    result = {"model":after["model"],"adapter":after["adapter"],"build_profile":after["build_profile"],"prompt_tokens":after["prompt_tokens"],"output_tokens":after["runs"][0]["generated_tokens"],"runs":{name:len(report["runs"]) for name,report in (("before",before),("after",after),("cuda",cuda))},"median":metrics,"decode_speedup":metrics["after"]["decode_tps"]/metrics["before"]["decode_tps"],"prefill_speedup":metrics["after"]["prefill_tps"]/metrics["before"]["prefill_tps"],"cuda_decode_speedup":metrics["cuda"]["decode_tps"]/metrics["after"]["decode_tps"],"before_after_matching_prefix":prefix,"cached_independent_fp32_reference":str(a.reference),"reference_checks":checks,"reference_pass":passed}
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not passed:
        raise SystemExit("FAIL: optimized wgpu differs from independent fp32 reference")


if __name__ == "__main__":
    main()
