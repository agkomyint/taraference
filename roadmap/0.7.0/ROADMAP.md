# Taraference v0.7.0 roadmap

## Goal

Keep Taraference's local CUDA text inference reliable while defining a small,
reviewable path into image processing. The first image feature to investigate is
background removal. Image generation is a separate, larger pipeline and is not a
v0.7.0 release requirement.

This is a proposed release plan. The workspace version remains `0.6.0` until the
release checks below pass and the version is deliberately bumped.

## Priorities

### 1. Preserve and validate the inference baseline

- Keep `Qwen2.5-3B-Instruct-Q4_K_M.gguf` as the official scoreboard model and
  retain its historical v0.4/v0.5 comparisons and 750 tok/s north star.
- Use exact model names in every benchmark report. Do not compare token rates across
  different models, quantizations, GPUs, or metric definitions.
- Qwen3.5 and other supported models may be profiled and used for speed A/B work;
  those results stay model-specific. Do not use 0.8B or 0.5B results as top-line
  speed wins or official scoreboard numbers.
- Validate correctness alongside speed: record model, quantization, GPU and compute
  capability, backend, decode rate, context behavior, and any approximation.

### 2. Make model behavior clear

- Confirm supported Qwen chat models load and complete a short multi-turn session on
  the supported NVIDIA targets.
- Keep embedding models on an embedding-oriented path. Do not treat repeated or
  malformed chat output from an embedding checkpoint as a valid chat result.
- Make unsupported model purpose or architecture errors actionable at load time.

### 3. Scope a background-removal image prototype

- Choose and document a dedicated segmentation/background-removal model and its
  license, runtime, memory needs, and supported image sizes before integrating it.
- Define a simple local input/output interface and preserve the original image;
  write the foreground result with an alpha channel or mask output.
- Prototype image decoding, preprocessing, inference, and output encoding as a
  separate pipeline so text GGUF loading and chat behavior remain understandable.
- Measure image quality and latency on representative photographs. Do not report
  image throughput as language-model tokens per second.

### 4. Evaluate image generation separately

- Write an architecture and dependency proposal for a diffusion pipeline, including
  model format, GPU memory, sampler, and output handling.
- Keep image generation out of the v0.7.0 release scope unless the prototype can
  meet the same build, quality, and memory checks as the background-removal feature.

## Milestones

1. **Baseline:** record a clean v0.6.0 reference profile on the official Qwen2.5 3B
   model and confirm a short chat works.
2. **Model path review:** document which local Qwen models support chat versus
   embeddings, and fix any confirmed correctness issues in the supported path.
3. **Image prototype:** select a background-removal model and produce a local
   foreground PNG with transparency or a separate alpha mask.
4. **Integration decision:** review usability, dependencies, license, VRAM, and
   measured image quality; decide whether the prototype belongs in `tarafer` or a
   companion executable.
5. **Release candidate:** build the chosen packages on Windows and Linux, run chat
   and image smoke checks, update user documentation, then bump to `0.7.0`.

## Release checks

- The release build succeeds for every platform being packaged.
- The official scoreboard model passes the short correctness and multi-turn checks.
- Any included image feature produces a valid output file and passes visual review
  on portrait, hair/fur, and complex-edge examples.
- Model-specific performance reports include the exact model name and hardware.
- No model weights, build output, or large run artifacts are required in the source
  repository.
- Update the README and release notes to describe exactly which model tasks are
  supported and which require separate tooling.

## Out of scope for v0.7.0

- Claiming an official speed milestone from Qwen3.5 or a sub-billion model.
- General-purpose multimodal chat across arbitrary image models.
- A production image-generation service or multi-user batching.
- Bundling large model weights in the repository or release binary.
