# Taraference v0.7.0 roadmap

## Scope

**The only new v0.7.0 feature is local image background removal.** The feature
uses a dedicated image-segmentation model and runtime. It does not change GGUF
chat or embedding behavior.

The selected starting model is BiRefNet Lite FP16 ONNX, pinned to a specific Hugging
Face revision and SHA-256. Its model page marks the checkpoint MIT-licensed and
lists a 115 MB FP16 ONNX file. See [Background removal](../../docs/BACKGROUND_REMOVAL.md)
for the model source, usage, and current processing limits.

The workspace version stays `0.6.0` until the v0.7.0 release checks pass and the
version is deliberately bumped.

## Milestones

1. **Model and runtime:** keep model download separate from GGUF downloads, verify
   the pinned model checksum, and use available ONNX Runtime GPU providers with
   CPU fallback.
2. **CLI:** support `tarafer image model download` and
   `tarafer image remove-bg <input>` with a transparent PNG output and clear
   device selection.
3. **Image pipeline:** support common photo formats, retain source resolution,
   preserve a soft alpha matte, and reject unreasonable image dimensions safely.
4. **Packaging:** include ONNX Runtime provider sidecars in platform archives and
   carry them through install and self-update flows. Keep weights out of releases.
5. **Release candidate:** review portrait, hair/fur, object-edge, and complex
   background examples; document limitations; then build Windows and Linux release
   packages and bump to `0.7.0`.

## Acceptance checks

- Release builds work on Windows and Linux.
- A fresh model download passes the pinned SHA-256 check; corrupt files fail with
  a recovery instruction.
- GPU inference works when the required provider and runtime libraries are
  available; CPU fallback produces the same output dimensions and a valid
  transparent PNG.
- Portrait, hair/fur, fine objects, and difficult edges receive visual review.
- CLI help explains model setup, input/output formats, and the `auto`, `cuda`,
  `directml`, and `cpu` device choices.
- Release archives include runtime provider files but exclude ONNX and GGUF model
  weights.

## Out of scope

- Image generation, multimodal chat, video segmentation, and additional image
  models.
- Changing the official Qwen2.5 3B speed scoreboard or making cross-model token
  rate comparisons.
- Claiming pixel-perfect cutouts for every image type.
