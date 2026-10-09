# Local background removal

Taraference uses **BiRefNet Lite FP16** from the ONNX Community conversion of
[`ZhengPeng7/BiRefNet_lite`](https://github.com/ZhengPeng7/BiRefNet). The pinned
ONNX model is approximately 115 MB, marked MIT-licensed, and designed for
background removal and foreground masks. It is a practical general-purpose model
for this 4 GB RTX 3050 Ti laptop; output quality still varies with image content.

The ONNX Community model card shows the image-segmentation pipeline and the
repository identifies the model license as MIT:
<https://huggingface.co/onnx-community/BiRefNet_lite-ONNX>.

## Usage

Download the model once:

```powershell
tarafer image model download
```

Remove the background and save an RGBA PNG:

```powershell
tarafer image remove-bg .\photo.jpg
```

By default, the output is `photo-no-bg.png`. Choose another output path or device:

```powershell
tarafer image remove-bg .\photo.jpg --output .\result.png --device cuda
tarafer image remove-bg .\photo.jpg --device cpu
```

`--device auto` tries CUDA, then DirectML on Windows, then CPU. If a GPU provider
starts but runs out of memory during inference, auto retries on CPU. Use
`--device cuda`, `--device directml`, or `--device cpu` to select one explicitly;
explicit GPU selection reports provider errors instead of silently switching.
The ONNX Runtime provider sidecars are included in the Windows ZIP and Linux
tarball. CUDA still requires compatible NVIDIA runtime libraries on the machine.

Input formats are JPEG, PNG, WebP, BMP, and TIFF. Transparent output is PNG so
the foreground alpha matte is preserved. The current model runs at 1024×1024 and
the mask is resized to the source image dimensions. Inputs above 100 megapixels
are rejected to avoid excessive memory use.

Model weights are downloaded into `models/background-removal/` and are ignored by
Git. The download is pinned to a Hugging Face revision and checked against its
SHA-256 before installation. Do not package model weights in the source tree or
release archive.
