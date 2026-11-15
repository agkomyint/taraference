# Taraference: the full picture

This document describes what taraference is meant to become. It is a direction, not a promise;
the dated plan is [ROADMAP.md](ROADMAP.md). Every model named here is a **candidate**: its licence,
size and speed must be verified on the upstream model card before it enters the registry.

**How to read the licence column.** `web` = reported by a 2026 web search (secondary sources, not
yet checked on the model card). `mem` = from the author's background knowledge, unverified.
`ours` = already in the registry and checked. Nothing here is legal advice.

## One sentence

Taraference is a lean local runtime for **small, specialised models**. One CLI, one registry,
one server, and every model runs on code we own, fast, on a 4 GB laptop GPU.

## Vocabulary

- **Specialised model:** a small model that does one job well (remove a background, transcribe
  speech, read text from an image) instead of a general assistant.
- **Task:** that job, exposed as a noun in the CLI (`text`, `speech`, `image`, `video`, ...).
- **Artifact:** what goes in and what comes out of a task. A text prompt, a `.wav`, a `.png`, an
  `.mp4`, a JSON result. Every task is defined as *input artifact -> output artifact*.
- **Model package:** weights pulled from the upstream publisher (pinned revision, SHA-256,
  licence), plus the engine in taraference that runs it. Weights never ship in our repo.
- **Engine:** our Rust + CUDA code that runs a model family (GGUF decoder, DiT, VITS, vision
  backbone). One shared op library underneath.

## Scope rules

A model belongs in taraference only if:

1. It is **small** (fits 4 GB with room to spare, ideally under about 2 GB).
2. It is **useful on its own**, with a clear input and output.
3. We can make it **fast** on modest hardware.
4. Its **licence** is shown before download; non-commercial weights need explicit acceptance.

Out of scope: agents, tool calling, retrieval pipelines, memory, training, fine-tuning, and large
language models.

---

# Catalog of candidates

## Text

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Generate | prompt -> text | Qwen3.5 0.8B / 2B / 4B | Apache-2.0 (ours) | shipped |
| Generate (baseline) | prompt -> text | Qwen2.5-3B | Qwen Research, non-commercial (ours) | shipped |
| Embed | text -> vector | Qwen3 embedding 4B | Apache-2.0 (ours) | shipped |
| Embed (small) | text -> vector | BGE-small, MiniLM, e5-small, nomic-embed | mem: MIT/Apache | idea |
| Rerank | query + passages -> scores | BGE reranker small, Qwen3 reranker | mem: Apache/MIT | idea |
| Translate | text -> text | Marian / opus-mt pairs, NLLB-200 distilled, MADLAD | mem: CC-BY 4.0 / CC-BY-NC | idea |
| Language ID | text -> language code | fastText lid, small encoder | mem: CC-BY-SA / MIT | idea |
| Classify / tag | text -> labels | DistilBERT / small BERT heads | mem: Apache-2.0 | idea |
| Sentiment / toxicity | text -> score | small encoder heads | mem: varies | idea |
| Named entities | text -> JSON | small NER encoders | mem: varies | idea |
| PII redaction | text -> masked text | NER + patterns | mem: varies | idea |
| Grammar / spelling fix | text -> text | small seq2seq (T5-small class) | mem: Apache-2.0 | idea |
| Summarise / rewrite | text -> text | covered by Generate | - | covered |
| Code completion | code -> code | small code LLMs (0.5-1.5B) | mem: varies | research |
| Sentence segmentation, tokenisation | text -> tokens | Burmese word/syllable segmenter | n/a (rules) | idea |
| Script conversion | text -> text | Zawgyi <-> Unicode converter | n/a (rules) | idea |

Why the last two matter: they are tiny, need no GPU, and fix real pain for Burmese text. They
also feed TTS and translation.

## Image

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Remove background | image -> PNG alpha | BiRefNet Lite | MIT (ours) | shipped |
| Remove background (alt) | image -> PNG alpha | U2-Net / IS-Net / RMBG | mem: Apache / non-commercial | idea |
| OCR (detect + recognise) | image -> text + boxes | PaddleOCR v5 | web: not confirmed | idea |
| OCR (end to end) | image -> text | Florence-2 (base ~0.23B) | web: MIT | idea |
| Handwriting | image -> text | TrOCR small | mem: MIT | idea |
| Document layout / tables | image -> regions | DocLayout-YOLO class, table detectors | mem: AGPL risk | research |
| Depth | image -> depth map | Depth Anything V2 **Small** | web: Apache-2.0 (Base/Large/Giant are CC-BY-NC) | idea |
| Segment anything | image + point/box -> mask | MobileSAM, EdgeSAM, SAM 3 | mem: Apache (MobileSAM); web: SAM 3 custom Meta terms | idea |
| Object detect | image -> boxes + labels | RF-DETR Nano/Small/Medium | web: Apache-2.0 | idea |
| Object detect (avoid) | - | YOLO26 | web: AGPL-3.0 | avoid |
| Open-vocabulary detect | image + text -> boxes | Grounding DINO | web: Apache-2.0 (size to check) | research |
| Upscale | image -> larger image | Real-ESRGAN small, Real-CUGAN | mem: BSD-3 / MIT | idea |
| Face detect | image -> boxes + landmarks | SCRFD, YuNet | mem: MIT / Apache | idea |
| Face restore | image -> image | GFPGAN, CodeFormer | mem: Apache / NC | research |
| Inpaint / object erase | image + mask -> image | LaMa | mem: Apache-2.0 | idea |
| Colourise | image -> image | DeOldify-class, ddcolor | mem: varies | research |
| Embed / search | image -> vector | CLIP / SigLIP small | mem: Apache / MIT | idea |
| Classify | image -> labels | MobileNet / EfficientNet | mem: Apache | idea |
| Caption | image -> text | Florence-2, SmolVLM, Moondream | web: MIT (Florence-2); others mem | research |
| Barcode / QR | image -> string | classical decoder (no ML) | n/a | idea |
| Image quality / blur score | image -> score | small CNN or classical | n/a | idea |

Why OCR is a priority: Burmese and other under-served scripts are a real need, it is
text-from-image (an easy artifact to explain), and detection + recognition are CNNs and
small transformers that share ops with BiRefNet.

## Audio and speech

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| TTS Burmese (quality) | text -> wav | F5-Myanmar-TTS v2 | CC BY-NC 4.0 (ours) | shipped |
| TTS Burmese (fast) | text -> wav | MMS-TTS Burmese (VITS) | CC BY-NC 4.0 (ours) | shipped |
| TTS Burmese (commercial) | text -> wav | MyanmarTTS | CC0 | planned |
| TTS English etc. | text -> wav | Kokoro 82M | web: Apache-2.0, runs on CPU, no cloning | idea |
| TTS (tiny) | text -> wav | Piper (VITS voices) | web: repo is GPL-3.0, voices vary; check | idea |
| TTS (cloning) | text + ref -> wav | CosyVoice2, Chatterbox | web: Apache-2.0 / MIT | research |
| TTS (non-commercial, avoid) | - | XTTS v2, Fish Audio S2 Pro, Voxtral TTS | web: non-commercial | avoid |
| Speech to text | wav -> text + timestamps | Whisper tiny / base / small | mem: MIT | planned 0.10.0 |
| Speech to text (Burmese) | wav -> text | freococo/myanmar_asr | reported Apache-2.0, verify | planned |
| Speech to text (edge) | wav -> text | Moonshine tiny/base, Vosk | web: on-device; licence not confirmed | idea |
| Voice activity | wav -> segments | Silero VAD | mem: MIT | idea |
| Denoise | wav -> wav | DeepFilterNet, RNNoise | mem: MIT/Apache, BSD | idea |
| Speaker diarisation | wav -> who spoke when | pyannote-class, ECAPA embeddings | mem: gated / MIT | research |
| Speaker verification | 2 wav -> same / not | ECAPA-TDNN small | mem: Apache | idea |
| Language ID | wav -> language | Whisper LID, MMS LID | mem: MIT / CC-BY-NC | idea |
| Keyword spotting | wav -> keyword | tiny CNN/transformer | mem: varies | idea |
| Sound event / tagging | wav -> labels | PANNs, BEATs-class | mem: varies | research |
| Music separation | wav -> stems | Demucs (small), Open-Unmix | mem: MIT | research |
| Beat / tempo | wav -> BPM | classical DSP | n/a | idea |
| Voice conversion | wav -> wav | RVC-class, kNN-VC | mem: MIT | research |
| Codec | wav <-> tokens | EnCodec / DAC small | mem: MIT / CC-BY-NC | research |
| Subtitles | wav/video -> .srt | Speech to text + VAD | - | idea |

## Video

Video starts as a **composition of image and audio tasks applied per frame or per segment**,
using the same engines. A dedicated video model is added only if it is small and clearly useful.

| Task | Input -> output | Approach / candidates | Licence | Status |
|---|---|---|---|---|
| Remove background | video -> alpha video | BiRefNet per frame + temporal smoothing | MIT (ours) | idea |
| Video matting (dedicated) | video -> alpha video | Robust Video Matting | mem: GPL-3.0, avoid for us | research |
| Subtitles | video -> .srt | extract audio, VAD, speech to text | - | idea |
| Upscale | video -> larger video | Real-ESRGAN / Real-CUGAN per frame; SeedVR2, FlashVSR (temporal) | web: tools reported, licences not confirmed | research |
| Frame interpolation | video -> higher fps | RIFE, FILM | web: recommended in 2026 workflows; licence to check | research |
| Upscale framework (avoid) | - | Video2X wraps these models | web: AGPL-3.0 | avoid |
| Scene cuts | video -> timestamps | classical + embeddings | n/a | idea |
| Thumbnails / key frames | video -> images | embeddings + diversity | n/a | idea |
| Caption / describe | video -> text | small VLM on sampled frames | mem: varies | research |
| Tracking | video -> boxes over time | detector + simple tracker | n/a | research |
| Denoise / stabilise | video -> video | classical + small CNN | n/a | research |
| Redact faces | video -> video | face detect + blur | mem: MIT/Apache | idea |

Video also needs decode/encode (ffmpeg or a Rust library), batching and streaming I/O. That is
an engineering item separate from the models.

## Multimodal (small)

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Describe image | image + prompt -> text | SmolVLM, Moondream, Florence-2 | web: SmolVLM small on-device; others mem | research |
| Document Q&A | page image + question -> text | SmolDocling / small VLMs | mem: varies | research |
| Image-text match | image + text -> score | CLIP / SigLIP | mem: Apache / MIT | idea |

## Documents and structured data

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| PDF to text | PDF -> text | text extraction + OCR fallback | n/a | idea |
| Layout to Markdown | page image -> Markdown | small doc models (SmolDocling class) | mem: varies | research |
| Table extraction | page image -> CSV/JSON | table detector + OCR | mem: varies | research |
| Form / receipt / ID parse | image -> JSON fields | small VLM or OCR + rules | mem: varies | research |
| Invoice / receipt totals | image -> JSON | OCR + rules | n/a | idea |
| Tabular prediction | CSV -> labels / numbers | gradient-boosted trees, small MLP | mem: Apache/MIT | idea |
| Anomaly detection | CSV / series -> flags | isolation forest, small autoencoder | n/a | idea |

## Time series and sensors

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Forecast | series -> future values | Chronos-Bolt small, TimesFM, small TFT | mem: Apache | research |
| Anomaly | series -> flags | small autoencoders | n/a | idea |
| Activity recognition | accelerometer -> label | tiny CNN/GRU | n/a | idea |
| Signal denoise | series -> series | classical filters + tiny net | n/a | idea |

## Geospatial and remote sensing

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Building / road extraction | satellite tile -> mask | small segmentation nets | mem: varies | research |
| Land-cover classification | tile -> labels | small CNN / ViT | mem: varies | research |
| Geocoding / address parse | text -> structured | small NER + rules | n/a | idea |

## 3D and spatial

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Monocular depth to point cloud | image -> 3D points | Depth Anything V2 Small + camera model | web: Apache-2.0 (Small) | idea |
| Normal estimation | image -> normals | small encoder-decoder | mem: varies | research |
| Pose estimation | image -> skeleton | RTMPose, MoveNet class | mem: Apache | idea |
| Hand / face landmarks | image -> points | MediaPipe-class nets | mem: Apache | idea |
| Gaze / head pose | image -> angles | small CNN | mem: varies | research |

## Cross-modal (small)

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Image to text search | text -> images | CLIP / SigLIP + local index | mem: Apache / MIT | idea |
| Audio to text search | text -> clips | CLAP-class | mem: varies | research |
| Speech to speech (translate) | wav -> wav | Speech to text + Translate + TTS chain | - | idea |
| Lip-sync / talking head | wav + image -> video | heavy; likely out of scope | - | out of scope |

## Utility pipelines (compose existing tasks)

| Pipeline | Chain |
|---|---|
| Subtitle a video | extract audio -> VAD -> speech to text -> `.srt` |
| Dub a video | speech to text -> translate -> TTS -> mix |
| Read a document aloud | OCR -> text clean-up -> TTS |
| Photo cut-out set | remove background over a folder, write PNGs |
| Meeting notes | VAD -> diarise -> speech to text -> Generate summary |
| Searchable photo library | image embeddings -> local index -> text query |
| Burmese text clean-up | Zawgyi->Unicode -> segment -> normalise -> TTS/translate |

## Modalities at a glance

| Modality | Artifact in | Artifact out | Today | Next |
|---|---|---|---|---|
| Text | prompt, document | text, vector, labels | generate, embed | translate, rerank, classify |
| Image | PNG/JPG | image, mask, text, boxes | remove background | OCR, depth, upscale |
| Audio / speech | WAV | WAV, text, labels | Burmese TTS | speech to text, VAD, denoise |
| Video | MP4 | video, text, timestamps | none | subtitles, per-frame cut-out |
| Documents | PDF, scan | Markdown, JSON | none | OCR, layout |
| Time series / tabular | CSV | numbers, flags | none | forecast, anomaly |
| 3D / pose | image | points, skeleton | none | depth, pose |
| Geospatial | tile | mask, labels | none | research only |

New modalities are added **only if** a model passes the scope rules above and we can write a
parity test for it. Order of expansion follows the ranking below, not this table.

## Bioinformatics and life sciences

Small sequence and structure models are a good fit for "local and specialised": the data is
sensitive or large, and many useful models are small encoders. All entries are `mem` and unverified.

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Protein embedding | amino-acid sequence -> vector | ESM-2 (8M-150M) | mem: MIT | research |
| Protein structure | sequence -> 3D coordinates | ESMFold, OpenFold, Boltz-1 | mem: MIT / Apache; AlphaFold 3 weights restricted | research |
| Protein function / property | sequence -> labels | ESM-2 heads, ProtTrans small | mem: varies (MIT / AFL) | research |
| DNA / genome embedding | DNA sequence -> vector | DNABERT-2, HyenaDNA small | mem: Apache-2.0 | research |
| Variant effect | sequence + variant -> score | small DNA/protein LMs | mem: varies | research |
| Single-cell annotation | expression matrix -> cell types | scGPT, Geneformer small | mem: MIT / Apache | research |
| Molecule property | SMILES -> properties | ChemBERTa, Chemprop (small) | mem: MIT | idea |
| Molecule embedding / search | SMILES -> vector | MolFormer-class | mem: Apache | idea |
| Retrosynthesis hints | molecule -> routes | small seq2seq models | mem: varies | research |
| Docking / binding | protein + ligand -> score | small scoring nets | mem: varies | research |
| Sequence alignment helpers | FASTA -> alignment | classical (no ML) | n/a | idea |
| Biomedical text | abstract -> entities, relations | PubMedBERT, BioBERT, SciSpaCy | mem: MIT / Apache | idea |
| Species / image ID | photo -> species | small vision classifiers | mem: varies | idea |
| Microscopy segmentation | image -> cell masks | Cellpose, StarDist | mem: BSD / BSD | research |

Why it fits: encoders such as ESM-2 are the same transformer family as our text engine; the
tokeniser is a 20-letter alphabet. Structure prediction is much heavier and probably out of
scope on 4 GB.

## Health and medical

Health models carry real risk. Taraference's role would be **running a model locally** so data
never leaves the device. It would **not** make clinical claims.

| Task | Input -> output | Candidates | Licence | Status |
|---|---|---|---|---|
| Clinical dictation | speech -> text | Whisper small + medical vocabulary | mem: MIT | idea |
| Clinical text NER | note -> entities | ClinicalBERT, SciSpaCy | mem: MIT / Apache | idea |
| De-identification | note -> masked note | NER + rules | mem: varies | idea |
| ICD / code suggestion | note -> codes | small classifiers | mem: dataset licences | research |
| Medical image segmentation | CT/MRI slice -> organ masks | nnU-Net, MedSAM, TotalSegmentator | mem: Apache-2.0 | research |
| Chest X-ray triage | X-ray -> findings | small CNN classifiers | mem: varies; datasets restricted | research |
| Skin / wound photo | photo -> category | small classifiers | mem: varies | research |
| Retinal scan | fundus image -> findings | small CNNs | mem: varies | research |
| ECG / EEG analysis | signal -> rhythm or state | small 1D CNN / transformer | mem: varies (PhysioNet) | research |
| Wearable vitals | sensor stream -> anomaly | tiny autoencoders | n/a | idea |
| Pill / label reader | photo -> drug name | OCR + lookup | n/a | idea |
| Symptom Q&A | question -> info | Generate with a small LLM | not recommended | out of scope |

Hard rules for this category:

- **No clinical or diagnostic claims.** Output is "model output", never medical advice.
- **Data stays local;** no telemetry, no upload.
- **Licences and dataset terms matter more here:** many medical weights are research-only or
  trained on restricted data; check both model and dataset.
- **Regulation varies by country;** shipping a diagnostic tool is not just a code problem.
- **Validation needs experts,** not a PyTorch parity test; parity only proves we match the
  reference, not that the model is correct for patients.

## Emerging and futuristic modalities (speculative watch-list)

None of these are planned. They are things small models are starting to do, or are likely to do,
that fit the "small, specialised, local" idea. Each enters the catalog only when a real, small,
openly licensed model exists and passes the scope rules. All entries are `mem`/speculation.

| Modality | Artifact in -> out | Why it might fit taraference | Blocker today |
|---|---|---|---|
| Tactile / haptics | touch-sensor array -> material or slip label | tiny nets, edge devices | few open models, needs hardware |
| Smell / chemical sensing | gas-sensor vector -> class | tiny classifiers | no standard data or models |
| EEG / biosignals | EEG, EMG, ECG -> state or label | small time-series nets, privacy-sensitive so local matters | medical-grade claims, licences, data |
| Brain-computer / neural decoding | neural signals -> text or command | research only | not open or small |
| Radar / LiDAR / event cameras | point cloud or events -> objects | small detectors for robots and drones | needs 3D ops, sparse conv |
| Wi-Fi / RF sensing | channel state -> presence or gesture | tiny local models | research quality |
| Egocentric / smart-glasses vision | camera stream -> scene text, objects | on-device by necessity | streaming engine, latency |
| Gesture and sign language | video -> text | accessibility value, small pose + sequence model | datasets, Burmese sign language data |
| Whole-body motion capture | video -> skeleton over time | animation, fitness | temporal models |
| Real-time streaming speech | mic -> text/speech with low latency | the natural next form of STT/TTS | streaming decoders, VAD |
| Speech emotion / paralinguistics | wav -> emotion, health cues | small encoders | accuracy, ethics |
| Text-to-music / sound effects | text -> audio | small generators may arrive | quality and size today |
| Text-to-3D / image-to-3D | image -> mesh | small feed-forward generators are appearing | memory, licences |
| Neural codecs for storage | image/audio -> compact bitstream | local compression | standards, speed |
| Personal / federated adaptation | user data -> small adapter | LoRA-style on-device tuning | we said out of scope (training) |
| Retrieval over personal files | files -> local index | embeddings already ours | scope rule: no pipelines |
| World / robotics policies | camera + goal -> action | small policies for robots | hardware, safety |
| Scientific small models | molecule, protein, weather, material -> property | compact predictors | domain data, licences |
| Accessibility transformers | image or screen -> description for blind users | small VLM + OCR + TTS | quality guarantees |
| Low-resource language suite | text/speech in Burmese, Shan, Karen, Mon -> all tasks | taraference's home turf | open datasets and models |

## Horizon: what models might look like around 2100 (pure speculation)

This is imagination, not research. No model below exists, none is planned, and none has a
licence, size or source. It is here to show which **artifacts** might matter if "small, local,
specialised" is still the right idea in 75 years. Every item would still have to pass the same
scope rules.

| Imagined task | Input -> output | What would make it a taraference-style model |
|---|---|---|
| Real-time universal interpreter | live speech in any language -> live speech in another, voice kept | tiny, always-on, private, on an earpiece |
| Personal health monitor | wearable and implant signals -> early warnings | runs on the device, never uploaded |
| Environment reader | camera, sound, air sensors -> a plain-language summary of a place | one small model per sense, composed locally |
| Language revival | a few recordings of a dying language -> speech, text and translation | built from tiny data, offline, for communities |
| Material and molecule designer | desired property -> candidate structure | small property predictors, run on a lab laptop |
| Climate and field forecaster | local sensors -> hours-ahead weather, flood, crop outlook | community scale, no cloud |
| Robot motor policies | body sensors + goal -> motor commands | small, verifiable, safety-bounded |
| Memory reconstructor | partial recordings -> a restored clip or photo | local restoration of family archives |
| Education tutor | learner answers -> adaptive explanation | small, curriculum-specific, offline |
| Neural interface decoder | neural signals -> intent | open, auditable, user-owned |
| Habitat 3D scanner | phone video -> editable 3D scene | on-device reconstruction |
| Verifier / provenance checker | media -> "edited, generated, or original" evidence | small detectors, open and auditable |

What would still be true, whatever the models are:

- the weights are small enough for a modest personal device,
- the user can read the licence and verify the files,
- the engine is code we own and can audit,
- nothing leaves the machine unless the user says so.

## How we decide when something is "next"

A futuristic modality becomes a real task only when all of these hold:

1. An **open, small** model exists (under about 2 GB, permissive or clearly stated licence).
2. It has a **clear input and output artifact** we can test.
3. Its ops are **already in `taraference-ops`** or a small addition.
4. There is a **fixture** we can compare to a PyTorch reference.
5. It is useful **without** an agent, a cloud, or a pipeline we have to invent.

Until then it stays on this list, and nothing is promised.

## The long-term shape

- **Short term (0.8-0.10):** text, image, speech, with owned engines.
- **Medium term (0.11-0.13):** documents, OCR, video built from image + audio, time series.
- **Long term (1.0+):** sensors, 3D, streaming, and low-resource language suites, as small models
  appear and the op library covers them.
- **Constant:** local, fast on 4 GB, licence shown before download, pinned and verified weights.

---

# Prioritisation

Ranked by **value to users x how much of the work is already shared**:

| Rank | Task | Why | Ops it needs beyond what we have |
|---|---|---|---|
| 1 | Speech to text (Whisper small) | Pairs with TTS, subtitles, strong demand | conv1d, log-mel, encoder-decoder attention |
| 2 | OCR (PaddleOCR / Florence-2) | Burmese text-from-image, easy to demo | CNN backbone, CTC decoder |
| 3 | Small embedder / reranker | Cheap, same engine as text | none (reuse GGUF/encoder ops) |
| 4 | Voice activity + denoise | Tiny, makes STT and subtitles better | small conv/GRU |
| 5 | Depth Anything V2 Small | Strong visual demo, ViT | ViT attention (shared with BiRefNet) |
| 6 | Kokoro TTS (English) | Apache-2.0, 82M, fast | StyleTTS2-class ops |
| 7 | Upscale (Real-ESRGAN small) | Image and video | conv, pixel shuffle |
| 8 | Translate (opus-mt) | Burmese <-> English | encoder-decoder |
| 9 | Inpaint (LaMa) | Object erase | FFT convs |
| 10 | Detect / segment | Generic vision | detection heads, mask decoder |

**Cheap first:** anything that reuses the text engine (embed, rerank, small encoders).
**Hard:** deformable conv (BiRefNet), diffusion and large autoregressive models (out of scope).

# Licence watch-list

- **Non-commercial:** F5-Myanmar-TTS, MMS-TTS, Qwen2.5-3B, Depth Anything Base/Large/Giant, XTTS v2,
  Fish Audio S2 Pro, Voxtral TTS (web). Shown with a prompt; never the default for commercial users.
- **Copyleft / source-sharing:** Piper repo (GPL-3.0), YOLO26 and Video2X (AGPL-3.0), Robust Video
  Matting (GPL-3.0, mem). We run these only if the user supplies them, and prefer alternatives.
- **Permissive and preferred:** Apache-2.0 and MIT models (Qwen3.5, BiRefNet, Kokoro, RF-DETR,
  Depth Anything V2 Small, Florence-2, Whisper).
- **Custom terms:** SAM 3, DINOv3 (Meta), Llama-based models (attribution). Read each licence.

---

# Surfaces

- **CLI:** `tarafer <task> <verb>`, pipe-friendly, `--json` for machines.
- **Server:** one OpenAI-compatible `tarafer serve`, models loaded lazily and kept warm.
  Endpoints: chat, completions, embeddings now; `/v1/audio/speech`, `/v1/audio/transcriptions`
  and an image endpoint planned.
- **Registry:** `tarafer models list | info | pull | verify | rm` for every task type.
  Users can eventually add their own entries (a local `registry.toml`).

# Engine strategy

1. **GGUF engine (done):** text and embeddings on custom CUDA.
2. **Shared op library (0.9.0):** GEMM, norms, attention, convolutions, activations, STFT, resize.
3. **Native model ports:** F5 (0.8.0), then MMS and BiRefNet (0.9.0), then every new model on the
   same ops.
4. **ONNX Runtime:** opt-in fallback for a model with no native port yet, so new models can ship
   before they are optimised.
5. **Every model gets the same report:** load time, run time, throughput or real-time factor, and
   the exact model name. Cross-task numbers are never compared.

# How a new model joins

1. Intake: size, licence, usefulness, which ops it needs.
2. Registry entry: pinned revision and SHA-256, licence, commercial flag.
3. Forward pass in Rust against the shared ops (or ORT fallback first).
4. Parity test against the PyTorch reference on fixed inputs.
5. Benchmark with `tarafer bench`.
6. Docs: `docs/MODELS.md` and one section for the task.

# What "done" looks like

A user installs one small binary, runs `tarafer models pull <id>`, and gets a fast, local,
specialised tool for text, images, audio or video: no Python, no cloud, no model-hosting by us,
and a clear licence prompt before anything is downloaded.

# Honest limits

- Each model port costs real engineering; "add any model" is not the promise.
- The reference machine is a 4 GB laptop GPU; larger models are not a goal.
- Quality claims need listening tests or labelled test sets; speed claims need exact model names.
- This catalog is research, not commitment. Sizes and licences come from secondary sources or
  memory and must be confirmed on each model card before any registry entry is written.
- Web research in this pass was one search per modality; OCR, video matting and several licences
  were only partly confirmed.

# Sources used for this pass

- TTS: [pinggy open-source TTS list](https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/),
  [Piper vs Kokoro](https://www.promptquorum.com/power-local-llm/piper-vs-kokoro-tts),
  [codesota TTS guide](https://codesota.com/guides/tts-models)
- Speech to text: [on-device transcription guide](https://www.siliconflow.com/articles/best-open-source-ai-for-on-device-transcription)
- Vision: [Roboflow open-source CV models](https://blog.roboflow.com/best-open-source-computer-vision-models/),
  [Depth Anything + SmolVLM](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/depth-anything-smolvlm-small-on-device-cv)
- Video: [Video2X](https://science.ecosyste.ms/projects/38260),
  [video upscale workflow notes](https://unpkg.com/comfyui-mcp@0.49.0/plugin/skills/video-upscale/SKILL.md)
