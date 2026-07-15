# Upscaler

English | [Русский](README.md)

A professional application for upscaling and enhancing images, combining AI
models, classic algorithms, and a full processing pipeline.

## From the author

I built this project for myself. I wanted a single tool that combines modern
AI upscalers, denoising, deblurring, face restoration, colorization, and
"Photoshop-class" adjustments — with automatic parameter selection, so I
wouldn't have to fiddle with a dozen sliders by hand. That's why the feature
set is so broad, sometimes even excessive: these are exactly the things I
needed when working with real photos. After playing with it for a while, I
decided to share it with everyone — maybe it will be useful to you too.

The project is deliberately designed to run **even on modest hardware** (a
mini-PC, or a laptop without a discrete GPU): almost everything works on the
CPU, and the heavy AI models are optional (see
[Running on a mini-PC](#running-on-a-mini-pc)).

## Features

- **11 upscaling methods** — 5 AI (Real-ESRGAN, HAT-S, SwinIR, OmniSR, DAT) + 6 classic (Lanczos, Bicubic, Sinc, NEDI, DCCI, EGGI/SIRE)
- **6 denoisers** — 2 AI (SCUNet, NAFNet) + 4 classic (BM3D, NL-Means, Bilateral, Wavelet)
- **7 adjusters** — Auto Contrast, Auto Tone, Auto Color, Brightness, Contrast, Saturation, Sharpness
- **3 colorizers** — DDColor (photo), DeOldify (photo and video), ColorMNet (photo and video)
- **SmartDeblur** — restoration of out-of-focus and motion-blurred images via deconvolution (Wiener / Tikhonov / Total Variation) with automatic blind parameter estimation
- **ICEdit** — edit an image with a natural-language instruction (FLUX.1-Fill-dev GGUF + LoRA); just describe the change in text
- **Face restoration (CodeFormer)** — automatic enhancement of faces in portraits
- **"AI assistant" checkbox** — turns off the vision model and switches the auto buttons to algorithmic parameter selection
- **Scales** — 2x, 4x, 8x, 16x, and an "enhance only" mode
- **Auto-downscale of soft images** — if the nominal resolution is high but real detail is lower (an upsample), auto mode first downscales the image to its effective resolution (estimated from the spectrum), then processes it normally. Toggled by a checkbox in the "Scale" group.
- **Variant blending** — an automatic feature: intermediate processing variants are composited on top of each other (25 Photoshop blend modes, mode and opacity chosen by a quality metric); enabled with the "Variant blending" checkbox.
- **Before/after canvas** — split view with a draggable divider, zoom, and pan
- **7 built-in presets** — Photo Realistic, Anime/Art, Text/Document, Maximum Quality, Fast Preview, Enhance Only, Archival Restore
- **Batch processing** — process whole folders with per-file error handling
- **Version history** — on-disk with thumbnails, revert, comparison, and session restore
- **Quality metrics** — BRISQUE, NIQE, histogram, artifact detection
- **GPU acceleration** — CUDA with automatic CPU fallback, tiled inference for large images

## Screenshots

| | |
|---|---|
| ![Main window, single mode](docs/screenshots/01-main-single.png) **Main window** — before/after canvas, left control panel, version history on the right | ![Correction sections](docs/screenshots/02-correction-sections.png) **Correction** — collapsible sections: tone, color, detail |
| ![SmartDeblur](docs/screenshots/03-smartdeblur.png) **SmartDeblur** — manual deconvolution with preview | ![ICEdit](docs/screenshots/04-icedit.png) **ICEdit** — editing by text instruction |
| ![Variant blending](docs/screenshots/05-blend.png) **Blending** — variant list + 25 blend modes | ![Face zones](docs/screenshots/06-face-zones.png) **Faces** — manual zone marking for CodeFormer |
| ![Before](docs/screenshots/200x200.png) **blurred image** | ![After](docs/screenshots/800x800.jpg) **Restored image** |
| ![Version history](docs/screenshots/07-history.png) **History** — versions with thumbnails, revert and compare | |

## Running on a mini-PC

The application was designed from the start to be **useful even without a
powerful GPU** — on a mini-PC, a thin laptop, or an office machine.

- **The entire core runs on the CPU.** The classic upscalers (Lanczos,
  Bicubic, Sinc, NEDI, DCCI, EGGI/SIRE), all denoisers except the neural ones,
  the adjusters (Photoshop-class correction), SmartDeblur (FFT deconvolution),
  and all analysis/metrics are built on numpy/OpenCV/scipy and need no GPU.
- **AI models are optional.** Neural upscalers, colorizers, face restoration,
  ICEdit, and the AI assistant are all opt-in. If you don't install their
  weights/dependencies, the corresponding steps are simply skipped and the app
  keeps working.
- **Modest memory use by default.** Processing runs in overlapping tiles, so
  large images don't require holding everything in memory at once. The tile
  size is configurable (`tile_size`) — lower it on a weak machine.
- **Resilience under low resources.** When VRAM runs short, the tile size is
  reduced automatically, then it transparently falls back to the CPU — no crash
  and no error shown to the user.
- **A single executable.** You can build a standalone `.exe` (PyInstaller; the
  CPU PyTorch variant produces a compact build) — the end user doesn't need to
  install anything. See [Building](#building-a-standalone-application-exe).

Practical scenario: on a mini-PC without a discrete GPU, use the classic
upscalers + denoising + adjustments + SmartDeblur — that already gives a
noticeable improvement, entirely on the CPU. Add the neural models as resources
allow.

## Supported formats

| Input | Output |
|-------|--------|
| PNG, JPEG, TIFF, BMP, WebP | PNG (8-bit) |
| OpenEXR, Radiance HDR | TIFF (16-bit) |
| RAW (CR2, NEF, ARW) | OpenEXR (32-bit float) |

## Installation

### Automatic (recommended)

The scripts create a `.venv` environment on Python 3.10 and install everything:
PyTorch, the dependencies from `requirements.txt`, and `llama-cpp-python` (for
the LLM advisor). If Python 3.10 is missing, they install it (Windows: winget;
Linux/macOS: pyenv / system package).

With the CUDA flag the scripts install **all GPU-capable libraries** in their
CUDA variant: PyTorch, `llama-cpp-python` (LLM/vision advisor),
`onnxruntime-gpu` (CodeFormer), and the ICEdit stack (diffusers, etc.), then
print a GPU-availability check. The default CUDA tag is `cu124` (valid for both
the PyTorch index and llama-cpp).

**Windows (PowerShell):**
```powershell
.\install.ps1                          # CPU builds
.\install.ps1 -Cuda                    # all GPU libraries (cu124)
.\install.ps1 -Cuda -CudaVersion cu121 # a different CUDA tag
```

**Linux/macOS:**
```bash
./install.sh                       # CPU
./install.sh --cuda                # all GPU libraries (cu124)
CUDA_VERSION=cu121 ./install.sh --cuda
```

> Important: a plain `pip install llama-cpp-python` installs the **CPU build**,
> so `-Cuda` force-replaces it with the CUDA wheel (`--force-reinstall
> --prefer-binary`). If the LLM assistant runs on the CPU after installation,
> make sure you launch the app from `.venv` and check the output of the
> "GPU availability check" step.

### Manual

```bash
pip install -r requirements.txt
# optional, for the LLM advisor:
pip install "llama-cpp-python>=0.3"
```

### Requirements

- Python 3.10+
- PySide6 >= 6.6
- PyTorch >= 2.5 (with CUDA for GPU acceleration)
- OpenCV, NumPy, Pillow, scipy, scikit-image, spandrel, safetensors

Optional (for individual plugins):
- `bm3d >= 4.0` — BM3D denoiser
- `PyWavelets >= 1.5` — Wavelet denoiser
- `rawpy >= 0.19` — camera RAW support
- `OpenEXR >= 3.2` — EXR format support
- `onnxruntime >= 1.16` — face restoration (CodeFormer-ONNX); without it the step is skipped. For GPU: `onnxruntime-gpu`.

## Maximizing GPU usage

The application uses the GPU wherever possible, but **different parts use
different runtimes**, and each one must be installed in its CUDA variant. A
plain package install pulls the CPU builds of `llama-cpp-python` and
`onnxruntime` — in which case those steps silently run on the CPU. Below is how
to enable the GPU to the fullest.

### What is accelerated, and by what

| Component | GPU runtime | What to install |
|-----------|-------------|-----------------|
| AI upscalers, denoisers, colorizers | **PyTorch + CUDA** | CUDA build of `torch` |
| SmartDeblur (FFT deconvolution Wiener/Tikhonov/TV/RL) | **PyTorch + CUDA** (`torch.fft`) | CUDA build of `torch` |
| AI assistant (LLM + vision/CLIP) | **llama-cpp-python with CUDA** | CUDA wheel of `llama-cpp-python` |
| Face restoration (CodeFormer-ONNX) | **onnxruntime-gpu** | `onnxruntime-gpu` |
| ICEdit (FLUX.1-Fill-dev) | **PyTorch + CUDA** (diffusers) | CUDA `torch` + diffusers stack |
| Face detection (YuNet, OpenCV) | — (CPU) | a light step, no GPU needed |

### Step 1. The only setting — `gpu_device`

In `~/.upscaler/settings.json` (or the "Settings" menu), the `gpu_device`
parameter:

- `"auto"` (default) — use CUDA if available, otherwise CPU. **This is the
  "maximum GPU" mode.**
- `"cuda:0"` — force a specific GPU.
- `"cpu"` — force CPU (the AI assistant is not offloaded to the GPU either).

No other switches are needed: with `"auto"`/`"cuda:*"` all torch models are
moved to the GPU (FP16 autocast, tiled inference, automatic tile-size
selection), and the AI assistant offloads all model layers to the GPU
(`n_gpu_layers=-1`).

### Step 2. Install the CUDA builds of the runtimes

Pick the `cuXXX` tag that matches your CUDA driver version (e.g. `cu121`,
`cu124`, `cu125`). Commands (in the project's activated environment):

```bash
# 1) PyTorch with CUDA (AI models, SmartDeblur, ICEdit) — see also install.ps1 -Cuda and BUILD.md
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# 2) llama-cpp-python with CUDA (AI assistant: LLM + vision model)
pip install --force-reinstall --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

# 3) onnxruntime-gpu (CodeFormer face restoration)
pip install onnxruntime-gpu

# 4) ICEdit stack (uses the CUDA torch from step 1)
pip install diffusers transformers accelerate peft gguf sentencepiece protobuf
```

> A plain `pip install llama-cpp-python` and `pip install onnxruntime` install
> the **CPU builds** — for the GPU you must use the CUDA llama wheel and the
> `onnxruntime-gpu` package. If there's no llama CUDA wheel for your tag,
> install the nearest lower one (CUDA drivers are backward compatible).

### Step 3. Verify the GPU is actually used

```bash
# PyTorch sees CUDA:
python -c "import torch; print('torch CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# llama-cpp built with GPU offload:
python -c "import llama_cpp; print('llama GPU offload:', llama_cpp.llama_cpp.llama_supports_gpu_offload())"

# onnxruntime has the CUDA provider:
python -c "import onnxruntime as o; print(o.get_available_providers())"  # should include CUDAExecutionProvider
```

Additionally:

- The application's **status bar** shows the active device name (e.g.
  `Device: NVIDIA GeForce RTX 3070 Ti`).
- When the AI assistant starts, llama prints a line like
  `load_tensors: offloaded 36/36 layers to GPU` in the log — this confirms
  offloading. Logs: `~/.upscaler/upscaler.log`.
- On Windows (WDDM), `nvidia-smi --query-compute-apps` often does **not** show
  the processes — that is not a sign of CPU-only work; rely on the `offloaded`
  line.

### VRAM and robustness

- Tiled inference automatically selects the tile size to fit free VRAM; on OOM
  the tile is halved (up to 2 attempts), then transparently falls back to CPU.
- FP16 autocast speeds up inference; if NaNs/garbage appear, it auto-retries in FP32.
- VRAM guidance: upscalers ~0.35–0.5 GB, colorizers ~0.6–0.9 GB, the AI
  assistant (gemma/qwen ~4B Q5 + vision) ~4–5 GB, ICEdit ~6–8 GB.
- If the GPU can't hold everything at once, that's fine: components load one at
  a time and are unloaded after use.

## Running

```bash
python run.py
```

or

```bash
python -m upscaler.main
```

## Building a standalone application (.exe)

The project builds into a standalone executable via PyInstaller — the end user
doesn't need to install Python or the dependencies.

**Quick way — the `build.py` script** (checks dependencies and models, runs the
tests, then runs PyInstaller):

```bash
pip install pyinstaller          # once
python build.py                  # dist/Upscaler/ folder (recommended)
python build.py --onefile        # single file dist/Upscaler.exe (slower startup)
python build.py --no-models      # don't bundle the AI models
python build.py --clean          # clean the previous build's artifacts
```

The script auto-detects all plugins (`--hidden-import`), collects `torch` and
`spandrel` in full (`--collect-all`), and bundles local models from
`upscaler/models/models/`. The result is `dist/Upscaler/Upscaler.exe`.

The build size depends on which PyTorch is installed: the CPU variant produces
a build several times smaller than CUDA. For which PyTorch wheel to install for
a CPU or GPU build, see [BUILD.md](BUILD.md) (variants A/B).

A detailed step-by-step guide (system requirements, installing the CUDA/CPU
PyTorch variant, pre-downloading models, the manual PyInstaller command, build
sizes, and troubleshooting) is in [BUILD.md](BUILD.md).

The application opens a window with three panels:

- **Left panel** — loading, preset selection, scale, plugin checkboxes,
  parameter sliders, the process button. Sections collapse/expand by clicking
  the header (the state is remembered between runs); a combo box for the
  upscaler model replaces the former checkbox list. The "Processing order"
  group lets you set the sequence of pipeline steps by drag-and-drop (the
  "Reset" button restores the automatic order). Switching between single and
  batch mode.
- **Center** — a "Face zones" toolbar above the canvas (marking mode, "Clear",
  a "Zones: N" counter synchronized with the "Faces" tab) and the before/after
  canvas with a divider. Drag the divider, scroll to zoom, hold to pan.
- **Right panel** — version history with thumbnails. Revert, compare, or delete
  any version.

The **RU/EN** button to the right of the "Process" button instantly switches
the interface language (menus, panels, status bar, dialogs) without restarting
the app; the language choice is remembered between runs.

### "Colorization" tab

A separate tab for colorizing black-and-white photos and video:

| Model | Type | Variants | VRAM |
|-------|------|----------|------|
| DDColor | Photo | Artistic, ModelScope | ~900 MB |
| DeOldify | Photo + video | Stable, Artistic, Video | ~600 MB |
| ColorMNet | Photo + video | — | ~700 MB |

### "SmartDeblur" tab

Restoration of out-of-focus and motion-blurred images via deconvolution (a port
of the SmartDeblur algorithms to numpy/scipy). The **"SmartDeblur"** checkbox
in the "Restoration" group of the left panel enables auto mode: the blur type,
radius, motion angle, and regularization are estimated automatically (blind
estimation from the spectrum cepstrum and the shape of the radial spectrum).
Deblur is also enabled automatically by the "Auto" button when noticeable blur
is detected.

The **"SmartDeblur"** tab provides manual tuning with a preview:

| Parameter | Purpose |
|-----------|---------|
| Blur type | Defocus (disk) / Motion / Gaussian |
| Radius | Blur kernel size |
| Angle | Motion direction (for motion blur) |
| Smoothing | Regularization (noise and ringing suppression) |
| Edge feather / Edge correction | Fine-tuning the defocus kernel |
| Method | Wiener (fast) / Tikhonov / Total Variation / Richardson-Lucy |
| Iterations | Iteration count for TV and Richardson-Lucy |
| Ringing suppression | Edge taper — edge blending to remove FFT ringing |

Deblur runs in the pre-processing stage, before upscaling (deconvolution at
native resolution). Ringing suppression (edge taper) and the Richardson-Lucy
method are implemented following the SmartDeblur papers.

### "ICEdit" tab

Editing an image by natural-language instruction (In-Context Edit). Just
describe the desired change — e.g. "make the hair green" or "remove the
watermark" — and ICEdit applies it. It uses a quantized FLUX.1-Fill-dev (GGUF,
~6–8 GB) + LoRA (`Normal` by default; `MoE` is experimental: its weights were
withdrawn by the ICEdit author, and the format requires custom code not
supported by the standard diffusers loader).

The "ICEdit" checkbox in the "Editing" group of the left panel lets the vision
model decide on its own whether an edit is needed and phrase the instruction
(with the checkbox off, ICEdit is ignored entirely). On the "ICEdit" tab you
can enter an instruction manually, choose the LoRA, the number of steps, seed,
quantization mode, and offload, and press "Preview".

Requirements: a GPU with ~6–8 GB VRAM (default `model` offload); the first run
downloads the weights (~7–9 GB total). Dependencies: diffusers, transformers,
accelerate, peft, gguf. Without them / without the weights, ICEdit is simply
skipped. The default parameters match the official ICEdit inference: guidance
50, 28 steps; if an edit isn't applied, check the log — LoRA loading is now
verified and a failure is reported explicitly.

### "Blending" tab

The variant list on the tab is built automatically from snapshots of the
pipeline steps of the CURRENT processing run (every step that changed the image
becomes its own variant, labeled with the step name); the list is reset and
repopulated on every processing run ("Process" / "Make beautiful"), so the tab
always reflects only the current run, not variants from a previous one.

Selecting two variants in the list (first/second, as in history) immediately
shows their comparison on the canvas (before/after). The "Blend selected"
button blends the first variant as the base and the second as the overlay layer
(a blend mode from 25 Photoshop modes: from Multiply/Screen/Overlay to the
component Hue/Saturation/Color/Luminosity, plus opacity) — the result is shown
on the canvas immediately and added to the variant list as a new item so it can
be used in further blending. "Preview" shows the result on the canvas without
saving; "Apply" saves it as a new version in the permanent history. The
"Auto-select" button builds a recipe automatically by a quality metric
(sharpness, contrast, saturation, exposure, noise) over all accumulated
variants, shows the result on the canvas immediately, and also adds it as a
variant.

The automatic mode of the same feature is enabled by the "Variant blending"
checkbox in the "Single" panel: then the pipeline itself collects the
intermediate variants and, if compositing improves the metric, applies the best
recipe as the final step (including in refinement iterations). Candidates are
scored by the algorithmic quality metric in both AI-assistant modes.

### Face restoration (CodeFormer)

If the image contains faces, auto mode enables CodeFormer: faces are detected
(YuNet), aligned, and restored by a neural network, then softly pasted back.
The "Face restoration" checkbox in the "Restoration" group and the "Faces" tab
(the "Fidelity" slider, "Enhance background") control the process. It runs after
upscaling. Requires `onnxruntime` (optional); without it the step is skipped.
The CodeFormer weights (~360 MB) and the YuNet detector (~340 KB) are downloaded
on demand when a working source is configured; if the source is unavailable or
`onnxruntime` isn't installed, the step is skipped without errors.

Face zones can be marked manually: on the "Faces" tab, enable "Marking mode" and
draw frames on the canvas (move and resize by the corners, rotate by the marker,
Delete to remove). Marked zones replace auto-detection and are processed by
CodeFormer at its pipeline step; the "Delete"/"Clear" buttons manage the zone
list.

### AI assistant (vision model)

The "AI assistant" checkbox turns the vision model on/off. When off, the
automatic buttons use only algorithmic parameter selection (`AutoConfigurator`)
and don't launch the iterative re-evaluation of the result.

### Automatic parameter selection (LLM advisor)

The **"Auto"** and **"Make beautiful"** buttons first compute the processing
parameters algorithmically (`AutoConfigurator` from `SourceAnalyzer` metrics).
If the `use_llm_advisor` setting is enabled and a local GGUF model is available
(`upscaler/models/models/`), those parameters, together with a description of
the image, are passed to the model, which returns a refined JSON of parameters
tailored to the specific content. The model can set the **entire set of
processing parameters**: scale, upscaler, denoiser and its strength, adjusters
(Auto Tone/Contrast/Color, brightness/contrast/saturation/sharpness),
post-processing sharpening/refocus, and the **full SmartDeblur block** (blur
type, radius, angle, smoothing, edge feather, edge correction, method,
iterations, edge taper). The deblur parameters for the "Auto"/"Make beautiful"
buttons are computed by the model and reflected on the "SmartDeblur" tab. The
result is validated, clamped to safe ranges, and layered on top of the
algorithmic config. Inference runs in a separate thread (QThread) without
blocking the interface.

The model receives the image directly if a multimodal runtime
(`llama-cpp-python`) is installed **and** an mmproj file (vision projector) is
present; otherwise a detailed text description of the image metrics is used. If
the runtime/model is missing or on any error, there's a transparent fallback to
algorithmic mode.

Installation (optional): `pip install "llama-cpp-python>=0.3"` and placing a
GGUF model in `upscaler/models/models/`. For the AI assistant to run on the GPU,
you need a **CUDA build** of `llama-cpp-python` — see
[Maximizing GPU usage](#maximizing-gpu-usage).

### Iterative refinement (agent mode)

After processing, the vision model looks at the result again and evaluates it.
If it can be improved, it forms a new refinement pipeline (without re-upscaling)
and applies it to the resulting image — and so on, up to a total maximum number
of pipelines. Each iteration is saved as a separate history entry. Works in
"Make beautiful" and "Process" when the AI advisor is enabled. The maximum
number of re-processings (1–10) is set manually in the "Single" panel (the
"Refinement" group): 1 — a single pass without evaluation, N — up to N−1
automatic refinements of a result judged to be poor.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+O` | Open image |
| `Ctrl+P` | Process |
| `Ctrl+Z` | Revert to the previous version |
| `Space` | Toggle before/after |
| `Ctrl+0` | Fit to window |
| `+` / `-` | Zoom in / out |

## Architecture

```
Image → ANALYSIS → PRE-PROCESSING (denoise → DEBLUR → adjust) → SELECTION → SCALE → POST-PROCESSING → FACE RESTORATION → COLORIZATION → BLENDING → VALIDATION → Result
```

**Nine pipeline stages:**

1. **Analysis** — noise estimation, dynamic range, color space
2. **Pre-processing** — denoising, deblur (SmartDeblur, deconvolution), and adjustment
3. **Selection** — upscaler choice, multi-pass strategy planning (8x = 4x + 2x)
4. **Scale** — tiled upscaling with overlap, reducing artifacts between passes
5. **Post-processing** — sharpening, refocus
6. **Face restoration** (`face_restore`) — detection (YuNet), alignment, CodeFormer-ONNX, paste-back; runs after upscaling; optional, skipped without `onnxruntime`
7. **Colorization** — DDColor, DeOldify, or ColorMNet (optional)
8. **Blending** (`blend`) — intermediate processing variants composited on top of each other (25 Photoshop modes), mode/opacity chosen by a quality metric; optional, enabled by the "Variant blending" checkbox
9. **Validation** — quality metrics and artifact detection

**Two-process architecture:** the GUI runs in one process, the engine in a
separate worker process communicating over JSON-lines via stdin/stdout. The UI
stays responsive during heavy GPU computation.

### Plugin system

All processing is built on auto-discovered plugins:

```python
from upscaler.plugins.registry import PluginRegistry

registry = PluginRegistry()
registry.discover_builtin()

upscalers  = registry.list_plugins("upscaler")    # 11 plugins
denoisers  = registry.list_plugins("denoiser")     # 6 plugins
adjusters  = registry.list_plugins("adjuster")     # 7 plugins
colorizers = registry.list_plugins("colorizer")    # 3 plugins
deblur     = registry.list_plugins("deblur")        # 1 plugin (SmartDeblur)
face       = registry.list_plugins("face")          # 1 plugin (CodeFormer-ONNX)
```

### AI models

Models are downloaded automatically on first use into `~/.upscaler/models/`.
Resumable download with SHA256 verification.

| Model | Type | Scales | VRAM | Download |
|-------|------|--------|------|----------|
| Real-ESRGAN | Upscaler | 2x, 4x | ~500 MB | Auto |
| HAT-S | Upscaler | 2x, 4x | ~450 MB | Auto |
| SwinIR | Upscaler | 2x, 4x | ~400 MB | Auto |
| OmniSR | Upscaler | 2x, 4x | ~350 MB | Auto |
| DAT | Upscaler | 2x, 4x | ~500 MB | Auto |
| SCUNet | Denoiser | — | ~300 MB | Auto |
| NAFNet | Denoiser (SIDD/GoPro) | — | ~250 MB | Local |
| DDColor | Colorizer (Artistic/ModelScope) | — | ~900 MB | Auto (HuggingFace) |
| DeOldify | Colorizer (Stable/Artistic/Video) | — | ~600 MB | Local |
| ColorMNet | Colorizer | — | ~700 MB | Local |

**Local models** — must be placed in `upscaler/models/models/`:
- `NAFNet-SIDD-width64.pth`, `NAFNet-GoPro-width64.pth`
- `ColorizeStable_gen.pth`, `ColorizeArtistic_gen.pth`, `ColorizeVideo_gen.pth`
- `DINOv2FeatureV6_LocalAtten_s2_154000.pth`

Multi-pass upscaling for large scales: 8x = 4x + 2x, 16x = 4x + 4x. On OOM the
tile size is halved (up to 2 attempts), then falls back to the CPU.

## Settings

Stored in `~/.upscaler/settings.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `gpu_device` | `"auto"` | `"auto"`, `"cpu"`, or `"cuda:0"` |
| `default_output_format` | `"png"` | `"png"`, `"tiff"`, or `"exr"` |
| `tile_size` | `512` | Tile size (128–2048) |
| `max_history_entries` | `50` | Max versions per session |
| `history_retention_days` | `7` | Auto-cleanup of old sessions |
| `theme` | `"system"` | `"system"`, `"light"`, or `"dark"` |
| `use_llm_advisor` | `true` | Refine auto parameters with a local GGUF model (if available) |
| `auto_predownscale` | `true` | Auto-downscale soft images before processing |
| `panel_sections` | `{}` | Internal, automatic: collapsed/expanded state of the left-panel sections |
| `window_geometry` | — | Internal, automatic: window geometry (base64), restored on launch |

## Testing

```bash
python -m pytest tests/ -v
```

## Logs

Logs are written to `~/.upscaler/upscaler.log` (DEBUG level).

## Feedback

Please send all improvement suggestions, remarks, and bug reports to:
**pipirstein@gmail.com**. I'll be glad to hear from you.

## Support the project

I made this project for myself, but I decided to share it with everyone. If you want to
support me - write me an email and I will send the details

Thank you for using Upscaler!
