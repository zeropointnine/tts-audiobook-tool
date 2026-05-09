# Sidon Integration

## Overview

[Sidon](https://github.com/sarulab-speech/Sidon) is a neural speech restoration system. It takes audio — typically noisy or low-quality — and reconstructs it as clean, high-fidelity speech. It is not conventional upsampling (interpolating existing samples); it uses a two-stage neural pipeline to synthesize new audio from semantic features extracted from the input.

**Pipeline:**
1. **Feature extractor** — a w2v-BERT 2.0 transformer that extracts speech features from 16 kHz input
2. **Vocoder** — a VOCOS-based decoder that synthesizes 48 kHz output audio from those features

Because the vocoder regenerates audio from scratch, output quality is determined by the vocoder's training data (LibriTTS-R, FLEURS), not by the quality of the input. Resolution loss from downsampling to 16 kHz before inference is irrelevant — the input is essentially just a semantic carrier.

---

## `SidonUtil`

**File:** [tts_audiobook_tool/sidon_util.py](../tts_audiobook_tool/sidon_util.py)

`SidonUtil` is a normal instantiable class (not static) following the same pattern as `PocketModel`, `ChatterboxModel`, etc. It holds loaded model instances and is intended to be created once and killed when no longer needed.

### API

```python
SidonUtil.has_sidon()                # → bool — is the sidon package installed?
util = SidonUtil()                   # downloads models on first run, loads into memory (CUDA only)
result = util.process(sound)         # Sound → Sound | str
util.kill()                          # releases model references
```

`process()` accepts a `Sound` at any sample rate and returns a `Sound` at 48 kHz, or an error string on failure.

### CUDA Only

`SidonUtil` is CUDA-only. The constructor accepts no `device` parameter — it always uses `"cuda"`. CPU model files exist on the HuggingFace repo but have hardcoded CUDA operations and do not work on CPU.

### Inference

- Input audio is resampled to **16 kHz** before feature extraction. w2v-BERT 2.0 was trained on 16 kHz audio and `extract_seamless_m4t_features` uses 16 kHz fbank parameters.
- Each chunk is **padded to a fixed 30-second window** (480,000 samples at 16 kHz) before feature extraction. The feature extractor requires this fixed-length input.
- Output from the decoder is trimmed back to the actual duration of the original input.
- Output sample rate is **48 kHz**.
- Every chunk runs a full **30-second inference pass** regardless of actual chunk length. For typical TTS sentence segments (3–10s) this is fine; the overhead is notable on GPU for many short segments.

### Inputs Longer Than ~29.8 Seconds

Inputs longer than a single chunk are split into sequential **29.8s** chunks (476,800 samples at 16 kHz) with a **200ms** overlap (3,200 samples at 16 kHz) at each boundary. Each chunk is processed independently, and outputs are crossfaded at boundaries using the 200ms overlap region (9,600 samples at 48 kHz). Because the VOCOS vocoder maps all chunks to the same learned distribution, timbral consistency across chunks is very good — the crossfade only needs to handle waveform-level discontinuities (phase and instantaneous amplitude), not spectral differences.

The full resampled waveform and all chunk tensors reside in memory simultaneously during batch processing, so peak memory usage scales with input duration.

### Loudness

Sidon does not normalize loudness. All outputs are regenerated through the same vocoder and will tend toward a consistent loudness level (reflecting the training data distribution), but this is a side effect rather than a guarantee. For reliable loudness consistency across segments, run `LoudnessNormalizationUtil` after Sidon as a separate step — the two complement each other.

### `has_sidon()`

The static method `SidonUtil.has_sidon()` returns `True` if the `sidon` Python package is importable, `False` otherwise. This allows the codebase to check for Sidon availability without requiring the package as a hard dependency. The `SidonUtil` class is safe to import even when the `sidon` package is not installed.

---

## App Integration

Sidon is used as an optional generative upsampler in the audio post-processing pipeline. It is not part of the core TTS generation flow, but rather a quality improvement step applied to generated audio segments.

### User-Facing Toggle

**File:** [concat_menu.py](../tts_audiobook_tool/concat_menu.py)

The concat menu exposes a "Generative upsampling" menu item that toggles `state.project.use_upsampler`. The option only appears when both conditions are met:

- `torch.cuda.is_available()` — a CUDA-capable GPU is present
- `SidonUtil.has_sidon()` — the `sidon` package is installed

If either condition is false, the menu item shows a subheading explaining why it is unavailable ("Requires CUDA" or "Sidon upsampler not installed").

### Model Lifecycle

**File:** [models_util.py](../tts_audiobook_tool/models_util.py)

`ModelsUtil` manages a singleton `SidonUtil` instance across the app:

- **`get_sidon_upsampler()`** — Lazy-initializes on first call. Checks CUDA availability and package presence, then creates `SidonUtil()` with a "Initializing Sidon upsampler (CUDA)..." message.
- **`clear_sidon_upsampler()`** — Calls `.kill()` on the instance, sets it to `None`, and runs `MemoryUtil.gc_ram_vram()` to release GPU memory.
- **`clear_all_models(except_sidon=True)`** — Clears TTS, STT, and YAMNet models while preserving the Sidon instance. Used before concat to free VRAM for the upsampling pass.

### Post-Processing Pipeline

**File:** [sound_app_util.py](../tts_audiobook_tool/sound_app_util.py)

`SoundAppUtil.apply_segment_post_processing()` is the central pipeline for generated audio segments, used by the concat, realtime playback, and voice chat flows. Sidon is step 2 in the chain:

1. Optional silence-gap limiting (`SilenceUtil.limit_silence_gaps`)
2. **Optional generative upsampler (Sidon)** — called via `ModelsUtil.get_sidon_upsampler().process(sound)`
3. Resample to `APP_SAMPLE_RATE`
4. Optional high-shelf EQ (`SoundExtraUtil.high_shelf_eq`)

If Sidon processing returns an error string, it is propagated up to the caller.

### Concat Flow

**File:** [concat_util.py](../tts_audiobook_tool/concat_util.py)

When `state.project.use_upsampler` is enabled, the concat flow performs these preflight checks:

1. **Package check** — If Sidon is enabled but the library is not installed, prints a warning and continues without upsampling.
2. **VRAM management** — If other models (TTS, STT, YAMNet) are loaded, clears them via `ModelsUtil.clear_all_models(except_sidon=True)` to free VRAM for the upsampling pass.
3. **Per-segment processing** — Each generated segment passes through `apply_segment_post_processing` with `use_upsampler=True`.
4. **Cleanup** — After concat completes, `ModelsUtil.clear_sidon_upsampler()` releases the model and runs GC.

### Installation

**File:** [start.py](../tts_audiobook_tool/start.py)

The `sidon` package is conditionally added to the list of packages to install during setup, allowing the app to function without it.

---

## Model Weights

Pre-trained weights are published at [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1). Four `.pt` files are provided (CPU and CUDA variants for each model component):

| File | Size |
|------|------|
| `feature_extractor_cpu.pt` | 795 MB |
| `feature_extractor_cuda.pt` | 795 MB |
| `decoder_cpu.pt` | 210 MB |
| `decoder_cuda.pt` | 210 MB |

`SidonUtil.__init__` downloads the **CUDA** pair (`feature_extractor_cuda.pt` and `decoder_cuda.pt`) via `hf_hub_download` on first use and caches them in the standard HuggingFace cache directory. The CPU model files exist on the repo but have hardcoded CUDA operations and are not usable.

---

## Dependencies

Sidon is installed from a project-specific fork into the pocket virtual environment:

```
sidon @ git+https://github.com/zeropointnine/tts-audiobook-tool-sidon
```

Fork repo: [zeropointnine/tts-audiobook-tool-sidon](https://github.com/zeropointnine/tts-audiobook-tool-sidon)

This fork exists because `tts-audiobook-tool` only uses Sidon for inference, not training. The upstream package includes training-oriented dependencies and import paths that are unnecessary for this app's Sidon usage.

The fork makes two project-specific changes:

- It trims install dependencies so they only cover the Sidon inference code path used by `tts-audiobook-tool`.
- It removes eager training-module imports from `src/sidon/model/__init__.py`, making the package safer to install and import in a lightweight inference environment.

The only runtime dependencies for inference are `torch`, `torchaudio`, and `numpy` — all already present in the various `requirements-*.txt` files.

---

## Security: .pt Files

PyTorch `.pt` files are pickle-serialized and can execute arbitrary code on load. `torch.jit.load()` (used for TorchScript models) carries the same risk. See **Next Steps** for the recommended ONNX migration path.

---

## Performance

| Environment | Real-Time Factor |
|-------------|-----------------|
| Single GPU (reference) | ~500× faster than real-time |
| RTX 3080 Ti (estimated) | ~500× — 1s audio ≈ 2ms |

CPU inference is not supported — the TorchScript model files contain hardcoded CUDA operations.

---

## Next Steps

### ONNX Migration

ONNX files are protobuf-serialized graph specs with embedded weights — no code execution on load, eliminating the `.pt` pickle risk.

Migration path:
1. Load the `.pt` files **once** in a sandboxed environment (throwaway container or VM)
2. Export both models to ONNX via `torch.onnx.export()`
3. Upload the resulting `.onnx` files to a personal HuggingFace repo
4. Update `SidonUtil` to download `.onnx` files via `hf_hub_download` and replace `torch.jit.load` with `onnxruntime.InferenceSession`

Because inputs are padded to a fixed 30s window, input shapes are deterministic, which makes TorchScript→ONNX export significantly less error-prone than variable-shape models. The main export consideration is the feature extractor's dict output (`["last_hidden_state"]`) — ONNX outputs are positional, so output names must be declared explicitly during export.
