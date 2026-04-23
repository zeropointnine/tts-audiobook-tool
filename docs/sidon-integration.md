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
util = SidonUtil(device="cuda")      # downloads models on first run, loads into memory
result = util.process(sound)         # Sound → Sound | str
util.kill()                          # releases model references
```

`process()` accepts a `Sound` at any sample rate and returns a `Sound` at 48 kHz, or an error string on failure.

### Inference

- Input audio is resampled to **16 kHz** before feature extraction. w2v-BERT 2.0 was trained on 16 kHz audio and `extract_seamless_m4t_features` uses 16 kHz fbank parameters.
- Input is **padded to a fixed 30-second window** (480,000 samples at 16 kHz). The feature extractor requires this fixed-length input.
- Output from the decoder is trimmed back to the actual duration of the original input.
- Output sample rate is **48 kHz**.
- Every call runs a full **30-second inference pass** regardless of actual clip length. For typical TTS sentence segments (3–10s) this is fine; on CPU with many short segments, the overhead accumulates.

### Inputs Longer Than 30 Seconds

Inputs longer than 30 seconds are split into sequential 30s chunks with a short overlap at each boundary. Each chunk is processed independently, and outputs are crossfaded at boundaries (20–50ms overlap). Because the VOCOS vocoder maps all chunks to the same learned distribution, timbral consistency across chunks is very good — the crossfade only needs to handle waveform-level discontinuities (phase and instantaneous amplitude), not spectral differences.

### Loudness

Sidon does not normalize loudness. All outputs are regenerated through the same vocoder and will tend toward a consistent loudness level (reflecting the training data distribution), but this is a side effect rather than a guarantee. For reliable loudness consistency across segments, run `LoudnessNormalizationUtil` after Sidon as a separate step — the two complement each other.

---

## Model Weights

Pre-trained weights are published at [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1). Four `.pt` files are provided (CPU and CUDA variants for each model component):

| File | Size |
|------|------|
| `feature_extractor_cpu.pt` | 795 MB |
| `feature_extractor_cuda.pt` | 795 MB |
| `decoder_cpu.pt` | 210 MB |
| `decoder_cuda.pt` | 210 MB |

`SidonUtil.__init__` downloads the appropriate pair via `hf_hub_download` on first use and caches them in the standard HuggingFace cache directory.

---

## Dependencies

Sidon is installed from git into the pocket virtual environment:

```
sidon @ git+https://github.com/sarulab-speech/Sidon
```

Its full `pyproject.toml` dependency tree includes training-only packages (hydra, wandb, ray, lightning, peft, etc.) that are not needed for inference. Install with `--no-deps` to avoid pulling in several GB of training stack:

```bash
pip install --no-deps "sidon @ git+https://github.com/sarulab-speech/Sidon"
```

The only runtime dependencies for inference are `torch`, `torchaudio`, and `numpy` — all already present in `requirements-pocket.txt`.

---

## Security: .pt Files

PyTorch `.pt` files are pickle-serialized and can execute arbitrary code on load. `torch.jit.load()` (used for TorchScript models) carries the same risk. See **Next Steps** for the recommended ONNX migration path.

---

## Performance

| Environment | Real-Time Factor |
|-------------|-----------------|
| Single GPU (reference) | ~500× faster than real-time |
| RTX 3080 Ti (estimated) | ~500× — 1s audio ≈ 2ms |
| CPU (estimated) | ~5–20× faster than real-time |

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
