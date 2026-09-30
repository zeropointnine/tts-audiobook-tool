# LavaSR v2 Integration

Last updated: 2026-08-14

## Overview

[LavaSR](https://huggingface.co/YatharthS/LavaSR) is an optional neural speech
restoration and bandwidth-extension model. The app uses its v2 `LavaEnhance2`
model to produce 48 kHz mono audio from generated speech.

## Adapter

The integration lives in
[`tts_audiobook_tool/sound/lava_sr_util.py`](../tts_audiobook_tool/sound/lava_sr_util.py).
`LavaSrUtil` can be imported without LavaSR installed because package imports
and model loading are deferred until availability checking or construction.

- `LavaSrUtil.has_lava_sr()` checks that `LavaSR.model.LavaEnhance2` exists.
- `LavaSrUtil()` downloads or loads `YatharthS/LavaSR` on first use.
- `process(sound, denoise=False)` restores one in-memory `Sound`, returning an
  error string on failure.
- `kill()` releases the model, worker, and accelerator cache.

The app's generative-upsample path deliberately calls `process()` with
`denoise=False`; it enables bandwidth enhancement without the optional denoise
pass.

## Devices and fallback

Automatic selection prefers CUDA, then Apple MPS, then CPU. A caller may also
request a supported device explicitly. If initialization or inference fails on
MPS, the adapter releases the MPS model and retries on CPU. Explicit CUDA and
CPU failures are returned to the normal app error path rather than silently
changing devices.

When the resolved device is CUDA and `isolate_cuda` is enabled (the default),
the model is not loaded in the application process. `LavaSrUtil` instead
spawns a dedicated worker process that exclusively owns the model and CUDA
context; NumPy audio and commands cross a `Pipe`. `kill()` asks the worker to
exit cleanly, then escalates to terminate/kill after a timeout, so driver
allocations are released when the process exits.

## Audio processing

Input is validated as non-empty, one-dimensional NumPy audio. It is resampled
to 16 kHz before inference and returned as 48 kHz float32 mono audio. Before
processing, the adapter configures LavaSR's Linkwitz-Riley refiner using half
the original sample rate as the source-bandwidth cutoff.

Long inputs are processed in 120-second chunks with 0.2-second overlap. The
48 kHz outputs are crossfaded in memory, and short or long model returns are
padded or trimmed to their expected lengths.

## App lifecycle and pipeline

`ModelManager` owns one lazy `lava_sr_upsampler` instance:

- `get_lava_sr_upsampler()` returns `None` when LavaSR is unavailable, or loads
  the adapter on first use.
- `clear_lava_sr_upsampler()` kills and drops the adapter. Model teardown,
  worker termination, and accelerator cache clearing all happen inside
  `kill()`; no additional memory collection is run.
- `clear_all_models(except_lava_sr=True)` can unload TTS, STT, and YAMNet while
  preserving an already-loaded LavaSR instance during export.

During concatenation, each segment is optionally routed through
`SoundPipeline.apply_lava_sr_upsampling()` before the app's normal 48 kHz
resampling and high-shelf processing.

## Optional-package behavior

The **Generative upsampling** setting is always visible. When LavaSR is not
installed, its submenu explains that status and refuses attempts to enable the
setting. The serialized field remains named `use_upsampler` for compatibility
with existing projects.

An older project may already contain `use_upsampler: true`. If such a project
is exported without LavaSR installed, concat emits one preflight warning and
continues without upsampling. The saved preference is not rewritten.

LavaSR is pinned in every requirements file and is listed in `start.py`'s
new-packages list, so the app exits at startup when the package is missing.
The graceful runtime paths above cover the case where an installed LavaSR
build does not expose the v2 `LavaEnhance2` class that
`LavaSrUtil.has_lava_sr()` looks for.
