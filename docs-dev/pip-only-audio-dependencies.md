# Pip-only audio dependencies (no system FFmpeg)

This document records the architecture and design rationale for the app's audio dependency boundary. A standard `pip install -r requirements-<model>.txt` gives a working app, including export, chapters, enhancement, previews, Tools, STT, and every local model, without installing FFmpeg on the system.

Related documents:

- [audio I/O](<audio-io.md>): module reference, coding rules for app-owned audio code, the passive coexistence warning, and verification tools. That is the day-to-day reference. This document explains why the boundary looks the way it does.
- [TTS model architecture](<tts-model-architecture.md#model-side-ffmpeg-boundary>): a short summary of the model-side boundary for adapter authors.
- [Enhance audiobook flow](<enhance-audiobook-flow.md>) and the [ABR metadata spec](<abr-metadata-spec.md>): the main consumers of the decoder and writers.

## Status

Implemented and in use. Earlier releases piped PCM into the `ffmpeg` executable for export, ran a second `ffmpeg` pass for M4B chapters, decoded with ffmpeg-python/`ffprobe`, and checked for the executable at startup. All of that is gone.

| Platform | State |
|---|---|
| Linux x86_64 | Validated: automated suites under the no-FFmpeg guard, the smoke script (plain, inside a no-system-FFmpeg runner, and with FFmpeg deliberately put on the search path), a clean `python:3.11-slim` container install, and every local model in the runner, except the variants listed under [Open items](#open-items). |
| Windows x86_64 | Largely validated by hand: smoke script, manual app checks, and model runs with FFmpeg hidden from `PATH`. No per-item record was kept. Issues found there were fixed (see [History](#history)). |
| macOS (Apple Silicon) | Unvalidated. The code paths, including dyld-based loaded-library reporting, exist; the latter has mock coverage only. |
| Linux aarch64 | Unvalidated. Wheels exist for the pinned versions. |

## Policy

The rule is applied at three levels:

1. **App-owned audio I/O is strict.** Export, chapters, enhancement, preview, Tools, common file loading, voice import, and STT input use only wheel-provided libraries (PyAV, soundfile, Mutagen). They **never fall back to system FFmpeg, even when it is installed**. Every machine therefore runs one tested code path with one pinned encoder and muxer. Unsupported or corrupt input is an application error, never a subprocess decoder.
2. **Third-party opportunistic use is tolerated.** Upstream model packages may load system FFmpeg when it happens to be present. The main example is torchaudio 2.8, whose `import` probes for and loads system FFmpeg 4/5/6. That is acceptable as long as no supported flow *depends* on it. The proof is a run in an environment with no system FFmpeg, not an inspection of what got loaded where it is present.
3. **Per-model exceptions are explicit.** A local model may require system FFmpeg shared libraries only when avoiding that would be unreasonable. It must then declare `requires_ffmpeg_libs = true` in the [model catalog](<../tts_audiobook_tool/tts_models/model_catalog.toml>), with a reason and a README install section, and it gets a model-specific startup message. **No model currently declares an exception.**

Clarifications:

- FFmpeg code bundled inside a PyAV binary wheel is allowed. The goal is pip-managed installation, not the removal of FFmpeg technology or of all native libraries.
- No supported format was dropped to make the dependency disappear: FLAC, AAC/M4A/M4B, chapters, ABR metadata, enhancement of arbitrary inputs, previews, and every local model's voice cloning are retained.
- External model servers (SGL-Omni, audio.cpp) have their own deployment dependencies and are out of scope. This covers the Python app and its in-process local models.
- Playback is a separate stack. `sounddevice` uses PortAudio, which is a system library on Linux (`libportaudio2`), and some ALSA plugins it loads link system FFmpeg. That is audio output, not file I/O, and is tolerated. `sounddevice` is imported lazily on first playback, so file-I/O modules do not pull it in.

## Supported platforms

| OS | Architecture | Notes |
|---|---|---|
| Windows | x86_64 | arm64 has PyAV/soundfile wheels but is not validated |
| Linux | x86_64, aarch64, glibc ≥ 2.28 | musl and ARMv7 are not supported. PortAudio (`libportaudio2`) is required for playback |
| macOS | Apple Silicon, **macOS 14 (Sonoma) or later** | |

Python ≥ 3.11 throughout.

Why macOS 14:

- Intel Macs were already excluded: PyTorch publishes no Intel macOS wheels for 2.6+, and every environment uses torch 2.6 or newer.
- PyAV's Apple Silicon wheels require macOS 14+ in every release from 16.1 through 19.x, so no PyAV version choice lowers the floor.
- Current mlx (used by mlx-whisper on Apple Silicon) is also macOS 14+ only.
- Sonoma runs on every Apple Silicon Mac and is a free upgrade.

## Packaging

### The PyAV pin

Every requirements file pins `av==18.1.0`, matching the repo's existing exact pins for soundfile, mutagen, and torch.

- **Why pin at all.** Unpinned venvs had drifted from 16.1 to 19.0, which bundle different FFmpeg builds (libavformat 62 vs 63). Since AAC encoding and MP4 muxing now happen inside PyAV, the bundled FFmpeg *is* the export encoder. One pinned version means one tested encoder and muxer.
- **Why 18.1.0.** av 19.x requires Python ≥ 3.12 (`cp312-abi3` wheels only), and several model venvs run Python 3.11. 18.1.0 ships one `cp311-abi3` wheel per platform, covering every supported target.
- **The one exception: VibeVoice.** [requirements-vibevoice](<../requirements-vibevoice.txt>) uses `av>=16.1,<17`, because vibevoice requires aiortc, and aiortc 1.14 caps av below 17. vibevoice's own code imports neither. This venv therefore runs export and M4B chapters on PyAV 16.1, which is verified separately (see [Writers](#writers)).
- faster-whisper only requires `av>=11`, and no other model environment constrains PyAV.

### Wheel-only installs

Every requirements file carries `--only-binary=av,soundfile` as an option line, so `pip install -r` enforces it with no extra step. On an unsupported platform pip fails clearly instead of trying to build PyAV against system FFmpeg headers.

soundfile has a caveat: on platforms without a libsndfile-bundling wheel, pip selects a generic `py2.py3-none-any` wheel that needs a system libsndfile. `--only-binary` does not catch that, because the generic wheel is still a binary distribution. This is acceptable only because those platforms are unsupported.

### Startup dependency checks

Existing venvs are not reinstalled when a user pulls, so startup checks the environment ([start.py](<../tts_audiobook_tool/start.py>)):

- `av` is in the new-packages list. Apple Silicon venvs never had it, since mlx-whisper does not need it.
- `av` has a minimum version of 16.1 (`get_minimum_package_versions`). Windows and Linux venvs already had *some* av through faster-whisper, at whatever version was current when the venv was made. A presence check alone would pass and then fail later, for example at the first chaptered M4B export (`OutputContainer.set_chapters()` arrived in PyAV 16.0). The check is a floor, not the exact pin.
- Either failure exits with a "dependencies have changed" message naming the package, the installed version, and the `pip install -r` command for the current venv.
- The old FFmpeg-executable check (`exit_on_missing_ffmpeg_exe`) was removed. The shared-library check survives only as the [per-model exception mechanism](#the-per-model-exception-mechanism).

pip does not remove packages dropped from requirements files: ffmpeg-python, and pydub in the Higgs venv, stay installed in old venvs. Nothing imports them, so this is harmless for users. When testing a dependency removal, use a fresh venv or uninstall the package by hand.

## App-owned audio I/O

### Component roles

| Library | Role |
|---|---|
| PyAV | Compressed-audio demux and decode, stateful resampling, AAC encoding, MP4/M4B muxing including chapters |
| soundfile (libsndfile) | Streaming FLAC writing, and reading the formats libsndfile supports (WAV, FLAC, OGG, MP3) |
| Mutagen | ABR custom tags and FLAC/MP4 metadata, written after the container is closed |
| librosa | DSP and resampling only, never file decoding |

### What replaced what

| Flow | Before | Now |
|---|---|---|
| Audiobook FLAC export | PCM chunks piped to `ffmpeg` | `FlacWriter` (soundfile, streaming) |
| Audiobook AAC/M4B export | PCM piped to `ffmpeg` | `AacWriter` (PyAV encoder and muxer) |
| M4B chapters | A second `ffmpeg` pass (FFMETADATA, stream copy) | `set_chapters()` on the same muxer, just before close |
| Enhancement transcription | ffmpeg-python decode, downmix, and resample | Shared `AudioDecoder` |
| Enhancement preview | `ffmpeg` seek and bounded decode | `AudioDecoder` range reads |
| Enhancement output | `ffmpeg` transcode of non-M4A/M4B inputs | `transcode_to_aac()` (the copy/tag path for M4A/M4B is unchanged) |
| Tools: directory concatenation | `ffmpeg` concat demuxer | `convert_audio_files()` |
| Tools: FLAC → AAC | `ffmpeg` encode, then Mutagen tags | `transcode_to_aac()` plus the existing metadata transfer |
| Common loading (`SoundFileUtil.load`) | `librosa.load`, with an audioread fallback | soundfile first, then PyAV |
| STT input | Arrays (by convention) | Arrays, enforced by `require_waveform()` |
| Legacy two-pass `loudnorm` | Production module | Developer-only script in `testx/` |

The [module table in audio I/O](<audio-io.md#modules>) lists the modules and their callers.

### Writers

[audio_writers](<../tts_audiobook_tool/sound/audio_writers.py>) provides `FlacWriter` and `AacWriter` behind `open_writer()`. Both were built directly from the export path rather than designed as a general interface ahead of use.

- **Lifecycle.** Open, write rendered chunks with explicit sample rate and channels, then close or abort. Export pushes every rendered segment into one open writer, so the DSP pipeline (pauses, gaps, reverb, room tone, panning, leveling, timing maps) is independent of the output backend.
- **Staging.** Output is written to `<dest>.partial` in the destination directory and moved into place with `os.replace` only after a successful close. Abort and every failure path close the native handle and delete the staging file. An existing destination is never touched by a failed write.
- **Failures are loud.** Encoder, muxer, and disk errors raise `AudioWriteError`. Closing a writer that received no audio is also an error: libsndfile would otherwise leave a zero-byte FLAC, and FFmpeg an MP4 with no track. The old export stream ignored `BrokenPipeError` and never checked `ffmpeg`'s exit code.
- **FLAC:** 16-bit (clip, ×32767), compression level 6 (`compression_level=0.75`, which libsndfile maps to `lrint(x * 8)`).
- **AAC:** PyAV's native `aac` encoder on planar float input (`fltp`; interleaved stereo is transposed), with continuous pts, the delayed packets flushed at close, and the existing bitrate choices. Container is `ipod` for `.m4a`/`.m4b` and `mp4` for `.mp4`, always `movflags=+faststart`.
- **Chapters** are set after the last packet is muxed and before close, from the final rendered durations. This was verified on PyAV 16.1 through 19.0 to produce the same structure as the old CLI remux: a Nero `chpl` atom, a QuickTime chapter text track referenced by `tref`/`chap`, and identical titles and times. Chapter times are rounded to milliseconds. [m4b_chapter_util](<../tts_audiobook_tool/sound/m4b_chapter_util.py>) keeps only the range computation and `make_chapters()`.
- **Title and artist** are set on the container for chaptered output, without `movflags=use_metadata_tags`. PyAV then writes iTunes `©nam`/`©ART`/`©too` atoms, which Mutagen's later ABR rewrite preserves and players read. The legacy remux never actually wrote these, so chaptered M4B files gained a real title and artist.
- **The file object.** `AacWriter` opens the staging file itself and hands PyAV a Python file object. PyAV only closes an output file it opened itself if the header was written, so a failed header write leaked the handle. Abort closes the file before the container, so PyAV skips the trailer and the fast-start copy of a partial file. On PyAV 16/17 (the VibeVoice range), which writes the trailer anyway, a discardable file wrapper turns later I/O into no-ops, so an aborted export does not print "PyAV library exception being dropped" tracebacks.

### Decoder

[audio_decoder](<../tts_audiobook_tool/sound/audio_decoder.py>) provides `AudioDecoder` and `read_audio()`, shared by enhancement, previews, conversion, common loading, and canonical voice references.

- **Stream selection** is `container.streams.best("audio")`. On a two-track file it picks the default-disposition track, matching `ffmpeg`'s default. Cover art (MP3 APIC, M4A `covr`) and video streams are ignored.
- **Bounded chunks.** It yields float32 chunks at an explicit output rate and channel count (at most 2). It never loads a whole audiobook.
- **Resampling.** One stateful libswresample converter per pass, flushed at EOF and rebuilt only if the input format changes mid-stream, so there is no per-chunk rounding drift.
- **Channel mapping is explicit numpy.** Stereo → mono is the channel mean, and mono → stereo duplicates at full level. More than two channels are first folded to stereo with libswresample's standard matrix. PyAV's float resampler does not normalize its rematrix (stereo → mono comes out as 0.707·(L+R), which can clip, and mono → stereo is −3 dB), so it is not used for this.
- **Timeline.** Zero is the stream's `start_time`, as with the CLI's `-ss`. Positions are counted in output samples from the first decoded frame, so timestamp gaps are not filled, also as with the CLI.
- **Range reads (previews)** seek 1 s early to cover AAC/MP3 decoder pre-roll, then discard and trim at the output rate. If the seek lands late, they retry from 30 s earlier and then from the start. The residual error against a full decode is the resampler's sub-sample phase.
- **Errors.** A corrupt packet (`InvalidDataError`) is skipped with a log warning, as the CLI did, so one bad frame does not stop transcription of a long audiobook. Open, demux, and other decode failures raise `AudioDecodeError`.
- **Cancellation.** `cancel_check` is polled between packets.
- It also exposes container tags and chapters, which `transcode_to_aac()` copies.

### Conversion

[audio_convert](<../tts_audiobook_tool/sound/audio_convert.py>) decodes sources one after another into a writer. Every decoder is closed before the writer closes, and writer staging, abort, and destination protection are inherited.

- **Single-file transcode** keeps the source rate if AAC accepts it (otherwise 48 kHz) and at most 2 channels. It copies container tags (except `encoder`) and chapters, matching the CLI's default `-map_metadata`/`-map_chapters`. Cover art, subtitle, and data streams are dropped, as `-vn -sn -dn` did.
- **Concatenation** uses the highest input sample rate (AAC-valid ones only) and stereo if any input is not mono. No tags or chapters are copied. Mixed codecs and rates through the old concat demuxer failed or produced broken output.

### Common loading

`SoundFileUtil.load` reads with soundfile first and uses PyAV only for formats libsndfile cannot read, such as AAC/M4A ([sound_file_util](<../tts_audiobook_tool/sound/sound_file_util.py>)). Both decoders come from wheels, so this is not a system fallback. soundfile-first keeps WAV/FLAC/MP3/OGG samples bit-identical to the `librosa.load` era.

`librosa.load` is not used for file decoding anywhere in app-owned code. It calls `audioread.available_backends()` on every call, before trying soundfile, so the first load in any process launched `ffmpeg -version`, and unreadable files fell through to audioread's FFmpeg backend.

### STT input

faster-whisper and mlx-whisper adapters accept only waveform arrays. `require_waveform()` in [stt](<../tts_audiobook_tool/stt.py>) raises `TypeError` for anything else before the backend is called, because mlx-whisper shells out to `ffmpeg` for path inputs. A runtime check was chosen over a test alone, since a test cannot catch future call sites.

### Handle discipline and Windows

Development happened on Linux, where file-locking mistakes are invisible. The design therefore follows these rules, and the tests use Linux proxies for Windows behavior:

- **Close before touching.** Windows cannot replace, delete, or rename an open file, and Mutagen cannot rewrite one. Every container, soundfile handle, and decoder is closed, including on error and cancellation paths, before `os.replace`, unlink, staging cleanup, or tag writes.
- **No lingering handles.** Preview and enhancement objects do not keep a decoder open on a file the user may later overwrite or that output staging replaces.
- **Linux proxy.** Tests assert that no file descriptor remains open on the input, staging, or output path after success, error, and abort (`/proc/self/fd`, or `psutil` open files in the smoke script).
- **Non-ASCII paths.** `str` paths are passed through unchanged. Tests and the smoke script repeat checks in non-ASCII folders.
- **No DLL search tricks.** App-owned code never adds DLL directories or relies on FFmpeg DLLs found via `PATH`. On Windows, Python 3.8+ does not search `PATH` for an extension module's dependent DLLs at all.

### Enforcement

The boundary is enforced by tests, not just convention:

- [no_ffmpeg_guard](<../tests/no_ffmpeg_guard.py>) patches `subprocess.Popen.__init__`, `os.system`, `os.posix_spawn[p]`, `os.spawn*`, and `os.exec*`. Blocked launches raise and are recorded, so a swallowed error still fails the test. Use the `no_ffmpeg_exe` marker, or `--no-ffmpeg-exe` for the whole suite.
- [test_no_ffmpeg_static_audit](<../tests/audio/test_no_ffmpeg_static_audit.py>) parses every production module outside `tts_models/` and flags `ffmpeg`/`ffprobe`/`ffplay` command strings, audioread/pydub/ffmpeg-python/torchaudio/TorchCodec imports, and `librosa.load`.
- App-owned modules must not import torchaudio. Importing torchaudio 2.8 loads system FFmpeg when present, which would break the "never falls back" guarantee for the process.

## Deliberate output changes

These were accepted on purpose and announced in the README's update highlights:

- **AAC input is float.** FFmpeg's AAC encoder only accepts planar float, and the old path converted float → int16 → float. This is a simplification, not an audible quality change: int16 noise (about −96 dBFS) is far below AAC's own noise at 64–128 kbps, and the level difference from dropping ×32767 scaling is about 0.0003 dB.
- **FLAC stays int16.** Bitstreams and compressed sizes differ from FFmpeg's encoder, but decoded samples do not.
- **All AAC/M4A/M4B output is fast-start**, including chaptered files (the old chapter remux put `moov` after `mdat`) and the Tools FLAC → AAC output.
- **Chapters are written in the same pass as the audio.** The separate remux and its `[chaptermeta]` intermediate file are gone. Debug-tag filtering of `[chaptermeta]` and `[norm]` files is kept so that old export folders stay clean.
- **Chaptered M4B files have a real title and artist** (see [Writers](#writers)).
- **Encoder failures are surfaced**, and success is never reported after incomplete output.
- **Enhancement windows are float32** straight from the decoder, instead of FFmpeg s16 / 32768.
- **Bug fixes made during migration:** chapter times for ranged output with several Book sections (wrong duration slice), the Tools concat output name (`".m4a"` instead of `concatenated.m4a`), and the FLAC → AAC existence check (`with_stem` instead of `with_suffix`).

## Model-side architecture

### The problem

Model libraries load voice references with their own loaders: librosa, torchaudio, soundfile, TorchCodec, pydub. Which of these need FFmpeg depends on the library version and on the input format, not on anything the app controls.

Behavior without system FFmpeg, measured in a no-system-FFmpeg runner with WAV, FLAC, MP3, and M4A inputs:

| Loader | `import` | WAV / FLAC / MP3 | M4A |
|---|---|---|---|
| torchaudio 2.6 / 2.8 | works (backends `sox`, `soundfile`) | works through soundfile | fails |
| torchaudio 2.9.1 (TorchCodec 0.8.1) | works | **every `load()`/`save()` fails** ("Could not load libtorchcodec"), even WAV with `backend="soundfile"` | fails |
| `import torchcodec.decoders` (0.8.1, 0.10.0) | fails | — | — |
| librosa 0.11 | works | works through soundfile | fails (`NoBackendError`) |

With system FFmpeg present, torchaudio 2.8 and 2.9 load all four formats, and 2.8 prefers its FFmpeg backend even for WAV. 2.8's import-time probe (`get_available_backends()` → `lazy_import_ffmpeg_ext().is_available()`, trying FFmpeg 6, 5, 4, then unversioned) cannot be disabled; `TORIO_USE_FFMPEG_VERSION` selects a version but does not skip the probe.

The resulting classification, used to audit every model:

- **Works without FFmpeg:** waveform or tensor inputs; librosa or soundfile reads of PCM WAV/FLAC; torchaudio 2.6/2.8 `load()` of PCM WAV/FLAC.
- **Hard-requires FFmpeg:** TorchCodec decoders, any torchaudio 2.9 file call, audioread or pydub on formats soundfile cannot read, and any import that fails without FFmpeg libraries.

### Canonical voice references

The main lever is to make sure upstream loaders only ever see files that soundfile can read. Adapters get voice paths from `ProjectVoiceUtil.resolve_model_voice_file_path()`, never `resolve_voice_file_path()`, and [canonical_reference](<../tts_audiobook_tool/sound/canonical_reference.py>) does the rest:

- **Canonical** means libsndfile format WAV, WAVEX, or FLAC, with subtype PCM_16/24/32, FLOAT, or DOUBLE. Such files pass through unchanged. Voice import already decodes with the app's decoder and saves 16-bit FLAC, so app-imported samples are always canonical.
- **Anything else**, typically an MP3 or M4A copied into an older project before import did the conversion, is decoded once with the app's PyAV decoder (native rate, at most two channels) into a 16-bit FLAC under `<temp>/tts_audiobook_tool/voice_references/`. The name is derived from the source path, mtime, and size, and older copies for the same source are pruned.
- **Cache invalidation is preserved.** Model voice-clone caches are keyed by path, mtime, and size. Because the canonical copy is named from the source's identity and never rewritten, those caches invalidate exactly when the source changes.
- **MP3 is not canonical** even though libsndfile reads it, because loaders disagree on it: the soundfile route includes the encoder delay (25344 vs 24000 samples in a probe).
- **No fallback.** An existing but undecodable reference raises `VoiceReferenceError`. Handing the original path to the model would let librosa or torchaudio retry it through system FFmpeg when present, which is an implicit fallback.
- Canonical copies live outside the project, so project transfer and the voice directory are unaffected, and a deleted cache simply regenerates.
- Remote adapters (SGL-Omni, audio.cpp) do not use this layer; their servers own their own loading.

### Hard requirements are avoided, not patched

Where an upstream path hard-requires FFmpeg, the adapter supplies already-decoded waveforms or tensors through a public API, or the canonical file. Loaders that merely *prefer* system FFmpeg when present are left alone, and there is no global monkeypatching of torchaudio or import blocking of TorchCodec.

MOSS is the case that needed this. Its torchaudio 2.9.1 file I/O always goes through TorchCodec, and the remote processor's `encode_audios_from_path` is a hard requirement. The adapter never passes paths: the voice clone is `SoundFileUtil.load` followed by `encode_audios_from_wav`, rolling continuation re-encodes the previous output waveform, messages carry code tensors only, and decoding goes from codes to a tensor. TorchCodec is installed in that venv but never imported.

### Model status

Result of the per-model audit and runs in the no-system-FFmpeg runner, with M4A-imported, MP3-imported, and legacy raw-MP3 references. No adapter needed code changes beyond using canonical references.

| Model | torchaudio | Reached loading path | Notes |
|---|---|---|---|
| Fish S1 | 2.8.0 | Adapter `torchaudio.load()` of the canonical reference (soundfile); upstream generation is token-only | pydub is a fish-speech dependency, not reached |
| Fish S2 | 2.8.0 | As Fish S1; rolling continuation reuses cached tokens | Reference path validated; full S2-Pro generation not run on Linux |
| GLM | 2.8.0 | Upstream `torchaudio.load()` of the canonical path (speech tokenizer, `load_wav`) | TorchCodec 0.10 and pydub come from `glm-tts-packaged`'s dependencies; neither is imported |
| MOSS | 2.9.1 | App loader → `encode_audios_from_wav`; code tensors only | Former `requires_ffmpeg_libs` exception, removed |
| Chatterbox (V3, V2, Turbo) | 2.6.0 | Upstream `prepare_conditionals` → `librosa.load` | |
| dots.tts (SOAR, MeanFlow) | 2.8.0 | `DotsTtsRuntime` → `librosa.load` | |
| Higgs v2 | 2.8.0 | `HiggsAudioTokenizer.encode(path)` → `librosa.load`; collator gets tokens only | pydub removed from its requirements file; validated on CPU |
| IndexTTS2 | 2.8.0 | `infer_v2._load_and_cut_audio` → `librosa.load` (speaker and emotion references); output returned as an array | opencv-python's bundled FFmpeg is a pip wheel and is not loaded |
| Mira | 2.8.0 | ncodec `encode` → `librosa.load` | |
| OmniVoice | 2.8.0 | `load_audio`: soundfile first | Imports pydub, but only builds `AudioSegment`s from arrays, which launches nothing |
| Qwen3 (Base) | 2.8.0 | `_load_audio_to_np` → `librosa.load` | CustomVoice/VoiceDesign take no reference audio |
| VibeVoice | 2.8.0 | Processor → `librosa.load` | av 16.1 (range exception) |
| Breeze | 2.8.0 | `sf.read` → waveform | |
| Pocket | 2.8.0 | stdlib `wave` for WAV, soundfile otherwise | |
| LavaSR (enhancement, all venvs) | 2.8.0 | In-memory arrays only | |

An `audioread` import by librosa, or a pydub import, is not counted as a failure. Only a call that decodes a non-soundfile format would launch `ffmpeg`, and the runner would fail it.

### The per-model exception mechanism

The mechanism is kept permanently, even though nothing uses it now, so that a future model with an unavoidable TorchCodec-only path has a supported route:

- Exceptions are data, not code paths. The model sets `requires_ffmpeg_libs = true`, `ui.ffmpeg_libs_reason`, and `ui.readme_install_section` in the [model catalog](<../tts_audiobook_tool/tts_models/model_catalog.toml>). The catalog parser requires both `ui` fields when the flag is set.
- At startup, `Start.exit_on_missing_ffmpeg_libs` runs the ctypes library search in [ffmpeg_util](<../tts_audiobook_tool/sound/ffmpeg_util.py>) only for a declaring model. If the libraries are missing, it exits with a message that names the model and the reason, says the rest of the app needs no FFmpeg, gives the step for the current OS, and links the model's README section. Other models are unaffected.
- The app never silently enables an FFmpeg-dependent path because the libraries happen to be present.
- The Windows DLL-directory registration (`os.add_dll_directory()`) was removed when MOSS dropped its exception. A future Windows exception has to restore it: since Python 3.8, neither the startup check nor TorchCodec would otherwise find an FFmpeg that is only on `PATH`.
- [test_ffmpeg_libs_exception](<../tests/models/test_ffmpeg_libs_exception.py>) covers the mechanism with a synthetic declaring model.

### Coexisting FFmpeg builds

A model worker may hold PyAV's bundled FFmpeg and a system FFmpeg loaded by torchaudio at the same time. On Linux this has always been the case whenever system FFmpeg is installed (for example faster-whisper's PyAV plus torchaudio's probe), without known problems. It is tolerated.

The worker and the standalone REST server take passive snapshots of their loaded FFmpeg libraries at operation boundaries and log a deduplicated **`FFMPEG_MULTIPLE_BUILDS`** warning to the persistent file log when they see more than one library directory or duplicate library families. This is a diagnostic signal, not evidence of a symbol or class conflict, and it never blocks an operation. Details, log locations, and limits are in [audio I/O](<audio-io.md#passive-coexistence-warning>) and [model worker architecture](<model-worker-architecture.md>).

Reconsider isolation only if a concrete clash is observed in a supported flow.

#### Using the warning in a hardening pass

The app is deliberately wired to log when multiple FFmpeg builds are loaded, so that a future hardening pass can start from evidence rather than speculation. If a native crash, hang, or garbled decode is ever suspected to come from coexisting builds, start here:

- **Where to look.** Search for `FFMPEG_MULTIPLE_BUILDS` in `<tempdir>/tts-audiobook-tool-worker.log` (interactive model worker) or `<tempdir>/tts-audiobook-tool.log` (standalone REST server). Ask users who report native crashes for these files.
- **What each entry gives you.** The PID, the operation context (worker startup, a command name, server init, or synthesis), every loaded FFmpeg library path, PyAV's bundle directories, and which of `av`, `torchaudio`, `torio`, `torchcodec`, `cv2`, and `sounddevice` were already imported. Correlating the context with the failing operation narrows down which model, venv, and code path brought in the second build. The module list is a hint, not attribution.
- **Producing a report on demand.** `ffmpeg_lib_report()` in [loaded_ffmpeg_libs](<../tts_audiobook_tool/system_support/loaded_ffmpeg_libs.py>) returns the same data (`libs`, `directories`, `outside_bundled`, `single_build`, `only_bundled`, `describe()`) for the current process. Calling it in a probe before and after a suspect import or call pins down exactly what loaded a library.
- **Reproducing the clean case.** Rerun the same model in the [no-system-FFmpeg runner](#verification-architecture) (Linux) or with FFmpeg removed from `PATH` (Windows). If the problem disappears there, coexistence is implicated; if it persists, it is not.
- **Known sources of a second build:** torchaudio 2.8's import-time probe; TorchCodec, if an upstream path ever imports it; PortAudio's ALSA plugins on Linux (for example `libasound_module_pcm_a52.so` links system `libavcodec`); and other pip wheels that bundle their own FFmpeg, such as opencv-python's `opencv_python.libs` in the IndexTTS2 venv.
- **Candidate measures,** previously judged unwarranted without a demonstrated clash (see [Rejected alternatives](#rejected-alternatives)), in rough order of intrusiveness: route the affected model's reference loading through waveforms instead of upstream path loaders; set `TORIO_USE_FFMPEG_VERSION` or otherwise steer torchaudio away from an incompatible system version; block Torio/TorchCodec initialization in the affected worker; or, for app-owned code, keep doing what the smoke script already enforces (bundle-only libraries).
- **Limits.** Checks are snapshots at operation boundaries. A native crash or hang before the next check, a library loaded and unloaded between checks, or a separate helper process can be missed. The absence of a warning is not proof of isolation, and inspection failures are logged only at DEBUG (`FFMPEG_LIBRARY_CHECK_UNAVAILABLE`). Windows and macOS behavior of the warning is not yet validated (see [Open items](#open-items)).

## Verification architecture

The tools are described in [audio I/O](<audio-io.md#verification>). Their design roles:

- **No-FFmpeg guard and static audit** keep app-owned code from launching or importing FFmpeg paths.
- **Loaded-library helper** ([loaded_ffmpeg_libs](<../tts_audiobook_tool/system_support/loaded_ffmpeg_libs.py>)) reports which FFmpeg libraries a process has mapped and where they come from: psutil memory maps on Linux and Windows, current-process dyld images on macOS. It lives in the production package because the smoke script and the worker diagnostics both use it. It locates PyAV's bundle (`av.libs`, or `av/.dylibs` on macOS) without importing `av`.
- **Smoke script** ([no_ffmpeg_smoke](<../testx/no_ffmpeg_smoke.py>)) is the portable end-to-end check of app-owned flows: FLAC and chaptered M4B export through the real `ConcatUtil.make_file`, enhancement decode and staged output, preview range reads, both Tools commands, and startup in a child process. Every check repeats in a non-ASCII folder. After each check it verifies no guard violations, no open handles, deletable outputs, and FFmpeg libraries only from PyAV's bundle. The strict library check is valid here because the smoke process runs only app-owned I/O. `--ffmpeg-on-path DIR` reruns it in a fresh process with an FFmpeg shared build on `PATH` (and on `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`) to prove that build is ignored.
- **No-system-FFmpeg runner** ([no_system_ffmpeg_run](<../testx/no_system_ffmpeg_run.py>), Linux) runs a venv's Python in an unprivileged user and mount namespace that bind-mounts `/dev/null` over every system FFmpeg executable and library, then maps back to the real uid/gid. It was chosen over a container because it reuses the venvs, GPU device nodes, and HF cache as-is, changes nothing on the host, and starts instantly. It verifies itself first. It needs unprivileged user namespaces (stock Ubuntu 24.04+ restricts them via AppArmor).
- **Model clone check** ([no_ffmpeg_model_clone](<../testx/no_ffmpeg_model_clone.py>)) imports M4A and MP3 references through the app, plus a legacy raw-MP3 reference, into a temporary non-ASCII project, generates once with each, and reports loaded libraries and imported FFmpeg-capable modules. It is strict inside the runner (`NO_SYSTEM_FFMPEG_RUNNER=1`) and report-only elsewhere.
- **Windows equivalent of the runner:** remove every `PATH` entry containing `ffmpeg.exe` or `avcodec-*.dll` for the session, confirm `Get-Command ffmpeg, ffprobe` finds nothing, then run the smoke script and the clone check directly. Conda and other apps sometimes put FFmpeg DLLs on `PATH`.
- **Probes kept for upgrades:** [pyav_m4b_chapters](<../testx/pyav_m4b_chapters.py>) (chapter and fast-start structure on a given PyAV version), [export_chapters_vs_cli](<../testx/export_chapters_vs_cli.py>) (structural comparison against the legacy CLI path; uses system `ffmpeg`, developer-only), [export_long_file](<../testx/export_long_file.py>) (multi-hour streaming export: memory, duration, seekability), and [pip_only_audio_check](<../testx/pip_only_audio_check.py>) (bare-environment check, e.g. in a `python:3.11-slim` container).

Long-file behavior from a 6-hour export: FLAC peak RSS growth was about 8 MB. M4B peak RSS grew linearly at about 11 MB per hour, which is FFmpeg's MP4 sample index held until `moov` is written, the same as the old `ffmpeg` subprocess.

## Maintenance

### Bumping PyAV

1. Check that the new version has `abi3` wheels for every supported platform at the lowest Python any venv uses, and for macOS 14.
2. Rerun [pyav_m4b_chapters](<../testx/pyav_m4b_chapters.py>) on the new version, plus `tests/audio` and `tests/enhance` with `--no-ffmpeg-exe` and the smoke script.
3. Check whether the VibeVoice range can follow (aiortc's cap), and whether the startup minimum in `get_minimum_package_versions` still matches the oldest version any requirements file allows.

### Adding a local model

1. Copy the app dependency block, including the PyAV pin and `--only-binary`, into the new requirements file.
2. Get voice paths only from `resolve_model_voice_file_path()`. If an upstream path hard-requires FFmpeg, pass waveforms or tensors instead.
3. Validate with `python3 -m testx.no_system_ffmpeg_run --venv <venv> -- -m testx.no_ffmpeg_model_clone`.
4. Declare an exception only if avoiding FFmpeg is unreasonable, and restore Windows DLL registration if that model supports Windows.

### Adding app-owned audio code

Use the decoder, writers, and conversion helpers. Follow the [rules in audio I/O](<audio-io.md#rules-for-app-owned-audio-code>). The static audit and the guard will flag most regressions.

## Rejected alternatives

- **Requiring a single FFmpeg build in model processes.** The original design also required that model processes never load system FFmpeg and hold at most one build. It was dropped: torchaudio 2.8's import probe cannot be disabled, torchaudio is in nearly every model venv, and mixed builds had already been the norm in Linux workers with no known problems. "Works without FFmpeg" is proven in the runner instead, and the `FFMPEG_MULTIPLE_BUILDS` warning exists so this can be revisited with evidence (see [Using the warning in a hardening pass](#using-the-warning-in-a-hardening-pass)).
- **Replacing PyAV with torchaudio for app I/O.** It would not yield a single build either: faster-whisper imports PyAV, and torchaudio's media backend differs between 2.6/2.8 and 2.9 (where it is TorchCodec, which needs system FFmpeg).
- **Import blockers or global audio-I/O monkeypatches** (blocking TorchCodec or Torio initialization, replacing `torchaudio.load`/`save`). These could reject otherwise working operations without a demonstrated clash, and they are fragile across upstream versions.
- **Keeping `ffmpeg` as an optional fallback** for odd inputs. Two code paths, with behavior depending on the machine, is what the design set out to remove.
- **Keeping legacy loudness normalization in the package.** It was the last production `ffmpeg` invocation. It now lives in [testx/ffmpeg_loudnorm.py](<../testx/ffmpeg_loudnorm.py>), used only by a developer A/B script.
- **A container for the no-FFmpeg runner.** See [Verification architecture](#verification-architecture).

## Open items

Nothing here blocks normal use. These are gaps in validation, plus follow-ups outside this repo.

- **macOS (Apple Silicon, macOS 14+):** clean install per README, startup, and the smoke script, ideally once with Homebrew FFmpeg installed to confirm app-owned I/O ignores it. mlx-whisper's path rejection and the dyld-based library report have only stand-in and mock coverage.
- **Linux aarch64:** run the smoke script in an aarch64 venv.
- **Large model variants in the Linux runner:** full Fish S2-Pro generation (needs about 27 GB of host RAM to load), full MOSS Delay v1.5 and Local v1.5 generation and a MOSS GPU run, Higgs on a GPU, and VibeVoice 7B. Each passed the reference/audio paths or a CPU run, or shares its loader with a variant that passed. The Windows pass may have covered some of them; no per-model record was kept.
- **Variants not run separately:** the Qwen3 Darwin fine-tune and Pocket's predefined voices. Both share a reference path with variants that passed.
- **Passive mixed-build warning on Windows and macOS:** confirm a single-build run stays quiet and that a deliberately mixed run logs `FFMPEG_MULTIPLE_BUILDS` once with correct paths.
- **Other installers:** whether `uv pip install -r` honors the `--only-binary` option line has not been checked.
- **glm-tts-packaged:** drop the unused TorchCodec (and pydub, after confirming nothing imports it) from that fork's dependencies, then reinstall venv-g and rerun the runner check.
- **Out-of-scope observation:** the Fish S1/S2 adapters run `_create_voice_clone` outside `torch.no_grad()`, so the DAC encode records autograd state. That costs VRAM headroom, not FFmpeg independence.

## History

The work was done in phases and landed on 2026-11-02 and 2026-11-03, in this order:

1. Packaging (the PyAV pin, `--only-binary`), the no-FFmpeg test guard, the loaded-library helper, and FFmpeg-free test fixtures.
2. Export through the new writers, with single-pass chapters (commit `c06b846`).
3. Enhancement, previews, Tools, common loading, and the STT guard through the shared decoder; ffmpeg-python removed (`4e627af`).
4. Startup executable check and README prerequisite removed, legacy loudnorm moved to `testx/`, static audit, smoke script, clean-container validation (`10b0c87`).
5. Model groundwork: canonical references, the runner, per-version torchaudio findings, the exception mechanism. Then per-model audits and runner validation, in which MOSS dropped its exception and pydub was removed from the Higgs requirements.
6. Windows validation, which led to fixes (`55413a8`, `df9e1c5`): the startup minimum-version check for `av`, quiet aborts on PyAV 16/17, screen clearing via an escape sequence instead of `cls` (which could leave a blank, unresponsive menu after Ctrl-C), and model-worker hangs after Ctrl-C (the Intel Fortran runtime's console handler, unbounded waits on workers Windows refuses to terminate).
7. Passive `FFMPEG_MULTIPLE_BUILDS` diagnostics and macOS dyld reporting (`6b649d0`).

Real-player checks of exported M4B and FLAC files (chapters, seeking, title, playback to the end) passed on 2026-11-02.
