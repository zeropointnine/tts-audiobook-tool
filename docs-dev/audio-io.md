# Audio file I/O and the FFmpeg boundary

The app's own audio file I/O — export, M4B chapters, enhancement, previews, the Tools commands, common file loading, and STT input — uses only libraries that pip installs from wheels:

- **PyAV** (`av`, pinned `18.1.0`): compressed-audio decoding, resampling, AAC encoding, MP4/M4B muxing including chapters. Its wheels bundle their own FFmpeg build (`site-packages/av.libs` on Linux/Windows, `av/.dylibs` on macOS).
- **soundfile** (bundles libsndfile): FLAC writing, and reading the formats libsndfile supports (WAV, FLAC, OGG, MP3).
- **Mutagen**: ABR metadata and FLAC/MP4 tags.

No system FFmpeg is needed: no `ffmpeg`/`ffprobe` executable and no FFmpeg shared libraries. The app also **never falls back** to system FFmpeg when it happens to be installed, so every machine runs the same tested encoder and muxer. Unsupported or corrupt input is an application error, never a subprocess decoder.

Every requirements file pins `av` and carries `--only-binary=av,soundfile`, so pip fails clearly rather than building either package from source on an unsupported platform.

## Modules

| Module | Role |
|---|---|
| [audio_writers](<../tts_audiobook_tool/sound/audio_writers.py>) | `FlacWriter` (soundfile, 16-bit, level 6) and `AacWriter` (PyAV native `aac`, fast-start, chapters at close), via `open_writer()`. Open / write / close / abort lifecycle; staging to `<dest>.partial` and `os.replace` on success; failures raise `AudioWriteError` and leave any existing destination untouched. |
| [audio_decoder](<../tts_audiobook_tool/sound/audio_decoder.py>) | `AudioDecoder` / `read_audio()`: decodes `streams.best("audio")` into bounded float32 chunks at an explicit rate and channel count, with one stateful resampler per pass, sample-accurate range reads (previews), and `cancel_check` between packets. Also exposes tags and chapters. Errors raise `AudioDecodeError`. |
| [audio_convert](<../tts_audiobook_tool/sound/audio_convert.py>) | File-to-file conversion built on the two above: `convert_audio_files()` (Tools concatenation) and `transcode_to_aac()` (enhancement output, Tools FLAC → AAC). |
| [sound_file_util](<../tts_audiobook_tool/sound/sound_file_util.py>) | `SoundFileUtil.load` / `read_mono` (soundfile first, PyAV for formats libsndfile can't read; librosa only resamples) and `load_range` (previews). |
| [stt](<../tts_audiobook_tool/stt.py>) | `require_waveform()`: STT adapters accept only arrays. mlx-whisper would launch `ffmpeg` for a path. |

Callers: export ([concat_util](<../tts_audiobook_tool/concat_util.py>)), enhancement ([enhance_alignment](<../tts_audiobook_tool/enhance/enhance_alignment.py>), [enhance_flow](<../tts_audiobook_tool/enhance/enhance_flow.py>)), and Tools ([mp3_concat](<../tts_audiobook_tool/sound/mp3_concat.py>), [transcode_util](<../tts_audiobook_tool/sound/transcode_util.py>)).

## Rules for app-owned audio code

- Do not launch `ffmpeg`/`ffprobe`, and do not decode files with `librosa.load` (its audioread fallback probes `ffmpeg`), audioread, pydub, or ffmpeg-python.
- Do not import torchaudio or TorchCodec from app-owned modules: importing torchaudio 2.8 loads system FFmpeg when present. Model adapters (`tts_models/`) are audited separately.
- Close every container, soundfile handle, and decoder before replacing, deleting, or tagging a file (Windows cannot touch open files). Long-lived objects must not hold a decoder open.
- Pass `str` paths through unchanged (non-ASCII paths must work).

Playback is separate: `sounddevice` (PortAudio) is imported only when something is played. On Linux, PortAudio is a system library (`libportaudio2`), and some ALSA plugins it loads link system FFmpeg; that is the audio-output stack, not file I/O.

## Models

Model libraries load voice samples with their own loaders. Without system FFmpeg, those still work on PCM WAV/FLAC: librosa and torchaudio 2.6/2.8 read them through soundfile. What fails without FFmpeg is formats libsndfile can't read (eg M4A, through audioread or torchaudio's FFmpeg backend), and every torchaudio 2.9 file call (always TorchCodec).

**Canonical voice references.** Adapters get voice paths from `ProjectVoiceUtil.resolve_model_voice_file_path`, never `resolve_voice_file_path`. Samples imported through the app are already canonical (decoded by the app, saved as 16-bit FLAC) and pass through unchanged. Anything else (eg an MP3 or M4A copied into an older project) is decoded once by the app's PyAV decoder into a cached FLAC under the app temp dir ([canonical_reference](<../tts_audiobook_tool/sound/canonical_reference.py>)), keyed by source path, mtime and size; so model voice-clone caches (keyed by path, mtime, size) invalidate as before. An undecodable file raises `VoiceReferenceError`; there is no fallback to the original file.

**The per-model exception.** A model may need system FFmpeg *shared libraries* only when avoiding it would be unreasonable. No model currently declares one: MOSS (torchaudio 2.9, whose file I/O always needs TorchCodec) avoids it by handing its processor app-decoded waveforms and audio codes, never paths. A declaring model sets `requires_ffmpeg_libs = true` in the [model catalog](<../tts_audiobook_tool/tts_models/model_catalog.toml>), plus `ui.ffmpeg_libs_reason` and `ui.readme_install_section` (both required by the catalog parser). At startup, `Start.exit_on_missing_ffmpeg_libs` looks for the libraries only for such a model ([ffmpeg_util](<../tts_audiobook_tool/sound/ffmpeg_util.py>)); if they are missing it exits with a message naming the model and the reason, the step for the current OS, and a link to the README section. Other models and the app itself are unaffected. The Windows DLL-directory registration (`os.add_dll_directory()`) was removed when MOSS dropped the exception, so a future Windows exception has to add it back. Third-party packages loading system FFmpeg opportunistically (e.g. torchaudio at import) is tolerated as long as no supported flow depends on it.

## Passive coexistence warning

The model worker checks its own loaded FFmpeg libraries at startup and between
commands (before waiting for the next command). The standalone REST server checks
after model initialization and after synthesis, including Python failures. This
catches lazy decoder imports without import hooks, polling threads, or checks in
audio callbacks. These are snapshots: a native crash/hang before a check, a library
loaded and unloaded between checks, or a separate helper process can be missed.

Search the persistent application logs for **`FFMPEG_MULTIPLE_BUILDS`**:
`<tempdir>/tts-audiobook-tool-worker.log` for the interactive worker, or
`<tempdir>/tts-audiobook-tool.log` for the standalone server. It is a WARNING with
the PID, operation context, loaded paths, PyAV bundle directories and already
imported consumer-module names (not proof of which module loaded a library).
Multiple directories or duplicate library families suggest multiple builds;
**this may be benign and is not evidence of a collision**. A system-only single
build does not trigger it. Each distinct suspicious path/family snapshot is
logged once per monitor/process; new library paths produce another warning.

This is file logging only, not a UI/session-log alert. It does not block imports,
change backends, unload libraries, or require any model to use a particular
loader. The check does not import PyAV/TorchAudio/TorchCodec. Inspection/logging
failures cannot fail an operation; `FFMPEG_LIBRARY_CHECK_UNAVAILABLE` is logged
at DEBUG once per monitor if inspection fails, and later checks still retry.

## Verification

- [no_ffmpeg_guard](<../tests/no_ffmpeg_guard.py>): blocks and records any FFmpeg executable launch. Mark tests with `pytest.mark.no_ffmpeg_exe`, or run the whole suite with `--no-ffmpeg-exe`.
- [Static audit test](<../tests/audio/test_no_ffmpeg_static_audit.py>): scans the production package (except `tts_models/`) for executable names, audioread/pydub/ffmpeg-python/torchaudio/TorchCodec imports, and `librosa.load`.
- [loaded_ffmpeg_libs](<../tts_audiobook_tool/system_support/loaded_ffmpeg_libs.py>): reports loaded FFmpeg libraries, their origins, and possible multiple builds. Uses psutil on Linux/Windows and dyld on macOS (current process only); also owns the passive runtime warning above.
- [no_ffmpeg_smoke](<../testx/no_ffmpeg_smoke.py>): portable end-to-end smoke test of export, enhancement, preview, Tools, and startup, in plain and non-ASCII folders, with handle, guard, and loaded-library checks after each step. `--ffmpeg-on-path DIR` reruns it with an FFmpeg shared build on `PATH` to prove it is not used.

  ```bash
  ./venv-base/bin/python -m testx.no_ffmpeg_smoke
  ```

- [no_system_ffmpeg_run](<../testx/no_system_ffmpeg_run.py>) (Linux): runs a venv's Python in a user + mount namespace that hides every system FFmpeg executable and library (bind-mounting `/dev/null` over them), reusing the venvs, GPU and HF cache. It verifies itself first (no executable, hidden libraries fail to load, nothing mapped from outside the venv after `import torchaudio`). Proves a model works on a machine without system FFmpeg.

  ```bash
  python3 -m testx.no_system_ffmpeg_run --venv venv-cb -- -m testx.no_ffmpeg_model_clone
  ```

- [no_ffmpeg_model_clone](<../testx/no_ffmpeg_model_clone.py>): for the venv's local model, imports M4A and MP3 references through the app (plus a legacy raw-MP3 reference) into a temporary project, generates with each, and checks the output and the loaded FFmpeg libraries. Meant to run inside the runner above.
- [pip_only_audio_check](<../testx/pip_only_audio_check.py>): minimal PyAV/soundfile/Mutagen check for a bare environment (e.g. a `python:3.11-slim` container).

Developer-only scripts in `testx/` may still use the system `ffmpeg`: [export_chapters_vs_cli](<../testx/export_chapters_vs_cli.py>) builds a legacy CLI baseline, and [ffmpeg_loudnorm](<../testx/ffmpeg_loudnorm.py>) (formerly `LoudnessNormalizationUtil`) runs whole-file two-pass `loudnorm` for [room_reverb_ab](<../testx/room_reverb_ab.py>).
