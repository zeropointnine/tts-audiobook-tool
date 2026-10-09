# MOSS-TTS model and backend architecture

## Terminology

MOSS has two independent axes that both use the word “local” in different ways:

1. **Application backend**
   - `TtsBackendKind.LOCAL`: the audiobook tool imports and runs `moss_tts` in its own model worker, normally from `venv-moss`.
   - `TtsBackendKind.SGL_OMNI`: the audiobook tool sends HTTP requests to an external SGL-Omni server, normally while running from `venv-client`.
   - `TtsBackendKind.AUDIO_CPP`: the audiobook tool sends HTTP requests to an external audio.cpp server, using the same remote-client environment.

2. **MOSS model architecture**
   - **Delay**: upstream `MossTTSDelay`, represented by `MossConfigs.DELAY`.
   - **Local Transformer**: upstream `MossTTSLocal`, represented by `MossConfigs.LOCAL`.

“MOSS Local” therefore means the upstream Local Transformer architecture. It does **not** necessarily mean in-process execution. The Local Transformer can be loaded directly by the app or served remotely through SGL-Omni or audio.cpp.

## Supported combinations

| Application backend | MOSS architecture | Catalog ID | Runtime implementation | Output rate |
|---|---|---|---|---:|
| Local/in-process | Delay or Local Transformer, selected by its private `target` setting | `moss_local` | `MossModel` | 24 kHz or 48 kHz |
| SGL-Omni | Delay | `moss_delay_sglomni` | `SglOmniBackendAdapter` with the `moss_delay_sglomni` TOML definition | 24 kHz |
| SGL-Omni | Local Transformer | `moss_local_sglomni` | `SglOmniBackendAdapter` with the `moss_local_sglomni` TOML definition | 48 kHz |
| audio.cpp | Community v1.5 8B Delay | `moss_delay_audiocpp` | `AudioCppBackendAdapter`, family `moss_tts_v15` | 24 kHz |
| audio.cpp | Local Transformer v1.5 | `moss_local_audiocpp` | `AudioCppBackendAdapter`, family `moss_tts_local` | 48 kHz |

The local/in-process catalog entry remains one `moss_local` type because its Hugging Face target is a private model setting and the loaded architecture is discovered from that target. Each remote backend exposes the two architectures as formal model types because server selection needs architecture-specific metadata even when no local model target is configured. The audio.cpp UI names mirror SGL-Omni: `MOSS-TTS Delay` and `MOSS-TTS Local`; Delay's community implementation is experimental and advertises English/Chinese support.

## Catalog identities and remote selection

The single [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) (`schema_version = 1`) declares the built-in local `moss_local` spec, four built-in remote variants, and the `"none"` placeholder alongside other model specs. It supplies stable IDs, backend kinds and metadata; remote entries also carry matching, behavior, parameters and menu definitions. The MOSS remote catalog IDs are:

- `moss_delay_sglomni`
- `moss_local_sglomni`
- `moss_delay_audiocpp`
- `moss_local_audiocpp`

The first pair has `TtsBackendKind.SGL_OMNI`; the second has `TtsBackendKind.AUDIO_CPP`. All use the `moss` file tag and own private generation settings, but primary voices now come from the shared project-wide list. Shared implementation and file tags do not imply shared generation state. Output sample rates and UI identities follow the selected architecture. [`tts_model_type.py`](<../tts_audiobook_tool/tts_models/tts_model_type.py>) installs canonical handles for every shipped entry in exact TOML order, including data-only models, without model-named class attributes or a Python declaration inventory. Use `TtsModelType.require_by_id("moss_delay_sglomni")` for a hardcoded ID (unknown IDs raise `ValueError`); use `TtsModelType.get_by_id(saved_id)` for persisted/external IDs (unknown IDs return the registered `"none"` handle, leaving the original string untouched). Both lookups return the same handle for a known ID; metadata is accessed through `.value`.

### SGL-Omni detection and matching

Remote discovery ([`remote_tts_discovery.py`](../tts_audiobook_tool/app_support/remote_tts_discovery.py)) queries the configured server and matches each served model ID to catalog types through [`sgl_omni_detection.py`](../tts_audiobook_tool/tts_models/sgl_omni_detection.py) using the catalog's `sgl_omni.match.model_id_substring`. Matching is case-insensitive and prefers the longest match:

- Local Transformer uses the specific `moss-tts-local` substring.
- Delay uses the generic `moss` fallback.

For example:

- `OpenMOSS-Team/MOSS-TTS-v1.5` resolves to `moss_delay_sglomni`.
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer` contains both matches, but the more specific Local match wins.

An unrecognized non-MOSS model ID does not resolve to either MOSS type. The compatibility helper `TtsModelType.find_tts_type_using_sgl_omni_model_id()` delegates to the same detector.

### audio.cpp detection and matching

[`audio_cpp_detection.py`](<../tts_audiobook_tool/tts_models/audio_cpp_detection.py>) matches structured server metadata, never model-ID substrings:

- `moss_tts_v15` + task `tts` or `clon` + mode `offline` resolves to `moss_delay_audiocpp`.
- `moss_tts_local` + task `tts` or `clon` + mode `offline` resolves to `moss_local_audiocpp`.

Server IDs are arbitrary operator-configured strings. No quantization/session-option hint is required, and the discovery API does not verify checkpoint revisions. Both families accept optional reference audio under either task token. Unrelated MOSS families and unsupported modes/tasks are not candidates.

### Selection and binding

The project stores a catalog ID; refer to [TTS model selection rules](<tts-model-selection.md>) for the canonical reconciliation and binding rules. For audio.cpp, `Tts.bind_project()` silently selects the first matching model entry in `/v1/models` response order, without sorting or preferring loaded entries. SGL-Omni still requires a unique match. Zero matches or an unreachable server block binding and leave the saved ID untouched.

Generation settings, output metadata, and architecture-dependent behavior come from the selected catalog definition; the runtime never reclassifies the served model from its ID during generation. The status UI still shows the exact server model ID as diagnostics, so a selection that disagrees with what the server serves is visible as a configuration error rather than silently overridden. Separate audio.cpp `tts` and `clon` entries for the same family remain one app model type; the first matching entry wins. Its exact model ID is runtime-only, so a later rebind can choose differently if the server response order changes.

### Pre-release IDs

Older pre-release builds persisted architecture-ambiguous IDs such as `server_moss`, plus a separate `sgl_omni_type` preference. Both are gone: those builds were never deployed, so no migration is provided. The current MOSS IDs are the five catalog types listed above; the new audio.cpp types do not rename any existing selection.

## Local model and configured server runtime

The in-process class hierarchy is `TtsBaseModel` → `MossBaseModel` → `MossModel`. `MossBaseModel` retains the MOSS configs, language-name mapping, local sampling and architecture behavior, and the local batch/rolling-continuation readiness rule. `MossConfigs.get_by_target()` identifies the architecture of a local target; the local `MossModel` loads it and runs inference.

Server generation does **not** subclass `MossBaseModel`. Each server variant's behavior and request definition lives alongside its spec in the schema-1 [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml), selected by its stable catalog ID. In SGL-Omni mode the complete catalog is validated before server specs are overlaid; optional validated server entries may add supported IDs with private registry settings. `ConfiguredModelSupport` provides metadata, readiness, output rate, music/trimming decisions and menu settings; one `SglOmniBackendAdapter` constructs the `/speech` request and calls `SglOmniUtil.generate_concurrent()`. Local MOSS inference still uses `MossModel`.

Local/in-process Local Transformer v1.5 installs an instance-scoped attention compatibility wrapper from [`moss_attention.py`](<../tts_audiobook_tool/tts_models/moss_attention.py>) after loading. Its pinned global Qwen decoder leaves finished samples in the batch with zero valid query tokens. Flash Attention 2 can fail on these mixed empty/nonempty query batches with a reshape error (reproduced as `shape '[2, 8, 4, 128]' is invalid for input of size 4096`). Only cached flash calls containing an empty query row fall back to the same decoder's SDPA path; prefill, active batches, single-item generation, the local GPT2 depth decoder and other model architectures remain unchanged. No upstream class, downloaded model source or shared Hugging Face cache is modified.

The SGL-Omni Delay and Local definitions use their own private sampling parameters and seed, the shared project voice references, a resolved seed and a fixed 1024-token limit. Their defaults and request semantics are unchanged. They do not inherit the local batch/rolling-continuation blocker. Architecture-dependent remote behavior follows the selected definition, never a second probe of the server model ID.

For audio.cpp, [`AudioCppModelSupport`](<../tts_audiobook_tool/tts_models/audio_cpp_configured.py>) supplies metadata/readiness and one `AudioCppBackendAdapter` sends sequential requests to `/v1/audio/speech`, bound to the exact discovered server ID. The model-specific controls map to top-level `temperature`, `top_p`, `top_k`, and `seed`; audio.cpp does not accept `audio_top_p`/`audio_top_k` as aliases on this path. Both variants support reference-less speech and optional base64 WAV references, including under task `tts`. Neither requests, requires, or transmits a reference transcript; local/SGL-Omni models may require one. The shared project's paired transcript nevertheless remains retained and editable when audio.cpp is selected.

Both audio.cpp variants keep their native sampling defaults but pin `options.text_chunk_size = 100000` to disable internal text splitting, preserving one app segment per generation. This fixed request option is not a project/menu setting. Repetition penalty is not exposed or overridden. Local pins `options.max_tokens = 1024` in every request, matching the local/SGL-Omni token budget numerically; audio.cpp counts audio frames per internal text chunk (about 81.92 seconds), rather than necessarily the same generation steps as the other backends. This fixed limit replaces the native 4096-frame default and is not a project/menu setting. Community Delay does not consume generic `max_tokens`, and its separate automatic duration bounds remain server-managed. No audio.cpp MOSS setting enables streaming, batching, concurrency, or rolling continuation.

Language hints reuse `MossBaseModel.get_language_name()` to map the project code to a full name. Delay reads `options.language`; Local reads top-level `language`. Unrecognized/empty MOSS hints are omitted. Local's server session caches prepared reference codes with one slot by default, configurable via the `moss_tts_local.reference_cache_slots` session option; community Delay reencodes each request. Neither cache is managed by the app.

## Project settings

MOSS generation settings are private under `model_settings.models.<catalog ID>`; there is no current `shared.moss` group. Primary voice references instead use top-level `Project.voice_references`, shared across all models within the project. Historically, the audio.cpp additions left project version 3 unchanged; the project-wide voice-list migration now uses project version 4. Existing IDs, catalog schema 1, and non-voice AuK/Qwen3 sharing are unchanged. Fish S2 later retired its local/server group through the same generic fork (see [project spec v4](<project-spec-v4.md>)).

| Owner | Private sampling settings |
|---|---|
| `moss_local` | All of `delay_temperature`, `delay_top_p`, `delay_top_k`, `local_temperature`, `local_top_p`, `local_top_k` and `local_v15_temperature`, `local_v15_top_p`, `local_v15_top_k` — one set per preset (`delay` = Delay 8B, `local` = Local Transformer v1.0, `local_v15` = Local Transformer v1.5) |
| `moss_delay_sglomni` | Only `delay_temperature`, `delay_top_p`, `delay_top_k` |
| `moss_local_sglomni` | Only `local_temperature`, `local_top_p`, `local_top_k` |
| `moss_delay_audiocpp` | Only `delay_temperature`, `delay_top_p`, `delay_top_k` |
| `moss_local_audiocpp` | Only `local_temperature`, `local_top_p`, `local_top_k` |

Every type reads the same project-wide ordered filename/transcript pairs. `moss_local` owns one seed per preset (`delay_seed`, `local_seed`, `local_v15_seed`); its retired shared `seed` migrates to all three on load, as does any historical `shared.moss`/flat `moss_seed` value. Remote types each own `parameters.seed`. Local/in-process and SGL-Omni types store `orchestration.batch_size`; audio.cpp types have no orchestration setting and do not consume reference transcripts, without discarding their stored paired text. Only `moss_local` owns `parameters.target` and `parameters.rolling_cont`; neither selects the remote architecture. Switching types or presets does not synchronize settings. Common MOSS controls and configured adapters reuse code, not storage.

Sampling `-1` still means “use this variant's default”; seed `-1` retains its random-seed meaning. Existing local/SGL-Omni defaults are unchanged, while audio.cpp uses its own native defaults:

| Backend | Architecture | Temperature default | Audio top-p default | Audio top-k default | Output rate |
|---|---|---:|---:|---:|---:|
| Local/in-process or SGL-Omni | Delay | 1.7 | 0.8 | 25 | 24 kHz |
| Local/in-process or SGL-Omni | Local Transformer v1.0 | 1.0 | 0.95 | 50 | 48 kHz |
| Local/in-process | Local Transformer v1.5 | 1.7 | 0.8 | 25 | 48 kHz |
| audio.cpp | Community Delay | 1.5 | 0.6 | 50 | 24 kHz |
| audio.cpp | Local Transformer | 1.7 | 0.8 | 25 | 48 kHz |

Only temperature, audio top-p, audio top-k, and seed are editable generation controls for the audio.cpp variants. Other native settings, including repetition penalty (Delay 1.1, Local 1.0), are left to the server and are not persisted app controls.

Community Delay parses request seeds with `std::stoi` before casting to `uint32_t`, so positive seeds above `2147483647` fail. Its spec uses the existing `max_random_seed = 2147483647` policy (as Echo does), applied by `Tts.generate_using_project()` before adapter dispatch. Caller caps can tighten but not widen this range; fixed seeds are not clamped or rewritten, so users must keep them within the server's accepted range. Local's seed range is unchanged.

The schema-1 [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) owns current persisted defaults, types and bindings, as well as server request defaults. [`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) resolves those bindings for menus and runtime consumers. Local architecture behavior remains in `MossConfigs`. [`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py) and [`model_settings_compat.py`](../tts_audiobook_tool/project_support/model_settings_compat.py) contain frozen history for migration only; flat names such as `moss_target` are legacy input, not current project attributes.

### Load compatibility

The load path explicitly converts old `shared.moss` and pre-v3 flat MOSS fields using frozen historical mappings. Seed and batch size go to the original three private types (`moss_local`, `moss_delay_sglomni`, `moss_local_sglomni`), both sampling sets go to `moss_local`, and only the relevant architecture's set goes to each SGL-Omni type. Target and rolling continuation remain local-only. These frozen non-voice mappings are not extended to audio.cpp: its types start with their own defaults.

Historically, v3 also copied voices/transcripts to the original three private types and respected each existing clone list atomically. V4 instead collects original legacy voice sources before that fan-out: one old `shared.moss` list counts once, while independently stored lists require a blocking choice even if identical. An explicit top-level `voice_references`, including `[]`, wins over stale lists; otherwise zero populated sources produce `[]`, one migrates silently, and two or more require a numbered choice of model names and all filenames. Every backend then consumes the selected shared project list according to its own capabilities.

Only missing private non-voice keys are filled: current raw keys win even when their value is `null`, a default or a sentinel. Conversion consumes `shared.moss` only after success; malformed groups remain non-destructively through existing load-error handling. Unknown whole objects for other models/groups remain preserved.

Successful on-disk migration now backs up the original settings before rewriting v4 immediately, without deleting audio; settings-snapshot imports require the same choice before destination writes. This supersedes historical v3 deferred-save timing without changing MOSS defaults/request behavior. See [Project Spec v4](<project-spec-v4.md>) for persistence logistics.

## Architecture-dependent behavior

The Local Transformer and Delay variants differ beyond sampling defaults:

- Delay output is 24 kHz; Local Transformer output is 48 kHz.
- Local Transformer is treated as capable of music hallucination.
- Local Transformer enables MOSS trailing token-noise trimming; Delay does not.

For local/in-process execution these decisions follow the private `target` setting or the loaded model. For SGL-Omni and audio.cpp they follow the selected server definition. Native audio.cpp Local output is stereo at 48 kHz and community Delay output is mono at 24 kHz; the existing WAV adapter downmixes returned audio to the app's mono `Sound` representation.

## Main implementation files

- [`tts_audiobook_tool/tts_models/model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml): sole v1 TOML source for built-in local/server/placeholder specs and MOSS server definitions; also supports validated additional server entries.
- [`tts_audiobook_tool/tts_models/tts_model_type.py`](../tts_audiobook_tool/tts_models/tts_model_type.py): canonical handle installation, strict/tolerant ID lookup, and model-ID matching entry point.
- [`moss_base_model.py`](../tts_audiobook_tool/tts_models/moss_base_model.py): architecture configs and local behavior.
- [`moss_model.py`](../tts_audiobook_tool/tts_models/moss_model.py): local/in-process implementation.
- [`tts_audiobook_tool/tts_models/sgl_omni_definition.py`](../tts_audiobook_tool/tts_models/sgl_omni_definition.py): server-definition validation from the shared catalog.
- [`sgl_omni_configured.py`](../tts_audiobook_tool/tts_models/sgl_omni_configured.py): common server support and adapter, with type-private MOSS state.
- [`tts.py`](../tts_audiobook_tool/tts.py): configured server runtime and local factories.
- [`voice_moss_shared.py`](../tts_audiobook_tool/menus/voice/voice_moss_shared.py): reusable local MOSS settings controls, not shared persistence.
- [`voice_configured_sgl_omni_menu.py`](../tts_audiobook_tool/menus/voice/voice_configured_sgl_omni_menu.py): configured server settings menu.
- [`sgl_omni_util.py`](../tts_audiobook_tool/app_support/sgl_omni_util.py): server model-ID and HTTP/audio utilities.
- [`audio_cpp_definition.py`](<../tts_audiobook_tool/tts_models/audio_cpp_definition.py>): validated audio.cpp definitions from the same catalog.
- [`audio_cpp_detection.py`](<../tts_audiobook_tool/tts_models/audio_cpp_detection.py>): metadata-based family/task/mode matching, independent of server IDs.
- [`audio_cpp_configured.py`](<../tts_audiobook_tool/tts_models/audio_cpp_configured.py>): lightweight support and sequential HTTP adapter, including MOSS language routing and Local music/trimming behavior.
- [`voice_audio_cpp_menu.py`](<../tts_audiobook_tool/menus/voice/voice_audio_cpp_menu.py>): definition-driven voice and sampling settings menu.
- [`audio_cpp_util.py`](<../tts_audiobook_tool/app_support/audio_cpp_util.py>): reference WAV encoding, HTTP transport, and response downmixing.
