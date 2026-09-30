# MOSS-TTS model and backend architecture

## Terminology

MOSS has two independent axes that both use the word “local” in different ways:

1. **Application backend**
   - `TtsBackendKind.LOCAL`: the audiobook tool imports and runs `moss_tts` in its own model worker, normally from `venv-moss`.
   - `TtsBackendKind.SGL_OMNI`: the audiobook tool sends HTTP requests to an external SGL-Omni server, normally while running from `venv-sgl-omni`.

2. **MOSS model architecture**
   - **Delay**: upstream `MossTTSDelay`, represented by `MossConfigs.DELAY`.
   - **Local Transformer**: upstream `MossTTSLocal`, represented by `MossConfigs.LOCAL`.

“MOSS Local” therefore means the upstream Local Transformer architecture. It does **not** necessarily mean in-process execution. The Local Transformer can be loaded directly by the app or served remotely through SGL-Omni.

## Supported combinations

| Application backend | MOSS architecture | Catalog ID | Runtime implementation | Output rate |
|---|---|---|---|---:|
| Local/in-process | Delay or Local Transformer, selected by `project.moss_target` | `moss_local` | `MossModel` | 24 kHz or 48 kHz |
| SGL-Omni | Delay | `moss_delay_sglomni` | `SglOmniBackendAdapter` with the `moss_delay_sglomni` TOML definition | 24 kHz |
| SGL-Omni | Local Transformer | `moss_local_sglomni` | `SglOmniBackendAdapter` with the `moss_local_sglomni` TOML definition | 48 kHz |

The local/in-process catalog entry remains one `moss_local` type because its Hugging Face target is a project-level model setting and the loaded architecture is discovered from that target. SGL-Omni exposes the two architectures as formal model types because the server selection needs architecture-specific metadata even when no local model target is configured.

## Catalog identities and SGL-Omni selection

The single [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) (`schema_version = 1`) declares the built-in local `moss_local` spec, both built-in server variants, and the `"none"` placeholder alongside other model specs. It supplies stable IDs, backend kinds and metadata; server entries also carry model-ID matching, behavior, request defaults, parameters and menu definitions. The MOSS server catalog IDs are:

- `moss_delay_sglomni`
- `moss_local_sglomni`

Both have `TtsBackendKind.SGL_OMNI`, share the same registry-declared MOSS voice/project storage fields, and use the `moss` file tag. Their output sample rates and UI identities differ. [`tts_model_type.py`](<../tts_audiobook_tool/tts_models/tts_model_type.py>) installs canonical handles for every shipped entry in exact TOML order, including data-only models, without model-named class attributes or a Python declaration inventory. Use `TtsModelType.require_by_id("moss_delay_sglomni")` for a hardcoded ID (unknown IDs raise `ValueError`); use `TtsModelType.get_by_id(saved_id)` for persisted/external IDs (unknown IDs return the registered `"none"` handle, leaving the original string untouched). Both lookups return the same handle for a known ID; metadata is accessed through `.value`.

### Detection and matching

Remote discovery ([`remote_tts_discovery.py`](../tts_audiobook_tool/app_support/remote_tts_discovery.py)) queries the configured server and matches each served model ID to catalog types through [`sgl_omni_detection.py`](../tts_audiobook_tool/tts_models/sgl_omni_detection.py) using the catalog's `sgl_omni.match.model_id_substring`. Matching is case-insensitive and prefers the longest match:

- Local Transformer uses the specific `moss-tts-local` substring.
- Delay uses the generic `moss` fallback.

For example:

- `OpenMOSS-Team/MOSS-TTS-v1.5` resolves to `moss_delay_sglomni`.
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer` contains both matches, but the more specific Local match wins.

An unrecognized non-MOSS model ID does not resolve to either MOSS type. The compatibility helper `TtsModelType.find_tts_type_using_sgl_omni_model_id()` delegates to the same detector.

### Selection and binding

The project stores a catalog ID; refer to [TTS model selection rules](<tts-model-selection.md>) for the canonical reconciliation and binding rules. For a remote selection, `Tts.bind_project()` requires exactly one server entry matching the selected type: zero matches, an unreachable server, or several matches block binding and leave the saved ID untouched.

Generation settings, output metadata, and architecture-dependent behavior come from the selected catalog definition (`moss_delay_sglomni` or `moss_local_sglomni`); the runtime never probes the served model ID a second time. The status UI still shows the exact server model ID as diagnostics, so a selection that disagrees with what the server serves is visible as a configuration error rather than silently overridden.

### Pre-release IDs

Older pre-release builds persisted architecture-ambiguous IDs such as `server_moss`, plus a separate `sgl_omni_type` preference. Both are gone: those builds were never deployed, so no migration is provided. The shipped MOSS IDs are `moss_local`, `moss_delay_sglomni`, and `moss_local_sglomni`.

## Local model and configured server runtime

The in-process class hierarchy is `TtsBaseModel` → `MossBaseModel` → `MossModel`. `MossBaseModel` retains the MOSS configs, language-name mapping, local sampling and architecture behavior, and the local batch/rolling-continuation readiness rule. `MossConfigs.get_by_target()` identifies the architecture of a local target; the local `MossModel` loads it and runs inference.

Server generation does **not** subclass `MossBaseModel`. Each server variant's behavior and request definition lives alongside its spec in the v2 [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml), selected by its stable catalog ID. In SGL-Omni mode the complete catalog is validated before server specs are overlaid; optional validated server entries may add supported IDs with private registry settings, while built-in MOSS bindings stay fixed. `ConfiguredModelSupport` provides metadata, readiness, output rate, music/trimming decisions and menu settings; one `SglOmniBackendAdapter` constructs the `/speech` request and calls `SglOmniUtil.generate_concurrent()`. Local MOSS inference still uses `MossModel`.

The Delay and Local definitions use their own sampling parameter keys and defaults, the existing shared MOSS voice/seed ownership, a resolved seed and a fixed token limit. They do not inherit the local batch/rolling-continuation blocker. Architecture-dependent server behavior follows the selected definition, never a second probe of the server model ID.

## Project settings

No additional project hyperparameter fields are required. Both architectures already have separate settings:

| Setting | Delay field | Local Transformer field |
|---|---|---|
| Temperature | `moss_delay_temperature` | `moss_local_temperature` |
| Audio top-p | `moss_delay_top_p` | `moss_local_top_p` |
| Audio top-k | `moss_delay_top_k` | `moss_local_top_k` |

A value of `-1` means “use the architecture default.” The canonical defaults and supported bounds live in `MossConfigs`:

| Architecture | Temperature default | Audio top-p default | Audio top-k default | Output rate |
|---|---:|---:|---:|---:|
| Delay | 1.7 | 0.8 | 25 | 24 kHz |
| Local Transformer | 1.0 | 0.95 | 50 | 48 kHz |

The following project storage is shared between architectures and is used by both local and server generation:

- `moss_voice_file_name`
- `moss_voice_transcript`
- `moss_seed`
- `moss_batch_size`

`moss_rolling_cont` applies only to local/in-process generation; the SGL-Omni request implementation does not perform local rolling continuation. `moss_target` likewise selects the model only for local/in-process execution. Neither field selects the SGL-Omni architecture; the project's server catalog ID does that.

The configured server settings menu uses the selected catalog definition and its registry-declared storage bindings, so it cannot preview one architecture while editing the other architecture’s fields. [`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) owns setting types, defaults and shared/private ownership; the built-in declarations and old flat project-field mappings remain in [`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py), not in the TOML catalog. The flat names above are legacy input mappings, not new top-level project fields.

## Architecture-dependent behavior

The Local Transformer and Delay variants differ beyond sampling defaults:

- Delay output is 24 kHz; Local Transformer output is 48 kHz.
- Local Transformer is treated as capable of music hallucination.
- Local Transformer enables MOSS trailing token-noise trimming; Delay does not.

For local/in-process execution these decisions follow `moss_target` or the loaded model. For SGL-Omni execution they follow the selected server definition.

## Main implementation files

- [`tts_audiobook_tool/tts_models/model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml): sole v1 TOML source for built-in local/server/placeholder specs and MOSS server definitions; also supports validated additional server entries.
- [`tts_audiobook_tool/tts_models/tts_model_type.py`](../tts_audiobook_tool/tts_models/tts_model_type.py): canonical handle installation, strict/tolerant ID lookup, and model-ID matching entry point.
- `tts_audiobook_tool/tts_models/moss_base_model.py`: architecture configs and local behavior.
- `tts_audiobook_tool/tts_models/moss_model.py`: local/in-process implementation.
- [`tts_audiobook_tool/tts_models/sgl_omni_definition.py`](../tts_audiobook_tool/tts_models/sgl_omni_definition.py): server-definition validation from the shared catalog.
- `tts_audiobook_tool/tts_models/sgl_omni_configured.py`: shared server support and adapter.
- `tts_audiobook_tool/tts.py`: configured server runtime and local factories.
- `tts_audiobook_tool/menus/voice/voice_moss_shared.py`: common local MOSS settings controls.
- `tts_audiobook_tool/menus/voice/voice_configured_sgl_omni_menu.py`: configured server settings menu.
- `tts_audiobook_tool/app_support/sgl_omni_util.py`: server model-ID and HTTP/audio utilities.
