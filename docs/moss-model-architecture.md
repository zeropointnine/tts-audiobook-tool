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

| Application backend | MOSS architecture | `TtsModelType` | Runtime implementation | Output rate |
|---|---|---|---|---:|
| Local/in-process | Delay or Local Transformer, selected by `project.moss_target` | `MOSS` | `MossModel` | 24 kHz or 48 kHz |
| SGL-Omni | Delay | `MOSS_DELAY_SERVER` | `SglOmniBackendAdapter` with the `server_moss_delay` TOML definition | 24 kHz |
| SGL-Omni | Local Transformer | `MOSS_LOCAL_SERVER` | `SglOmniBackendAdapter` with the `server_moss_local` TOML definition | 48 kHz |

The local/in-process catalog entry remains one `MOSS` type because its Hugging Face target is a project-level model setting and the loaded architecture is discovered from that target. SGL-Omni exposes the two architectures as formal model types because the server selection needs architecture-specific metadata even when no local model target is configured.

## Catalog identities and SGL-Omni selection

The single [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) (`schema_version = 3`) declares the built-in local `MOSS` spec, both built-in server variants, and the `NONE` placeholder alongside other model specs. It supplies stable IDs/handles, backend kinds and metadata; server entries also carry model-ID matching, behavior, request defaults, parameters and menu definitions. The MOSS server handles are:

- `MOSS_DELAY_SERVER`, serialized as `server_moss_delay`
- `MOSS_LOCAL_SERVER`, serialized as `server_moss_local`

Both have `TtsBackendKind.SGL_OMNI`, share the same registry-declared MOSS voice/project storage fields, and use the `moss` file tag. Their output sample rates and UI identities differ. [`tts_model_type.py`](../tts_audiobook_tool/tts_models/tts_model_type.py) installs stable built-in handles from the catalog rather than maintaining a second set of built-in specs.

### Auto-detect

When the SGL-Omni preference is Auto, `Tts.update_tts_type()` fetches the served model ID and calls `TtsModelType.find_tts_type_using_sgl_omni_model_id()`.

Matching uses case-insensitive substrings and prefers the longest match:

- Local Transformer uses the specific `moss-tts-local` substring.
- Delay uses the generic `moss` fallback.

For example:

- `OpenMOSS-Team/MOSS-TTS-v1.5` resolves to `MOSS_DELAY_SERVER`.
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer` contains both matches, but the more specific Local match wins.

An unrecognized non-MOSS model ID does not resolve to either MOSS type.

### Explicit selection

The user can explicitly choose Delay or Local in the SGL-Omni model-type menu. An explicit selection is authoritative: generation settings, output metadata, and architecture-dependent behavior come from the selected type and do not probe the model ID again.

The UI still displays the server model ID as diagnostics. A user who explicitly selects a type that does not match the model actually served has made a configuration error; the runtime does not silently change the explicit choice.

## Legacy `server_moss` migration

Older versions persisted one architecture-ambiguous ID, `server_moss`.

It cannot safely be renamed directly to either new type:

- generation used Delay hyperparameter fields and defaults;
- sample rate and some architecture behavior were inferred from the server model ID and could therefore be Local.

Compatibility is context-sensitive:

- A legacy `prefs.json` value of `server_moss` migrates to Auto (`sgl_omni_type = None`). This preserves model-ID-based architecture selection and is rewritten as the normal empty Auto value on a normal preference load/save.
- A legacy project `current_model_type` value of `server_moss` migrates to `NONE`. The project stamp is historical metadata, and there is no reliable way to reconstruct which architecture was previously served. Treating it as unknown avoids a false model-change warning.

New explicit selections and new project stamps use the two unambiguous IDs.

## Local model and configured server runtime

The in-process class hierarchy is `TtsBaseModel` → `MossBaseModel` → `MossModel`. `MossBaseModel` retains the MOSS configs, language-name mapping, local sampling and architecture behavior, and the local batch/rolling-continuation readiness rule. `MossConfigs.get_by_target()` identifies the architecture of a local target; the local `MossModel` loads it and runs inference.

Server generation does **not** subclass `MossBaseModel`. Each server variant's behavior and request definition lives alongside its spec in the v3 [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml), selected by its stable catalog ID. In SGL-Omni mode the complete catalog is validated before server specs are overlaid; optional validated server entries may add supported IDs with private registry settings, while built-in MOSS bindings stay fixed. `ConfiguredModelSupport` provides metadata, readiness, output rate, music/trimming decisions and menu settings; one `SglOmniBackendAdapter` constructs the `/speech` request and calls `SglOmniUtil.generate_concurrent()`. Local MOSS inference still uses `MossModel`.

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

`moss_rolling_cont` applies only to local/in-process generation; the SGL-Omni request implementation does not perform local rolling continuation. `moss_target` likewise selects the model only for local/in-process execution. Neither field selects the SGL-Omni architecture; the active server `TtsModelType` does that.

The configured server settings menu uses the selected catalog definition and its registry-declared storage bindings, so it cannot preview one architecture while editing the other architecture’s fields. [`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) owns setting types, defaults and shared/private ownership; the built-in declarations and old flat project-field mappings remain in [`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py), not in the TOML catalog. The flat names above are legacy input mappings, not new top-level project fields.

## Architecture-dependent behavior

The Local Transformer and Delay variants differ beyond sampling defaults:

- Delay output is 24 kHz; Local Transformer output is 48 kHz.
- Local Transformer is treated as capable of music hallucination.
- Local Transformer enables MOSS trailing token-noise trimming; Delay does not.

For local/in-process execution these decisions follow `moss_target` or the loaded model. For SGL-Omni execution they follow the selected server definition.

## Main implementation files

- [`tts_audiobook_tool/tts_models/model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml): sole v3 TOML source for built-in local/server/`NONE` specs and MOSS server definitions; also supports validated additional server entries.
- [`tts_audiobook_tool/tts_models/tts_model_type.py`](../tts_audiobook_tool/tts_models/tts_model_type.py): stable model handles derived from the catalog and model-ID matching.
- `tts_audiobook_tool/tts_models/moss_base_model.py`: architecture configs and local behavior.
- `tts_audiobook_tool/tts_models/moss_model.py`: local/in-process implementation.
- [`tts_audiobook_tool/tts_models/sgl_omni_definition.py`](../tts_audiobook_tool/tts_models/sgl_omni_definition.py): server-definition validation from the shared catalog.
- `tts_audiobook_tool/tts_models/sgl_omni_configured.py`: shared server support and adapter.
- `tts_audiobook_tool/tts.py`: configured server runtime and local factories.
- `tts_audiobook_tool/menus/voice/voice_moss_shared.py`: common local MOSS settings controls.
- `tts_audiobook_tool/menus/voice/voice_configured_sgl_omni_menu.py`: configured server settings menu.
- `tts_audiobook_tool/app_support/sgl_omni_util.py`: server model-ID and HTTP/audio utilities.
