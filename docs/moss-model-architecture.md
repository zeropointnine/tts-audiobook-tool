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
| SGL-Omni | Delay | `MOSS_DELAY_SERVER` | `MossDelayServerModel` using the shared `MossServerModel` request implementation | 24 kHz |
| SGL-Omni | Local Transformer | `MOSS_LOCAL_SERVER` | `MossLocalServerModel` using the shared `MossServerModel` request implementation | 48 kHz |

The local/in-process catalog entry remains one `MOSS` type because its Hugging Face target is a project-level model setting and the loaded architecture is discovered from that target. SGL-Omni exposes the two architectures as formal model types because the server selection needs architecture-specific metadata even when no local model target is configured.

## Catalog identities and SGL-Omni selection

The server variants are defined in `tts_models/tts_model_type.py`:

- `MOSS_DELAY_SERVER`, serialized as `server_moss_delay`
- `MOSS_LOCAL_SERVER`, serialized as `server_moss_local`

Both have `TtsBackendKind.SGL_OMNI`, share the same MOSS voice/project storage fields, and use the `moss` file tag. Their output sample rates and UI identities differ.

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

## Class hierarchy

The relevant hierarchy is:

```text
TtsBaseModel
└── MossBaseModel
    ├── MossModel
    └── MossServerBaseModel
        └── MossServerModel
            ├── MossDelayServerModel
            └── MossLocalServerModel
```

### `MossBaseModel`

`MossBaseModel` contains behavior shared by local and server execution:

- supported language-name mapping;
- MOSS architecture/config definitions (`MossConfigs` and `MossConfig`);
- sampling-parameter resolution from project fields and architecture defaults;
- the local-generation readiness rule for the batch-size/rolling-continuation incompatibility;
- architecture-related interfaces such as output sample rate, trailing-noise handling, and music-hallucination capability.

`MossConfigs.get_by_target()` identifies a local model target as Local Transformer when its target identifies the Local model; otherwise it conservatively uses Delay.

### `MossModel`

`MossModel` is the in-process implementation. It loads the configured Hugging Face target through the MOSS library and determines its architecture from `project.moss_target`/the loaded model.

At generation time it resolves `MossConfigs.DELAY` or `MossConfigs.LOCAL`, selects the corresponding project hyperparameters, and calls the local MOSS generation code.

### `MossServerBaseModel`

`MossServerBaseModel` adds behavior shared by all SGL-Omni MOSS variants:

- SGL-Omni readiness checks;
- fixed architecture-derived sample rate and behavior through a subclass `CONFIG`;
- server-oriented model display behavior.

Its readiness override checks server connectivity. It intentionally does not inherit the local batch/rolling-continuation blocker because the server request implementation does not use local rolling continuation.

### `MossServerModel`

`MossServerModel` implements the common SGL-Omni `/speech` request:

- prepares prompt text and language metadata;
- resolves the architecture-specific sampling values;
- supplies seed and `max_new_tokens`;
- sends voice clone audio as a base64 data URI when configured;
- invokes `SglOmniUtil.generate_concurrent()`.

The SGL-Omni API payload is the same for Delay and Local Transformer. There is intentionally one request implementation rather than duplicated HTTP model classes.

`MossDelayServerModel` and `MossLocalServerModel` are thin concrete subclasses. Each fixes:

- `INFO` to its formal `TtsModelType` specification;
- `CONFIG` to `MossConfigs.DELAY` or `MossConfigs.LOCAL`.

This allows generic app code to use class metadata without querying the server model ID.

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

The server settings menu derives `MossConfigs` from the selected server model type, so it cannot preview one architecture while editing the other architecture’s fields.

## Architecture-dependent behavior

The Local Transformer and Delay variants differ beyond sampling defaults:

- Delay output is 24 kHz; Local Transformer output is 48 kHz.
- Local Transformer is treated as capable of music hallucination.
- Local Transformer enables MOSS trailing token-noise trimming; Delay does not.

For local/in-process execution these decisions follow `moss_target` or the loaded model. For SGL-Omni execution they follow the fixed concrete server variant.

## Main implementation files

- `tts_audiobook_tool/tts_models/tts_model_type.py`: formal catalog entries and model-ID matching.
- `tts_audiobook_tool/tts_models/moss_base_model.py`: architecture configs and shared behavior.
- `tts_audiobook_tool/tts_models/moss_model.py`: local/in-process implementation.
- `tts_audiobook_tool/tts_models/moss_server_base_model.py`: shared SGL-Omni model behavior.
- `tts_audiobook_tool/tts_models/moss_server_model.py`: shared request implementation and concrete Delay/Local server classes.
- `tts_audiobook_tool/tts.py`: factories and registry entries.
- `tts_audiobook_tool/menus/voice/voice_moss_shared.py`: common MOSS settings controls.
- `tts_audiobook_tool/menus/voice/voice_moss_server_menu.py`: variant-aware server settings menu.
- `tts_audiobook_tool/app_support/sgl_omni_util.py`: server model-ID and HTTP/audio utilities.
