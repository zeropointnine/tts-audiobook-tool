# Project Spec v5

## Scope and version boundaries

Project spec v5 is the **on-disk project settings format**. It keeps v4's one top-level, ordered `voice_references` list shared by all models **within a project** (not an application-global voice library). V5's defining change is that each entry is a formal **voice** (`Voice` in [`app_types/voice.py`](../tts_audiobook_tool/app_types/voice.py)) carrying per-voice settings: the project-wide `room_reverb` setting moved onto each voice, and crop times are stored as numbers. Model-specific generation settings remain in backend-neutral objects under `model_settings`, and the split between settings and book text remains in place. This applies to local inference, SGL-Omni, and audio.cpp alike.

Historically, v3 moved flat model fields into private/shared model-setting objects, including model-scoped clone lists. V4 retained that organization for non-voice settings but replaced those clone lists with the project-wide list of `{file_name, transcript}` pairs. V1, v2, v3, and v4 remain accepted migration inputs.

Independent version numbers should not be conflated:

- `project.json` has `"version": 5` (`PROJECT_SPEC_VERSION` in [`constants.py`](../tts_audiobook_tool/constants.py)).
- `project_text.json` still uses `"format": "book.v2"` ([`book_serialization.py`](../tts_audiobook_tool/app_types/book_serialization.py)). Neither project spec v4 nor v5 introduces a new book format.
- The [ABR metadata format](<abr-metadata-spec.md>) is unchanged; its version is independent of the project settings version.
- The application-shipped [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) uses `schema_version = 1`. This is the model-catalog schema, not a project-file version. The loader rejects other catalog schema versions; it does not migrate them. This one TOML file declares the built-in local, SGL-Omni and audio.cpp server, and `"none"` placeholder specs, plus server-specific definitions and optional additional configured server entries. Every shipped entry receives a canonical Python handle in TOML order, looked up by its catalog ID; no model-named attributes or Python declaration inventory is needed. `TtsModelType.require_by_id("fish_s2_local")` is strict for hardcoded IDs and raises `ValueError` for an unknown ID. `TtsModelType.get_by_id(saved_id)` is tolerant for persisted/external IDs and falls back to the registered `"none"` handle without changing the saved string.

The canonical project directory has:

| Artifact | Responsibility |
| --- | --- |
| `project.json` | Project-wide settings and ordered `voice_references` (voices), selected model ID, and non-voice `model_settings` overrides. |
| `project_text.json` | Structured `book.v2` text, sections, phrase groups, and their `voice_index` values. |
| Optional `project_text_raw.txt`, `project_text.epub`, `voice/`, generated audio | Source material, voice clips, and outputs; these are not inline model-setting objects. |

See [`Project`](../tts_audiobook_tool/project.py), [`ProjectSerializationUtil`](../tts_audiobook_tool/project_support/project_serialization_util.py), and [`ProjectTextIOUtil`](../tts_audiobook_tool/project_support/project_text_io_util.py) for the respective in-memory, settings-file, and text-file boundaries.

The selected catalog model is `Project.tts_model_type`. The additive `audio_cpp_model_id` string stores an exact audio.cpp entry alongside it (default `""` for older projects and other backends); save/export/import/copy preserve opaque values, even when unavailable. It is not a model-setting owner or checkpoint fingerprint, and no endpoint identity is stored. For startup, reconciliation and binding policy, refer to [TTS model selection rules](<tts-model-selection.md>).

## Canonical settings shape

The following is illustrative: absent override keys are normal, and these objects are not a complete listing of project-wide settings.

```json
{
  "version": 5,
  "language_code": "en",
  "voice_select_mode": "custom",
  "room_reverb_stereo": "late_w100",
  "voice_references": [
    {"file_name": "sample.flac", "transcript": "Reference text.", "room_reverb": "small_drr22", "pan": -0.3},
    {"file_name": "second.flac", "transcript": "",
     "crop_file_name": "0f3c…e1.flac", "crop_start": 1.5, "crop_end": 9.0, "crop_transcript": "Trimmed text.",
     "room_reverb": "off"},
    {"file_name": "third.flac", "transcript": "", "room_reverb": "off",
     "presets": {"pocket_local": "alba", "qwen3tts_local": "Ryan"},
     "instructions": {"qwen3tts_local": "Speak warmly, with a slight rasp."}}
  ],
  "model_settings": {
    "models": {
      "fish_s2_sglomni": {
        "parameters": {"top_k": 50},
        "orchestration": {"concurrent_requests": 2}
      },
      "higgs_v3_sglomni": {
        "parameters": {"temperature": 0.8},
        "orchestration": {"batch_size": 2}
      }
    },
    "shared": {
      "auk": {
        "model_ids": ["auk_sglomni", "auk_flash_sglomni"],
        "parameters": {"speed": 1.2}
      }
    }
  }
}
```

Top-level `voice_references` is the sole authoritative clone list and is always serialized. Each entry is one voice: a project-local voice filename with its transcript, in order; an unused/absent transcript is represented by `""`. Output does not store authoritative clone lists under private models or shared groups.

**A project always has at least one voice.** An empty list (absent, `[]`, or migrated from legacy data with no populated source) loads as one default voice (`Voice()`: no sample, reverb off, centered), and removing the last voice leaves a fresh default voice. A voice's sample is optional: `file_name: ""` is a voice without a sample, kept in place so later voices keep their indexes. When no voice has a sample, no clone reference is sent; per-voice effects (pan, room reverb) still apply. A voice without a sample generates with the model's built-in voice where supported; for a model that requires a clone sample it is a pre-flight error only if that voice is needed by the job. Unused sample-less voices never block a job. Builds that predate sample-less voices drop `file_name: ""` entries on load (shifting later voice indexes).

### Voice entries

In memory the list is `list[Voice]`. `Voice` and its optional `VoiceCrop` are frozen dataclasses (edit with `dataclasses.replace`); `Voice.to_dict()` writes the flat on-disk form and `Voice.from_dict()` parses it. Direct assignment to `Project.voice_references` is not validated, so callers assign `Voice` objects.

| Key | In memory | Notes |
| --- | --- | --- |
| `file_name` | `Voice.file_name` | Project-local sample path; `""` means the voice has no sample (`Voice.has_sample`). A non-empty unusable path drops the entry with a warning. |
| `transcript` | `Voice.transcript` | `""` when absent. |
| `crop_file_name`, `crop_start`, `crop_end`, `crop_transcript` | `Voice.crop: VoiceCrop \| None` | All or none; see below. |
| `room_reverb` | `Voice.room_reverb: RoomReverb` | `RoomReverb` id. **Always written**, including `"off"`. Missing loads as off (or, in v4 input, the migrated project value); an unknown id loads as off. New voices default to off. |
| `pan` | `Voice.pan: float` | Stereo position, -1 (full left) to 1 (full right). **Written only when nonzero**. Missing or malformed (non-numeric, non-finite) loads as 0 (centered); out-of-range values are clamped. Optional, so no version bump. |
| `presets` | `Voice.presets: dict[str, str]` | Preset voice ids for models with built-in voices (Pocket TTS predefined voice, Qwen3-TTS CustomVoice speaker), keyed by TTS model id. **Written only when non-empty**. Missing or non-object loads as `{}`; entries with a non-string key or value are dropped. A former single-string `preset` key is ignored. Optional, so no version bump. |
| `instructions` | `Voice.instructions: dict[str, str]` | Free-text voice instructions for "instruct" models (e.g. Qwen3-TTS Instruct), keyed by TTS model id. **Written only when non-empty**. Missing or non-object (including a legacy plain string, which is not migrated) loads as `{}`; entries with a non-string key or value are dropped. Optional, so no version bump. |

**Instructions** are owned only by `Voice.instructions`, keyed by the exact TTS model ID. They are not model settings, and old model-global instruction values are not migrated or used as a fallback. Local OmniVoice, local Breeze TTS 2, local Qwen3-TTS CustomVoice/VoiceDesign, audio.cpp OmniVoice, Breeze TTS 2 and CosyVoice3, and SGL-Omni CosyVoice3, AuK and AuK-Flash consume the selected voice's value. AuK and AuK-Flash use it only while the voice has no voice sample (as its voice description); the submenu still offers it on a voice with a sample, marked as unused, and the voice list previews it only when used. CosyVoice3 (either backend, each under its own model ID) uses it only in Instruct mode; its Mode choice remains model-wide. SGL-Omni CosyVoice3 stores that Mode as `template_name` (default `zero_shot`, so existing projects are unchanged); it is never sent to the server. The voice submenu shows `Instructions/{model short name}` (or the catalog's optional `spec.ui.instructions_label` in place of `Instructions`) when the runtime `can_instructions(project)` is true, followed by Clear instructions only when this model's value is present. Editing preserves case, strips surrounding whitespace, and rejects more than 4,000 characters without truncation; blank/cancel leaves the value unchanged. Clear removes only this model's key. Failed saves restore the previous voice list. Catalog `spec.ui.instructions_message` supplies model-specific prompts. Local OmniVoice validates entered text with OmniVoice's own instruct resolver and rejects unsupported items without saving; the stored value is the text as entered, not the normalized form.

**Presets** are owned only by `Voice.presets`, keyed by the exact TTS model ID, so a voice can hold a Pocket predefined voice and a Qwen3-TTS speaker side by side. They replace the retired model-global settings `pocket_local.parameters.predefined_voice` and `qwen3tts_local.parameters.speaker_id` (and the pre-v3 flat `pocket_predefined_voice` / `qwen3_speaker_id`), which are dropped on load without migration. A model uses them while its runtime `can_voice_presets(project)` is true (Pocket always; Qwen3-TTS for CustomVoice checkpoints). A voice's preset, when set, takes precedence over its voice sample; both may be stored at once; the voice list then shows just the preset, and the submenu notes the precedence. Setting or clearing a sample leaves the preset unchanged, and vice versa. Qwen3-TTS CustomVoice requires a speaker on each voice a job uses (`requires_voice_preset`; a voice pre-flight problem, not project-wide readiness); with a loaded checkpoint, readiness also flags stored speakers the checkpoint lacks. A single-speaker checkpoint always uses its only speaker.

**Room reverb** is per voice; `room_reverb_stereo` (the stereo effect) remains a project-wide setting. Concat and realtime playback put each segment in its voice's room: the phrase group's `voice_index` in user-defined voice selection, otherwise voice 0. Voice chat uses voice 0's. Both are set in the voice sample submenu (Effects > Stereo position, Room reverb).

**Stereo position** (`pan`) is per voice and applies to concat and realtime playback only (not voice chat): user-defined voice selection uses the phrase group's voice, Disabled uses voice 0's, Auto-advance stays centered. Any heard off-center voice makes the output stereo. See [room reverb](<room-reverb.md>).

**Crops.** A finite crop range must satisfy `end > start >= 0`; v5 stores the times as JSON numbers (v4 stored decimal strings, which still load). The explicitly owned `crop_file_name` stores only a basename (`<uuid>.flac`); validation rejects paths, absolute names, and traversal components. Its disk path remains `voice/crops/<basename>`. `VoiceCrop.relative_path` (also `ProjectVoiceUtil.get_cropped_voice_relative_path(voice)`) derives `crops/<basename>` for the generation interface and transfer supporting-file manifest; `resolve_cropped_voice_file_path(project, voice)` and `resolve_crop_file_path(project, basename)` centralize full-path resolution. A new crop reserves a unique file, and re-editing normally retains that filename. Without a valid explicit basename or range, all crop metadata is dropped with a warning (the voice is kept), including old development entries storing paths such as `crops/<uuid>.flac`. The feature has not shipped, so there is no migration or `_crop` sibling-name fallback. Original filenames/transcripts and old files remain untouched. Project transfers copy both originals and active crop files, preserve nested paths, and report missing crops. See [voice sample crop architecture](<voice-sample-crop.md>).

`model_settings.models` is keyed by stable **model ID**; `model_settings.shared` is keyed by a declared **storage group**, not a selectable model. Inside a recognized object, `parameters` holds named overrides, `files` holds declared additional project-local file strings, and `orchestration` holds batch/concurrency values. Sections appear only when needed. The same model may read some non-voice settings from a shared object and others from a private object; there is no duplicate authoritative copy.

`min_segment_sentence_silence` (`MinSegmentSentenceSilence` in `app_types`; JSON ids `"disabled"` (default) / `"enabled"`, each carrying a duration in seconds) governs whether concatenation enforces a minimum silence at sentence boundaries inside segments, using the segment sidecar's per-phrase `reason` (see [validation architecture](<tts-validation-architecture.md>)). It is an optional key with a default, so adding it needed **no project version bump**: a missing or unrecognized value loads silently as `disabled`, and nothing in the migration path keys off it.

Top-level values such as `language_code`, `markers`, `generate_range`, `voice_select_mode`, export options, and the compatibility placeholder `none_voice_file_name` remain project-wide. A project does not record which TTS model produced its audio: remote servers expose several models at once, so "the model in use" is not a stable property to compare against. `project_text.json` remains separately serialized as a structured `Book`; `voice_index` belongs to each phrase group there, whereas the voice samples it selects are the project-wide ordered pairs here. For selection modes and scheduling, see [`generate-files-ordering.md`](generate-files-ordering.md).

All voice consumers use this same ordered list: menus and editing, generation and rotation, per-line/dialog assignment, local and server adapters, workers, and clone/transfer/snapshot paths. A transcript is collected for every added primary sample — sidecar text file or STT — regardless of the selected model, so the shared list is complete for any model. Before a full-screen TTS feature launches (generation, chat, realtime playback), the interactive app pre-flights the job's voice dependencies: Disabled checks voice 1, User-defined checks the distinct clamped assignments of the actual submitted lines (or playback range), and Auto-advance checks all voices, regardless of job length, batching or retries. Chat and literal previews have no per-line assignment, so User-defined uses voice 1 there. `ProjectVoiceUtil.get_job_voice_indices()` resolves that scope without filtering/reordering the shared list or advancing rotation; `get_effective_voice_index()` shares the assignment clamp with inference and batch scheduling. Within that scope a missing required sample, missing or undecodable active file, or unresolvable required transcript blocks launch. Empty required transcripts are auto-transcribed only for selected voices; unused voice files/transcripts are neither checked nor repaired. The console-free quick-job pre-flight uses the same scope but does not auto-transcribe. Both validator APIs accept `voice_indices=None` to check the whole list and an explicit empty iterable to check none. There is no load-time and no readiness-based voice-file verification: missing or undecodable references survive loading untouched, readiness deliberately ignores voice state, and model adapters surface voice problems only at generation time. Transcripts stay paired with their filenames during append, replace, reorder, and clear operations. They remain stored and editable even while a model that does not use reference transcripts is selected; that model simply does not send/use them. Clone support and transcript requirements remain model-specific capabilities, not a reason to create separate storage.

Pocket's selected predefined voice takes precedence during generation without clearing the shared list. IndexTTS2's secondary emotion reference remains a model-specific file setting, not a second project clone list.

## Ownership and interpretation

[`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) resolves `(model ID, setting name) → section, private model or shared group, type and storage default` from the current catalog. `models.settings` declares persisted defaults/types/owners independently of request defaults; the root `setting_groups` table declares exact shared membership. Remote backend parameters, seed policies and `behavior.orchestration` supply standard private declarations when no explicit override is needed. Voice/transcript bindings remain compatibility/capability metadata; project accessors route primary clone values to the project-wide list rather than private/shared storage. Every model follows this same registration rule, including data-only entries available through canonical ID lookup; no Python model declaration is required. Projects cannot create sharing relationships by editing membership or using matching setting names.

[`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py) and [`model_settings_compat.py`](../tts_audiobook_tool/project_support/model_settings_compat.py) freeze pre-v3 input fields, defaults and storage mappings solely for migration. They do not initialize current settings.

The only built-in shared group is `auk` (AuK/Flash servers); its non-voice ownership is unchanged. Sharing is **per field**, not per model family: a group owns only the settings whose declarations carry its `group` key, and every other setting of a member stays private. Primary voice references are now shared across all models in the project, independently of these groups. A model switch never copies or clamps a shared override. A server variant may resolve a different effective default, cap an outgoing value, omit a request field, or warn about a shared value without modifying what the local variant will see.

Retired groups are listed in `RETIRED_GROUPS` in [`model_settings_compat.py`](../tts_audiobook_tool/project_support/model_settings_compat.py): currently `moss`, `fish_s2` and `qwen3`. Fish S2's local (`fish_s2_local`) and server (`fish_s2_sglomni`) types each privately own `temperature`, `top_p` and `top_k`; only the local type owns `seed`, since the server never sends one. Server Fish S2 still caps outgoing top-k at 30 without rewriting its stored value. Qwen3's local (`qwen3tts_local`) and server (`qwen3tts_sglomni`) types each privately own `temperature`, `top_k`, `top_p`, `repetition_penalty` and `seed`, so the old shared values, seed included, are copied to both. Server Qwen3 sends its seed as SGL-Omni's request-scoped `seed`. To retire another group, remove it from the catalog's `setting_groups` and its settings' `group` keys, append it to `RETIRED_GROUPS`, and, if a local member has no catalog parameter bounds of its own, map its server counterpart in `BORROWED_PARAMETER_BOUNDS` ([`model_settings.py`](../tts_audiobook_tool/project_support/model_settings.py)) so saved values keep being validated.

MOSS has no current shared model-setting group. The unchanged IDs `moss_local`, `moss_delay_sglomni`, and `moss_local_sglomni` each privately own seed and batch size, but use the same project-wide voice references as every other model. Local MOSS owns both Delay and Local Transformer sampling sets plus local-only target/rolling continuation; each remote type owns only its architecture's sampling set. Common controls/adapters reuse code, not private generation state. Historically, the MOSS-only ownership fork happened within project v3 without changing catalog schema 1, defaults, request semantics or backends; v4 separately changes primary voice storage. See [MOSS architecture](<moss-model-architecture.md>).

A missing override means “use the declaration's default”; a stored value equal to that default is still an explicit override. `Project.get_model_setting(model_id, name)` reads through the registry, while `set_model_setting(..., reset=True)` removes an override. Some older numeric controls use `-1` as a *default* sentinel and reset to absence; seed `-1` can instead mean *random* and remains an explicit value. There is no global interpretation of `-1`. Variant-specific effective defaults, validation, menu presentation, and HTTP request policy are separate from storage ownership. Menus and other callers refer to a setting by `SettingRef(model_id, name)` rather than a flat Python attribute name.

Recognized objects are reconciled against the declarations: unsupported fields inside them are pruned and declared values are checked; the top-level voice-reference pairs are validated separately, and a recognized shared group's membership must match its authorized members. Unusable numeric parameter overrides may be pruned during reconciliation; callers should not count on them round-tripping. The project loader reports a reconciliation error as a warning and retains the supplied whole objects rather than silently replacing the entire store. **Entire unrecognized model or shared-group objects are preserved** when the corresponding definition is unavailable, so a project can travel between environments without losing those objects. Retention does not make an unavailable model usable for generation. Do not depend on unknown fields *inside* a recognized object surviving reconciliation. Stored voice/file paths declared project-local are normalized to portable relative forms; model repository IDs and other machine-local targets are not rewritten as though they were project files.

## Relationship to configured SGL-Omni models

All shipped model identities, including entries without Python aliases such as CosyVoice3, are installed from the catalog at import time. Startup validates remote request policies and installs their specs/settings through one common registration path, preserving existing handles. Local inference keeps its Python model implementations, but its settings are catalog-declared too; metadata and settings for all backends remain available regardless of the active environment. Canonical shipped real-model IDs use the backend suffixes `_local`, `_sglomni`, and `_audiocpp`; `none` is unchanged. File tags, storage defaults, shared-group names and field sharing remain independent of those IDs. The pre-release ID rename needs no migration for undeployed IDs; v1/v2 legacy flat fields retain their old names and migrate to the current canonical owners. Main and model-worker processes compare catalog fingerprints; editing the application catalog requires a restart.

The catalog controls model metadata, persisted storage, server numeric parameters, named behavior policies, request defaults and dynamic voice/model menus. Backend-specific adapters still own execution. Adding a supported SGL-Omni entry does not require Python `Project` fields, a legacy declaration, or a new server class. `behavior.orchestration` is valid for every SGL-Omni entry; preserve existing storage key names (`batch_size` versus `concurrent_requests`) when editing shipped entries. `can_batch = false` disables use of that storage without deleting saved values. The catalog is not an arbitrary plugin language: new protocol behavior, control types and execution policies still require implementation and validation.

## Migration and persistence logistics

Migration is **load-driven and one-way**. [`ProjectUtil.load_using_dir_path`](../tts_audiobook_tool/project_support/project_util.py) delegates to [`ProjectLoadUtil`](../tts_audiobook_tool/project_support/project_load_util.py), which reads the settings, obtains the book from the inline legacy representation or external text file as appropriate, and validates through `Project.model_validate`. Its normalization first remaps recognized old names and legacy text/settings representations, remembers which flat model values were actually supplied, then reconciles them with any supplied `model_settings`. This works for older project files and settings snapshots; callers should not manually rename individual model fields.

### Selecting the project voice list (v1–v3 → v4)

An **explicit top-level `voice_references`, including `[]`, wins** over all stale legacy clone lists. It is not merged with or repopulated from them. Otherwise, migration collects populated legacy sources from v1/v2 flat filename/transcript fields and v3 private/shared clone lists, retaining each list's order and paired transcripts. Empty lists do not count as populated sources, and legacy entries without a sample are not migrated as voices.

| Populated independent legacy sources | Action |
| --- | --- |
| Zero | Use one default voice; do not prompt. |
| One | Adopt its complete list silently. |
| Two or more | Block until the user chooses one complete list to keep. Do not merge or auto-select. |

Count **storage sources**, not model/backend variants: one existing shared-group list counts once even when several model IDs read it. Independent private/flat lists remain independent even if their filenames and transcripts are identical; equality does not authorize automatic selection. Collect sources before any historical conversion fans one shared source out into several private objects. Do not favor the currently selected/available model.

The blocking choice uses this exact prompt template (including the spaces before the two line breaks). The opening subject is `Project {dir_name}` when the project directory name is known to the caller, and `This project` otherwise:

```python
f'🔔 {COL_ACCENT}Action required:\n{COL_DEFAULT}{subject} contains voice clone lists for multiple models, but the app \nnow uses one voice clone list per project, shared by all models. \nChoose which model’s voice clone list to keep:'
```

Insert a blank line before the choices. Format each as `[{n}] {model_name} {model id}: {COL_DIM} {filename1}, {filename2}`, showing **all filenames** and the relevant model names/IDs for a shared source. End with `[0] Abort` followed by a blank line. Empty input is invalid, not Abort: print `Please choose a number from 1 to {n}, or 0 to abort.` followed by a blank line and ask again. The choice keeps the selected ordered pairs only; no referenced audio files are deleted, including files from unselected lists. Cancellation must leave the original project untouched. Noninteractive callers must fail with an actionable ambiguity error (identify the sources and direct the user to open/choose interactively or provide an explicit top-level list), not choose arbitrarily.

### v4 → v5: per-voice room reverb

[`prepare_project_voice_references`](../tts_audiobook_tool/project_support/voice_reference_migration.py) removes a top-level `room_reverb` and applies its value to every voice; a voice entry's own `room_reverb` wins. An unknown id becomes off. With no voices, the value is dropped. `room_reverb_stereo` is unchanged. Crop time strings are parsed and rewritten as numbers. The same step runs for project loads, settings imports, and ABR snapshots.

### Historical non-voice and text compatibility

- Except for the explicit MOSS conversion below, existing v3 **private or shared objects win as whole objects** over corresponding flat non-voice fields. Missing settings inside such an object are not backfilled from the old representation. This prevents a stale flat value from unexpectedly reappearing after a reset.
- Where there is no new owner object, declared old non-voice values move to the correct private/shared section. Shared values move **once** to their group. Legacy default/sentinel values generally become absent overrides, but seed/random values retain their distinct meaning. Recognized input aliases and pre-rename spellings are accepted in the load path.
- Flat model keys are removed from the normalized settings and are **never emitted** by the serializer. Historical mappings remain input-only to keep v1/v2/v3 projects readable. Unknown whole model/shared objects survive reconciliation, as described above, without becoming authoritative clone storage.
- Earlier text migrations remain supported: inline book text and accepted legacy external payloads are normalized to the split layout and `book.v2` when rewritten. This is independent of settings migration; v4/v5 do not duplicate book text inside `project.json`.

Retired-group compatibility (`fork_retired_group`) consumes each old `model_settings.shared.<group>` and converts the group's pre-v3 flat fields using frozen historical ownership/defaults, not current bindings. Every historical value is copied to each original member that still owns a same-named setting; members that no longer own it do not receive it (e.g. the old Fish S2 `seed` goes only to `fish_s2_local`). The same precedence, non-destructive and voice-source rules described for MOSS below apply to every retired group. A retired group in an already-current project is forked in memory on each load and persisted at the next ordinary save; it does not force a rewrite or backup. MOSS compatibility explicitly consumes old `model_settings.shared.moss` and converts pre-v3 flat MOSS fields the same way. Seed and batch size seed the original three private types; both sampling sets go to local MOSS, and only the matching set goes to each remote type. Target and rolling continuation stay local. Conversion only fills missing private keys; **current raw keys win**, including `null`, default values and sentinels, before ordinary reconciliation. Historically, v3 also copied voices/transcripts to those private types and treated each existing clone list atomically. V4 instead selects one project-wide voice source under the rules above, counting the original shared MOSS list once rather than its historical fan-out. The old group is consumed only after successful conversion; malformed groups are retained non-destructively by existing load-error handling. Unknown whole objects for other groups remain preserved. Successfully migrated settings omit `shared.moss` when saved.

### Saving and importing

**Successful on-disk migration saves the current version immediately**, after any required choice. Any project whose `version` is not current (or that still has a top-level `room_reverb`) is migrated. Before rewriting the original settings, preserve the original `project.json` as `project.json.pre-v{N}.bak`, where N is the version being migrated **to** (`PROJECT_SPEC_VERSION`, currently `pre-v5`), whatever the original version; if that backup already exists, use an unused numbered suffix (`.pre-v5.bak.2`, …) rather than overwriting it. Backup failure must prevent the rewrite. This is a settings-file backup, not a complete project/audio backup, and migration never deletes reference or generated audio. Historical v3 loading could defer a settings rewrite until a later save; that timing is superseded since the v4 migration.

`ProjectTextIOUtil.save_book` writes the separate book payload when text itself needs persisting. Ordinary settings changes otherwise require a save. JSON artifacts are serialized before writing and saved through [`JsonSaveUtil`](../tts_audiobook_tool/app_support/JsonSaveUtil.py) with artifact-specific locking and atomic replacement; the settings and book files are not a single cross-file transaction. Writes do not retain parallel legacy fields (clone lists, project-wide `room_reverb`) for old app versions; an older app refuses a newer `version`.

Clone/transfer, worker project transfer, and settings snapshots use the same current representation: one project voice list plus non-voice model settings. ABR metadata snapshots omit executable `dir_path` and carry `source_dir_display` for display. Import validates and normalizes snapshots, including v1–v4 settings (v4 snapshots get the room reverb migration below), and applies **the same blocking voice-list choice before any destination writes or supporting-file copies**. A cancelled or ambiguous noninteractive import must not partially create/modify the destination. Project-local supporting text/voice files are copied separately where available; filenames are resolved in the importing project's context, not an untrusted snapshot source path. The ABR format itself does not change. See [`project_transfer_util.py`](../tts_audiobook_tool/project_support/project_transfer_util.py) and [`abr-metadata-spec.md`](abr-metadata-spec.md).

## Maintenance rules and executable references

- Keep stable model IDs, group membership, owner bindings, stored types, and the semantics of persisted values stable. Renames, regrouping, or reinterpretation require explicit compatibility migration, not a JSON edit or silent reconciliation. The historical MOSS ownership fork used load compatibility within v3 without a schema/version bump, the Fish S2 fork did the same within v4, and the Qwen3 fork within v5; project-wide clone storage is the separate v4 change.
- A new configured model gets private settings by default; same-name parameters do not imply sharing. Extend the validated adapter vocabulary when genuinely new server behavior is required.
- Route primary voice consumers through the shared project ordered pairs; compatibility voice accessors must not recreate authoritative model-scoped lists. Use `get_model_setting`/`set_model_setting` and registry-derived bindings for non-voice model settings. Do not reintroduce flat model fields or duplicate storage names in `TtsModelSpec`.
- Preserve unknown whole objects when a definition is unavailable, while keeping recognized ownership and paths validated. Maintain text/book-format migrations independently of the model-settings schema.

The focused executable references are [`test_model_settings_registry.py`](../tests/models/test_model_settings_registry.py) (ownership, precedence, defaults, pruning and unknown-object retention), [`test_project_book_integration.py`](../tests/project/test_project_book_integration.py) (book and old-project migration), [`test_configured_new_model.py`](../tests/models/test_configured_new_model.py) (additional configured ID), [`test_configured_sgl_omni_variants.py`](../tests/models/test_configured_sgl_omni_variants.py) (server variants), and [`test_project_new_menu.py`](../tests/project/test_project_new_menu.py) (ABR import). Tests with mocked server transport do not by themselves establish live-server compatibility.
