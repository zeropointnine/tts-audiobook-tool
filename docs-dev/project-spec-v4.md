# Project Spec v4

## Scope and version boundaries

Project spec v4 is the **on-disk project settings format**. Its defining change is one top-level, ordered `voice_references` list shared by all models **within a project**, not an application-global voice library. Each entry is a `{file_name, transcript}` pair. Model-specific generation settings remain in backend-neutral objects under `model_settings`, and the split between settings and book text remains in place. This applies to local inference, SGL-Omni, and audio.cpp alike.

Historically, v3 moved flat model fields into private/shared model-setting objects, including model-scoped clone lists. V4 retains that organization for non-voice settings but replaces those clone lists with the project-wide list. V1, v2, and v3 remain accepted migration inputs.

Independent version numbers should not be conflated:

- `project.json` has `"version": 4` (`PROJECT_SPEC_VERSION` in [`constants.py`](../tts_audiobook_tool/constants.py)).
- `project_text.json` still uses `"format": "book.v2"` ([`book_serialization.py`](../tts_audiobook_tool/app_types/book_serialization.py)). Project spec v4 does **not** introduce a new book format.
- The [ABR metadata format](<abr-metadata-spec.md>) is unchanged; its version is independent of the project settings version.
- The application-shipped [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) uses `schema_version = 1`. This is the model-catalog schema, not a project-file version. The loader rejects other catalog schema versions; it does not migrate them. This one TOML file declares the built-in local, SGL-Omni and audio.cpp server, and `"none"` placeholder specs, plus server-specific definitions and optional additional configured server entries. Every shipped entry receives a canonical Python handle in TOML order, looked up by its catalog ID; no model-named attributes or Python declaration inventory is needed. `TtsModelType.require_by_id("fish_s2_local")` is strict for hardcoded IDs and raises `ValueError` for an unknown ID. `TtsModelType.get_by_id(saved_id)` is tolerant for persisted/external IDs and falls back to the registered `"none"` handle without changing the saved string.

The canonical project directory has:

| Artifact | Responsibility |
| --- | --- |
| `project.json` | Project-wide settings and ordered `voice_references`, selected model ID, and non-voice `model_settings` overrides. |
| `project_text.json` | Structured `book.v2` text, sections, phrase groups, and their `voice_index` values. |
| Optional `project_text_raw.txt`, `project_text.epub`, `voice/`, generated audio | Source material, voice clips, and outputs; these are not inline model-setting objects. |

See [`Project`](../tts_audiobook_tool/project.py), [`ProjectSerializationUtil`](../tts_audiobook_tool/project_support/project_serialization_util.py), and [`ProjectTextIOUtil`](../tts_audiobook_tool/project_support/project_text_io_util.py) for the respective in-memory, settings-file, and text-file boundaries.

For the rules governing the selected model ID (`Project.tts_model_type`), refer to [TTS model selection rules](<tts-model-selection.md>). Selection policy is documented there rather than in this storage-format reference.

## Canonical settings shape

The following is illustrative: absent override keys are normal, and these objects are not a complete listing of project-wide settings.

```json
{
  "version": 4,
  "language_code": "en",
  "voice_select_mode": "custom",
  "voice_references": [
    {"file_name": "sample.flac", "transcript": "Reference text."},
    {"file_name": "second.flac", "transcript": ""}
  ],
  "model_settings": {
    "models": {
      "fish_s2_sglomni": {
        "orchestration": {"concurrent_requests": 2}
      },
      "higgs_v3_sglomni": {
        "parameters": {"temperature": 0.8},
        "orchestration": {"batch_size": 2}
      }
    },
    "shared": {
      "fish_s2": {
        "model_ids": ["fish_s2_local", "fish_s2_sglomni"],
        "parameters": {"top_k": 50}
      }
    }
  }
}
```

Top-level `voice_references` is the sole authoritative clone list and is always serialized, including `[]`. It pairs each project-local voice filename with its transcript in order; an unused/absent transcript is represented by `""`. V4 output does not store authoritative clone lists under private models or shared groups.

Entries may additionally carry `crop_file_name`, `crop_start`, `crop_end`, and `crop_transcript`. Crop times persist as decimal strings; a finite range must satisfy `end > start >= 0`. The explicitly owned `crop_file_name` stores only a basename (`<uuid>.flac`); validation rejects paths, absolute names, and traversal components. Its disk path remains `voice/crops/<basename>`. `ProjectVoiceUtil.get_cropped_voice_relative_path(entry)` derives `crops/<basename>` for the generation interface and transfer supporting-file manifest; `resolve_cropped_voice_file_path(project, entry)` centralizes full-path resolution. A new crop reserves a unique file, and re-editing normally retains that filename. Without a valid explicit basename, all crop metadata is dropped with a warning, including old development entries storing paths such as `crops/<uuid>.flac`. The feature has not shipped, so there is no migration or `_crop` sibling-name fallback. Original filenames/transcripts and old files remain untouched. Project transfers copy both originals and active crop files, preserve nested paths, and report missing crops. See [voice sample crop architecture](<voice-sample-crop.md>).

`model_settings.models` is keyed by stable **model ID**; `model_settings.shared` is keyed by a declared **storage group**, not a selectable model. Inside a recognized object, `parameters` holds named overrides, `files` holds declared additional project-local file strings, and `orchestration` holds batch/concurrency values. Sections appear only when needed. The same model may read some non-voice settings from a shared object and others from a private object; there is no duplicate authoritative copy.

Top-level values such as `language_code`, `markers`, `generate_range`, `voice_select_mode`, export options, and the compatibility placeholder `none_voice_file_name` remain project-wide. A project does not record which TTS model produced its audio: remote servers expose several models at once, so "the model in use" is not a stable property to compare against. `project_text.json` remains separately serialized as a structured `Book`; `voice_index` belongs to each phrase group there, whereas the voice samples it selects are the project-wide ordered pairs here. For selection modes and scheduling, see [`generate-files-ordering.md`](generate-files-ordering.md).

All voice consumers use this same ordered list: menus and editing, generation and rotation, per-line/dialog assignment, local and server adapters, workers, and clone/transfer/snapshot paths. A transcript is collected for every added primary sample — sidecar text file or STT — regardless of the selected model, so the shared list is complete for any model. Before a full-screen TTS feature launches (generation, chat, realtime playback), the interactive app pre-flights the list: a required-but-empty list, a missing or undecodable file, or an unresolvable transcript blocks the launch, and empty transcripts are auto-transcribed for a transcript-using selection. There is no load-time and no readiness-based voice-file verification: missing or undecodable references survive loading untouched, readiness deliberately ignores voice state, and model adapters surface voice problems only at generation time. Transcripts stay paired with their filenames during append, replace, reorder, and clear operations. They remain stored and editable even while a model that does not use reference transcripts is selected; that model simply does not send/use them. Clone support and transcript requirements remain model-specific capabilities, not a reason to create separate storage.

Pocket's selected predefined voice takes precedence during generation without clearing the shared list. IndexTTS2's secondary emotion reference remains a model-specific file setting, not a second project clone list.

## Ownership and interpretation

[`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) resolves `(model ID, setting name) → section, private model or shared group, type and storage default` from the current catalog. `models.settings` declares persisted defaults/types/owners independently of request defaults; the root `setting_groups` table declares exact shared membership. Remote backend parameters, seed policies and `behavior.orchestration` supply standard private declarations when no explicit override is needed. Voice/transcript bindings remain compatibility/capability metadata; project accessors route primary clone values to the project-wide list rather than private/shared storage. Every model follows this same registration rule, including data-only entries available through canonical ID lookup; no Python model declaration is required. Projects cannot create sharing relationships by editing membership or using matching setting names.

[`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py) and [`model_settings_compat.py`](../tts_audiobook_tool/project_support/model_settings_compat.py) freeze pre-v3 input fields, defaults and storage mappings solely for migration. They do not initialize current settings.

The built-in shared groups are `auk` (AuK/Flash servers), `fish_s2` (local/server), and `qwen3` (local/server); their non-voice ownership is unchanged. Sharing of those settings is **per field**, not per model family: for example Fish S2's sampling parameters are shared, its local-only options are private, and server concurrency is private. Primary voice references are now shared across all models in the project, independently of these groups. A model switch never copies or clamps a shared override. A server variant may resolve a different effective default, cap an outgoing value, omit a request field, or warn about a shared value without modifying what the local variant will see. In particular, server Fish S2 caps outgoing top-k without overwriting the shared value.

MOSS has no current shared model-setting group. The unchanged IDs `moss_local`, `moss_delay_sglomni`, and `moss_local_sglomni` each privately own seed and batch size, but use the same project-wide voice references as every other model. Local MOSS owns both Delay and Local Transformer sampling sets plus local-only target/rolling continuation; each remote type owns only its architecture's sampling set. Common controls/adapters reuse code, not private generation state. Historically, the MOSS-only ownership fork happened within project v3 without changing catalog schema 1, defaults, request semantics or backends; v4 separately changes primary voice storage. See [MOSS architecture](<moss-model-architecture.md>).

A missing override means “use the declaration's default”; a stored value equal to that default is still an explicit override. `Project.get_model_setting(model_id, name)` reads through the registry, while `set_model_setting(..., reset=True)` removes an override. Some older numeric controls use `-1` as a *default* sentinel and reset to absence; seed `-1` can instead mean *random* and remains an explicit value. There is no global interpretation of `-1`. Variant-specific effective defaults, validation, menu presentation, and HTTP request policy are separate from storage ownership. Menus and other callers refer to a setting by `SettingRef(model_id, name)` rather than a flat Python attribute name.

Recognized objects are reconciled against the declarations: unsupported fields inside them are pruned and declared values are checked; the top-level voice-reference pairs are validated separately, and a recognized shared group's membership must match its authorized members. Unusable numeric parameter overrides may be pruned during reconciliation; callers should not count on them round-tripping. The project loader reports a reconciliation error as a warning and retains the supplied whole objects rather than silently replacing the entire store. **Entire unrecognized model or shared-group objects are preserved** when the corresponding definition is unavailable, so a project can travel between environments without losing those objects. Retention does not make an unavailable model usable for generation. Do not depend on unknown fields *inside* a recognized object surviving reconciliation. Stored voice/file paths declared project-local are normalized to portable relative forms; model repository IDs and other machine-local targets are not rewritten as though they were project files.

## Relationship to configured SGL-Omni models

All shipped model identities, including entries without Python aliases such as CosyVoice3, are installed from the catalog at import time. Startup validates remote request policies and installs their specs/settings through one common registration path, preserving existing handles. Local inference keeps its Python model implementations, but its settings are catalog-declared too; metadata and settings for all backends remain available regardless of the active environment. Canonical shipped real-model IDs use the backend suffixes `_local`, `_sglomni`, and `_audiocpp`; `none` is unchanged. File tags, storage defaults, shared-group names and field sharing remain independent of those IDs. The pre-release ID rename needs no migration for undeployed IDs; v1/v2 legacy flat fields retain their old names and migrate to the current canonical owners. Main and model-worker processes compare catalog fingerprints; editing the application catalog requires a restart.

The catalog controls model metadata, persisted storage, server numeric parameters, named behavior policies, request defaults and dynamic voice/model menus. Backend-specific adapters still own execution. Adding a supported SGL-Omni entry does not require Python `Project` fields, a legacy declaration, or a new server class. `behavior.orchestration` is valid for every SGL-Omni entry; preserve existing storage key names (`batch_size` versus `concurrent_requests`) when editing shipped entries. `can_batch = false` disables use of that storage without deleting saved values. The catalog is not an arbitrary plugin language: new protocol behavior, control types and execution policies still require implementation and validation.

## Migration and persistence logistics

Migration is **load-driven and one-way**. [`ProjectUtil.load_using_dir_path`](../tts_audiobook_tool/project_support/project_util.py) delegates to [`ProjectLoadUtil`](../tts_audiobook_tool/project_support/project_load_util.py), which reads the settings, obtains the book from the inline legacy representation or external text file as appropriate, and validates through `Project.model_validate`. Its normalization first remaps recognized old names and legacy text/settings representations, remembers which flat model values were actually supplied, then reconciles them with any supplied `model_settings`. This works for older project files and settings snapshots; callers should not manually rename individual model fields.

### Selecting the v4 voice list

An **explicit top-level `voice_references`, including `[]`, wins** over all stale legacy clone lists. It is not merged with or repopulated from them. Otherwise, migration collects populated legacy sources from v1/v2 flat filename/transcript fields and v3 private/shared clone lists, retaining each list's order and paired transcripts. Empty lists do not count as populated sources.

| Populated independent legacy sources | Action |
| --- | --- |
| Zero | Use `voice_references: []`; do not prompt. |
| One | Adopt its complete list silently. |
| Two or more | Block until the user chooses one complete list to keep. Do not merge or auto-select. |

Count **storage sources**, not model/backend variants: one existing shared-group list counts once even when several model IDs read it. Independent private/flat lists remain independent even if their filenames and transcripts are identical; equality does not authorize automatic selection. Collect sources before any historical conversion fans one shared source out into several private objects. Do not favor the currently selected/available model.

The blocking choice uses this exact prompt template (including the spaces before the two line breaks). The opening subject is `Project {dir_name}` when the project directory name is known to the caller, and `This project` otherwise:

```python
f'🔔 {COL_ACCENT}Action required:\n{COL_DEFAULT}{subject} contains voice clone lists for multiple models, but the app \nnow uses one voice clone list per project, shared by all models. \nChoose which model’s voice clone list to keep:'
```

Insert a blank line before the choices. Format each as `[{n}] {model_name} {model id}: {COL_DIM} {filename1}, {filename2}`, showing **all filenames** and the relevant model names/IDs for a shared source. End with `[0] Abort` followed by a blank line. Empty input is invalid, not Abort: print `Please choose a number from 1 to {n}, or 0 to abort.` followed by a blank line and ask again. The choice keeps the selected ordered pairs only; no referenced audio files are deleted, including files from unselected lists. Cancellation must leave the original project untouched. Noninteractive callers must fail with an actionable ambiguity error (identify the sources and direct the user to open/choose interactively or provide an explicit top-level list), not choose arbitrarily.

### Historical non-voice and text compatibility

- Except for the explicit MOSS conversion below, existing v3 **private or shared objects win as whole objects** over corresponding flat non-voice fields. Missing settings inside such an object are not backfilled from the old representation. This prevents a stale flat value from unexpectedly reappearing after a reset.
- Where there is no new owner object, declared old non-voice values move to the correct private/shared section. Shared values move **once** to their group. Legacy default/sentinel values generally become absent overrides, but seed/random values retain their distinct meaning. Recognized input aliases and pre-rename spellings are accepted in the load path.
- Flat model keys are removed from the normalized settings and are **never emitted** by the v4 serializer. Historical mappings remain input-only to keep v1/v2/v3 projects readable. Unknown whole model/shared objects survive reconciliation, as described above, without becoming authoritative clone storage.
- Earlier text migrations remain supported: inline book text and accepted legacy external payloads are normalized to the split layout and `book.v2` when rewritten. This is independent of settings migration; v4 does not duplicate book text inside `project.json`.

MOSS compatibility explicitly consumes old `model_settings.shared.moss` and converts pre-v3 flat MOSS fields using frozen historical ownership/defaults, not current bindings. Seed and batch size seed the original three private types; both sampling sets go to local MOSS, and only the matching set goes to each remote type. Target and rolling continuation stay local. Conversion only fills missing private keys; **current raw keys win**, including `null`, default values and sentinels, before ordinary reconciliation. Historically, v3 also copied voices/transcripts to those private types and treated each existing clone list atomically. V4 instead selects one project-wide voice source under the rules above, counting the original shared MOSS list once rather than its historical fan-out. The old group is consumed only after successful conversion; malformed groups are retained non-destructively by existing load-error handling. Unknown whole objects for other groups remain preserved. Successfully migrated settings omit `shared.moss` when saved.

### Saving and importing

**Successful on-disk migration saves v4 immediately**, after any required choice. Before rewriting the original settings, preserve the original `project.json` as `project.json.pre-v4.bak`; if that backup already exists, use an unused numbered suffix rather than overwriting it. Backup failure must prevent the rewrite. This is a settings-file backup, not a complete project/audio backup, and migration never deletes reference or generated audio. Historical v3 loading could defer a settings rewrite until a later save; that timing is superseded for v4 migration.

`ProjectTextIOUtil.save_book` writes the separate book payload when text itself needs persisting. Ordinary settings changes otherwise require a save. JSON artifacts are serialized before writing and saved through [`JsonSaveUtil`](../tts_audiobook_tool/app_support/JsonSaveUtil.py) with artifact-specific locking and atomic replacement; the settings and book files are not a single cross-file transaction. V4 writes do not retain parallel legacy clone fields for old app versions.

Clone/transfer, worker project transfer, and settings snapshots use the same v4 representation: one project voice list plus non-voice model settings. ABR metadata snapshots omit executable `dir_path` and carry `source_dir_display` for display. Import validates and normalizes snapshots, including v1/v2/v3 settings, and applies **the same blocking voice-list choice before any destination writes or supporting-file copies**. A cancelled or ambiguous noninteractive import must not partially create/modify the destination. Project-local supporting text/voice files are copied separately where available; filenames are resolved in the importing project's context, not an untrusted snapshot source path. The ABR format itself does not change. See [`project_transfer_util.py`](../tts_audiobook_tool/project_support/project_transfer_util.py) and [`abr-metadata-spec.md`](abr-metadata-spec.md).

## Maintenance rules and executable references

- Keep stable model IDs, group membership, owner bindings, stored types, and the semantics of persisted values stable. Renames, regrouping, or reinterpretation require explicit compatibility migration, not a JSON edit or silent reconciliation. The historical MOSS ownership fork used load compatibility within v3 without a schema/version bump; project-wide clone storage is the separate v4 change.
- A new configured model gets private settings by default; same-name parameters do not imply sharing. Extend the validated adapter vocabulary when genuinely new server behavior is required.
- Route primary voice consumers through the shared project ordered pairs; compatibility voice accessors must not recreate authoritative model-scoped lists. Use `get_model_setting`/`set_model_setting` and registry-derived bindings for non-voice model settings. Do not reintroduce flat model fields or duplicate storage names in `TtsModelSpec`.
- Preserve unknown whole objects when a definition is unavailable, while keeping recognized ownership and paths validated. Maintain text/book-format migrations independently of the model-settings schema.

The focused executable references are [`test_model_settings_registry.py`](../tests/test_model_settings_registry.py) (ownership, precedence, defaults, pruning and unknown-object retention), [`test_project_book_integration.py`](../tests/test_project_book_integration.py) (book and old-project migration), [`test_configured_new_model.py`](../tests/test_configured_new_model.py) (additional configured ID), [`test_configured_sgl_omni_variants.py`](../tests/test_configured_sgl_omni_variants.py) (server variants), and [`test_project_new_menu.py`](../tests/test_project_new_menu.py) (ABR import). Tests with mocked server transport do not by themselves establish live-server compatibility.
