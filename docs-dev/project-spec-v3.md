# Project Spec v3

## Scope and version boundaries

Project spec v3 is the **on-disk project settings format**. Its defining change is that model-specific values live in backend-neutral, model-scoped objects under `project.json`'s `model_settings`, rather than in top-level model fields. This applies to local inference and SGL-Omni models alike. The project-wide settings and the split between settings and book text remain in place.

Three independent version numbers should not be conflated:

- `project.json` has `"version": 3` (`PROJECT_SPEC_VERSION` in [`constants.py`](../tts_audiobook_tool/constants.py)).
- `project_text.json` still uses `"format": "book.v2"` ([`book_serialization.py`](../tts_audiobook_tool/app_types/book_serialization.py)). Project spec v3 does **not** introduce `book.v3`.
- The application-shipped [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) uses `schema_version = 1`. This is the model-catalog schema, not a project-file version. The loader rejects other catalog schema versions; it does not migrate them. This one TOML file declares the built-in local, SGL-Omni and audio.cpp server, and `"none"` placeholder specs, plus server-specific definitions and optional additional configured server entries. Every shipped entry receives a canonical Python handle in TOML order, looked up by its catalog ID; no model-named attributes or Python declaration inventory is needed. `TtsModelType.require_by_id("fish_s2_local")` is strict for hardcoded IDs and raises `ValueError` for an unknown ID. `TtsModelType.get_by_id(saved_id)` is tolerant for persisted/external IDs and falls back to the registered `"none"` handle without changing the saved string.

The canonical project directory has:

| Artifact | Responsibility |
| --- | --- |
| `project.json` | Project-wide settings, selected model ID, and `model_settings` overrides. |
| `project_text.json` | Structured `book.v2` text, sections, phrase groups, and their `voice_index` values. |
| Optional `project_text_raw.txt`, `project_text.epub`, `voice/`, generated audio | Source material, voice clips, and outputs; these are not inline model-setting objects. |

See [`Project`](../tts_audiobook_tool/project.py), [`ProjectSerializationUtil`](../tts_audiobook_tool/project_support/project_serialization_util.py), and [`ProjectTextIOUtil`](../tts_audiobook_tool/project_support/project_text_io_util.py) for the respective in-memory, settings-file, and text-file boundaries.

For the rules governing the selected model ID (`Project.tts_model_type`), refer to [TTS model selection rules](<tts-model-selection.md>). Selection policy is documented there rather than in this storage-format reference.

## Canonical settings shape

The following is illustrative: absent override keys are normal, and these objects are not a complete listing of project-wide settings.

```json
{
  "version": 3,
  "language_code": "en",
  "voice_select_mode": "custom",
  "model_settings": {
    "models": {
      "fish_s2_sglomni": {
        "orchestration": {"concurrent_requests": 2}
      },
      "higgs_v3_sglomni": {
        "parameters": {"temperature": 0.8},
        "voice_references": [
          {"file_name": "sample.flac", "transcript": "Reference text."}
        ],
        "orchestration": {"batch_size": 2}
      }
    },
    "shared": {
      "fish_s2": {
        "model_ids": ["fish_s2_local", "fish_s2_sglomni"],
        "parameters": {"top_k": 50},
        "voice_references": [
          {"file_name": "fish.flac", "transcript": "Fish reference."}
        ]
      }
    }
  }
}
```

`model_settings.models` is keyed by stable **model ID**; `model_settings.shared` is keyed by a declared **storage group**, not a selectable model. Inside a recognized object, `parameters` holds named overrides, `voice_references` pairs each project-local voice filename with its optional transcript in order, `files` holds declared additional project-local file strings, and `orchestration` holds batch/concurrency values. Sections appear only when needed. The same model may read some settings from a shared object and others from a private object; there is no duplicate authoritative copy.

Top-level values such as `language_code`, `markers`, `generate_range`, `voice_select_mode`, export options, and the compatibility placeholder `none_voice_file_name` remain project-wide. A project does not record which TTS model produced its audio: remote servers expose several models at once, so "the model in use" is not a stable property to compare against. `project_text.json` remains separately serialized as a structured `Book`; `voice_index` belongs to each phrase group there, whereas the voice samples it selects are model settings here. For selection modes and scheduling, see [`generate-files-ordering.md`](generate-files-ordering.md).

## Ownership and interpretation

[`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) resolves `(model ID, setting name) → section, private model or shared group, type and storage default` from the current catalog. `models.settings` declares persisted defaults/types/owners independently of request defaults; the root `setting_groups` table declares exact shared membership. Remote backend parameters, voices, seed policies and `behavior.orchestration` supply standard private declarations when no explicit override is needed. Every model follows this same registration rule, including data-only entries available through canonical ID lookup; no Python model declaration is required. Projects cannot create sharing relationships by editing membership or using matching setting names.

[`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py) and [`model_settings_compat.py`](../tts_audiobook_tool/project_support/model_settings_compat.py) freeze pre-v3 input fields, defaults and storage mappings solely for migration. They do not initialize current settings.

The built-in shared groups are `auk` (AuK/Flash servers), `fish_s2` (local/server), and `qwen3` (local/server); their ownership is unchanged. Sharing is **per field**, not per model family: for example Fish S2's voices and sampling parameters are shared, its local-only options are private, and server concurrency is private. A model switch never copies or clamps a shared override. A server variant may resolve a different effective default, cap an outgoing value, omit a request field, or warn about a shared value without modifying what the local variant will see. In particular, server Fish S2 caps outgoing top-k without overwriting the shared value.

MOSS has no current shared group. The unchanged IDs `moss_local`, `moss_delay_sglomni`, and `moss_local_sglomni` each privately own voices/transcripts, seed and batch size. Local MOSS owns both Delay and Local Transformer sampling sets plus local-only target/rolling continuation; each remote type owns only its architecture's sampling set. Common controls/adapters reuse code, not state. This MOSS-only ownership change leaves project version 3, catalog schema 1, defaults and request semantics unchanged and adds no backend. See [MOSS architecture](<moss-model-architecture.md>).

A missing override means “use the declaration's default”; a stored value equal to that default is still an explicit override. `Project.get_model_setting(model_id, name)` reads through the registry, while `set_model_setting(..., reset=True)` removes an override. Some older numeric controls use `-1` as a *default* sentinel and reset to absence; seed `-1` can instead mean *random* and remains an explicit value. There is no global interpretation of `-1`. Variant-specific effective defaults, validation, menu presentation, and HTTP request policy are separate from storage ownership. Menus and other callers refer to a setting by `SettingRef(model_id, name)` rather than a flat Python attribute name.

Recognized objects are reconciled against the declarations: unsupported fields inside them are pruned, declared values and voice-reference structures are checked, and a recognized shared group's membership must match its authorized members. Unusable numeric parameter overrides may be pruned during reconciliation; callers should not count on them round-tripping. The project loader reports a reconciliation error as a warning and retains the supplied whole objects rather than silently replacing the entire store. **Entire unrecognized model or shared-group objects are preserved** when the corresponding definition is unavailable, so a project can travel between environments without losing those objects. Retention does not make an unavailable model usable for generation. Do not depend on unknown fields *inside* a recognized object surviving reconciliation. Stored voice/file paths declared project-local are normalized to portable relative forms; model repository IDs and other machine-local targets are not rewritten as though they were project files.

## Relationship to configured SGL-Omni models

All shipped model identities, including entries without Python aliases such as CosyVoice3, are installed from the catalog at import time. Startup validates remote request policies and installs their specs/settings through one common registration path, preserving existing handles. Local inference keeps its Python model implementations, but its settings are catalog-declared too; metadata and settings for all backends remain available regardless of the active environment. Canonical shipped real-model IDs use the backend suffixes `_local`, `_sglomni`, and `_audiocpp`; `none` is unchanged. File tags, storage defaults, shared-group names and field sharing remain independent of those IDs. The pre-release ID rename needs no migration for undeployed IDs; v1/v2 legacy flat fields retain their old names and migrate to the current canonical owners. Main and model-worker processes compare catalog fingerprints; editing the application catalog requires a restart.

The catalog controls model metadata, persisted storage, server numeric parameters, named behavior policies, request defaults and dynamic voice menus. Backend-specific adapters still own execution. Adding a supported SGL-Omni entry does not require Python `Project` fields, a legacy declaration, or a new server class. `behavior.orchestration` is valid for every SGL-Omni entry; preserve existing storage key names (`batch_size` versus `concurrent_requests`) when editing shipped entries. `can_batch = false` disables use of that storage without deleting saved values. The catalog is not an arbitrary plugin language: new protocol behavior, control types and execution policies still require implementation and validation.

## Migration and persistence logistics

Migration is **load-driven and one-way**. [`ProjectUtil.load_using_dir_path`](../tts_audiobook_tool/project_support/project_util.py) delegates to [`ProjectLoadUtil`](../tts_audiobook_tool/project_support/project_load_util.py), which reads the settings, obtains the book from the inline legacy representation or external text file as appropriate, and validates through `Project.model_validate`. Its normalization first remaps recognized old names and legacy text/settings representations, remembers which flat model values were actually supplied, then reconciles them with any supplied `model_settings`. This works for older project files and settings snapshots; callers should not manually rename individual model fields.

- Except for the explicit MOSS conversion below, existing v3 **private or shared objects win as whole objects** over corresponding flat fields. Missing settings inside such an object are not backfilled from the old representation. This prevents a stale flat value from unexpectedly reappearing after a reset.
- Where there is no new owner object, declared old values move to the correct private/shared section. Voice files and transcripts are paired by index; shared values move **once** to their group. Legacy default/sentinel values generally become absent overrides, but seed/random values retain their distinct meaning. Recognized input aliases and pre-rename spellings are accepted in the load path.
- Flat model keys are removed from the normalized settings and are **never emitted** by the v3 serializer. Historical mappings remain input-only to keep older projects readable. Unknown whole model/shared objects survive reconciliation, as described above.
- Earlier text migrations remain supported: inline book text and accepted legacy external payloads are normalized to the split layout and `book.v2` when rewritten. This is independent of the v3 settings migration; v3 does not duplicate book text inside `project.json`.

MOSS compatibility explicitly consumes old `model_settings.shared.moss` and converts pre-v3 flat MOSS fields using frozen historical ownership/defaults, not current bindings. Relevant settings seed the three private types: voices/transcripts, seed and batch size go to all three; both sampling sets go to local MOSS, and only the matching set goes to each remote type. Target and rolling continuation stay local. Conversion only fills missing private keys; **current raw keys win**, including `null`, default values and sentinels, before ordinary reconciliation. `voice_references` is atomic: an existing list, including `[]`, wins without merging its entries. The old group is consumed only after successful conversion; malformed groups are retained non-destructively by existing load-error handling. Unknown whole objects for other groups remain preserved. Successfully migrated settings omit `shared.moss` when saved.

Loading converts the in-memory `Project`; **loading a flat-settings project alone does not guarantee an immediate rewrite of `project.json`**. MOSS conversion does not change save timing. A subsequent `project.save()` stamps version 3 and writes the canonical settings shape. The loader also saves when its existing text-migration, legacy-applied-field cleanup, path-warning, or corrupt-voice handling calls for it. `ProjectTextIOUtil.save_book` writes the separate book payload when text itself needs persisting. Settings changes otherwise remain in memory until an explicit save. JSON artifacts are serialized before writing and saved through [`JsonSaveUtil`](../tts_audiobook_tool/app_support/JsonSaveUtil.py) with artifact-specific locking and atomic replacement; the two files are not a single cross-file transaction. Keep a copy of a project before opening it with a newer app if a reversible downgrade matters: v3 writes do not keep parallel flat fields for old app versions.

Clone/transfer, worker project transfer, and settings snapshots use the same model-scoped representation. ABR metadata snapshots omit executable `dir_path` and carry `source_dir_display` for display; import validates and normalizes snapshots (including legacy flat settings) before applying them. Project-local supporting text/voice files are copied separately where available. See [`project_transfer_util.py`](../tts_audiobook_tool/project_support/project_transfer_util.py) and [`abr-metadata-spec.md`](abr-metadata-spec.md).

## Maintenance rules and executable references

- Keep stable model IDs, group membership, owner bindings, stored types, and the semantics of persisted values stable. Renames, regrouping, or reinterpretation require explicit compatibility migration, not a JSON edit or silent reconciliation. MOSS's ownership fork uses load compatibility without a schema/version bump.
- A new configured model gets private settings by default; same-name parameters do not imply sharing. Extend the validated adapter vocabulary when genuinely new server behavior is required.
- Route new project model-setting consumers through `get_model_setting`/`set_model_setting` and registry-derived voice/orchestration bindings. Do not reintroduce flat model fields or duplicate storage names in `TtsModelSpec`.
- Preserve unknown whole objects when a definition is unavailable, while keeping recognized ownership and paths validated. Maintain text/book-format migrations independently of the model-settings schema.

The focused executable references are [`test_model_settings_registry.py`](../tests/test_model_settings_registry.py) (ownership, precedence, defaults, pruning and unknown-object retention), [`test_project_book_integration.py`](../tests/test_project_book_integration.py) (book and old-project migration), [`test_configured_new_model.py`](../tests/test_configured_new_model.py) (additional configured ID), [`test_configured_sgl_omni_variants.py`](../tests/test_configured_sgl_omni_variants.py) (server variants), and [`test_project_new_menu.py`](../tests/test_project_new_menu.py) (ABR import). Tests with mocked server transport do not by themselves establish live-server compatibility.
