# Project Spec v3

## Scope and version boundaries

Project spec v3 is the **on-disk project settings format**. Its defining change is that model-specific values live in backend-neutral, model-scoped objects under `project.json`'s `model_settings`, rather than in top-level model fields. This applies to local inference and SGL-Omni models alike. The project-wide settings and the split between settings and book text remain in place.

Three independent version numbers should not be conflated:

- `project.json` has `"version": 3` (`PROJECT_SPEC_VERSION` in [`constants.py`](../tts_audiobook_tool/constants.py)).
- `project_text.json` still uses `"format": "book.v2"` ([`book_serialization.py`](../tts_audiobook_tool/app_types/book_serialization.py)). Project spec v3 does **not** introduce `book.v3`.
- The application-shipped [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) uses `schema_version = 3`. This is the model-catalog schema, not a project-file version. The loader rejects other catalog schema versions; it does not migrate them. This one TOML file declares the built-in local, SGL-Omni server, and `NONE` placeholder specs, plus server-specific definitions and optional additional configured server entries.

The canonical project directory has:

| Artifact | Responsibility |
| --- | --- |
| `project.json` | Project-wide settings, selected model ID, and `model_settings` overrides. |
| `project_text.json` | Structured `book.v2` text, sections, phrase groups, and their `voice_index` values. |
| Optional `project_text_raw.txt`, `project_text.epub`, `voice/`, generated audio | Source material, voice clips, and outputs; these are not inline model-setting objects. |

See [`Project`](../tts_audiobook_tool/project.py), [`ProjectSerializationUtil`](../tts_audiobook_tool/project_support/project_serialization_util.py), and [`ProjectTextIOUtil`](../tts_audiobook_tool/project_support/project_text_io_util.py) for the respective in-memory, settings-file, and text-file boundaries.

## Canonical settings shape

The following is illustrative: absent override keys are normal, and these objects are not a complete listing of project-wide settings.

```json
{
  "version": 3,
  "current_model_type": "server_fish_s2",
  "language_code": "en",
  "voice_select_mode": "custom",
  "model_settings": {
    "models": {
      "server_fish_s2": {
        "orchestration": {"concurrent_requests": 2}
      },
      "server_higgs_v3": {
        "parameters": {"temperature": 0.8},
        "voice_references": [
          {"file_name": "sample.flac", "transcript": "Reference text."}
        ],
        "orchestration": {"batch_size": 2}
      }
    },
    "shared": {
      "fish_s2": {
        "model_ids": ["fish_s2", "server_fish_s2"],
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

Top-level values such as `language_code`, `markers`, `generate_range`, `voice_select_mode`, export options, and the compatibility placeholder `none_voice_file_name` remain project-wide. `current_model_type` is a stable model ID, not a storage owner or a copy of that model's definition. `project_text.json` remains separately serialized as a structured `Book`; `voice_index` belongs to each phrase group there, whereas the voice samples it selects are model settings here. For selection modes and scheduling, see [`generate-files-ordering.md`](generate-files-ordering.md).

## Ownership and interpretation

[`ModelSettingsRegistry`](../tts_audiobook_tool/project_support/model_settings.py) is the authority for `(model ID, setting name) → section, private model or shared group, type and storage default`. The project records overrides and, for recognized shared groups, their exact `model_ids`; it cannot create a new sharing relationship by supplying matching names or editing membership. Stable built-in setting declarations and old flat input-field mappings are maintained via [`model_settings_declarations.py`](../tts_audiobook_tool/project_support/model_settings_declarations.py) and the registry, regardless of the active backend; they are not model specs or settings-ownership declarations in the TOML catalog.

The built-in shared groups are `auk` (AuK/Flash servers), `fish_s2` (local/server), `moss` (local and two servers), and `qwen3` (local/server). Sharing is **per field**, not per model family: for example Fish S2's voices and sampling parameters are shared, its local-only options are private, and server concurrency is private. A model switch never copies or clamps a shared override. A server variant may resolve a different effective default, cap an outgoing value, omit a request field, or warn about a shared value without modifying what the local variant will see. In particular, server Fish S2 caps outgoing top-k without overwriting the shared value.

A missing override means “use the declaration's default”; a stored value equal to that default is still an explicit override. `Project.get_model_setting(model_id, name)` reads through the registry, while `set_model_setting(..., reset=True)` removes an override. Some older numeric controls use `-1` as a *default* sentinel and reset to absence; seed `-1` can instead mean *random* and remains an explicit value. There is no global interpretation of `-1`. Variant-specific effective defaults, validation, menu presentation, and HTTP request policy are separate from storage ownership. Menus and other callers refer to a setting by `SettingRef(model_id, name)` rather than a flat Python attribute name.

Recognized objects are reconciled against the declarations: unsupported fields inside them are pruned, declared values and voice-reference structures are checked, and a recognized shared group's membership must match its authorized members. Unusable numeric parameter overrides may be pruned during reconciliation; callers should not count on them round-tripping. The project loader reports a reconciliation error as a warning and retains the supplied whole objects rather than silently replacing the entire store. **Entire unrecognized model or shared-group objects are preserved** when the corresponding definition is unavailable, so a project can travel between environments without losing those objects. Retention does not make an unavailable model usable for generation. Do not depend on unknown fields *inside* a recognized object surviving reconciliation. Stored voice/file paths declared project-local are normalized to portable relative forms; model repository IDs and other machine-local targets are not rewritten as though they were project files.

## Relationship to configured SGL-Omni models

The configured SGL-Omni route is now the sole server-model route. Built-in handles and local/server/`NONE` specs are parsed from the single [`model_catalog.toml`](../tts_audiobook_tool/tts_models/model_catalog.toml) at import time. At startup the app detects its backend; in SGL-Omni mode it loads and validates the catalog's server behavior, parameters, request defaults and menus, overlays built-in server specs without replacing their stable handles, registers additional supported server IDs, and finalizes their private storage bindings before projects and preferences are resolved. Local inference keeps its Python model implementations and does not load server definitions at backend startup, but uses the **same** v3 settings registry. Built-in IDs, file tags, and existing shared ownership remain stable. Main and model-worker processes compare catalog fingerprints to avoid generating under different definitions; changes require a restart. Invalid/missing server definitions fail clearly rather than falling back to removed server subclasses.

The catalog controls server model metadata, numeric parameters, named behavior policies, request defaults, and dynamic voice menus. The generic adapter delegates HTTP/streaming/audio decoding to the shared SGL-Omni transport. An additional validated server entry can add a new supported server ID with private v3 settings without adding Python `Project` fields, serializers, or a per-model server class; built-in entries retain registry-declared ownership. The catalog is **not** an arbitrary plugin language: new parameter kinds, menu controls, protocol behavior, or sharing relationships require explicit implementation and validation. See [`model_catalog.py`](../tts_audiobook_tool/tts_models/model_catalog.py), [`sgl_omni_definition.py`](../tts_audiobook_tool/tts_models/sgl_omni_definition.py), [`sgl_omni_configured.py`](../tts_audiobook_tool/tts_models/sgl_omni_configured.py), and [`model-worker-architecture.md`](model-worker-architecture.md) for the runtime side of that boundary.

## Migration and persistence logistics

Migration is **load-driven and one-way**. [`ProjectUtil.load_using_dir_path`](../tts_audiobook_tool/project_support/project_util.py) delegates to [`ProjectLoadUtil`](../tts_audiobook_tool/project_support/project_load_util.py), which reads the settings, obtains the book from the inline legacy representation or external text file as appropriate, and validates through `Project.model_validate`. Its normalization first remaps recognized old names and legacy text/settings representations, remembers which flat model values were actually supplied, then reconciles them with any supplied `model_settings`. This works for older project files and settings snapshots; callers should not manually rename individual model fields.

- Existing v3 **private or shared objects win as whole objects** over corresponding flat fields. Missing settings inside such an object are not backfilled from the old representation. This prevents a stale flat value from unexpectedly reappearing after a reset.
- Where there is no new owner object, declared old values move to the correct private/shared section. Voice files and transcripts are paired by index; shared values move **once** to their group. Legacy default/sentinel values generally become absent overrides, but seed/random values retain their distinct meaning. Recognized input aliases and pre-rename spellings are accepted in the load path.
- Flat model keys are removed from the normalized settings and are **never emitted** by the v3 serializer. Historical mappings remain input-only to keep older projects readable. Unknown whole model/shared objects survive reconciliation, as described above.
- Earlier text migrations remain supported: inline book text and accepted legacy external payloads are normalized to the split layout and `book.v2` when rewritten. This is independent of the v3 settings migration; v3 does not duplicate book text inside `project.json`.

Loading converts the in-memory `Project`; **loading a flat-settings project alone does not guarantee an immediate rewrite of `project.json`**. A subsequent `project.save()` stamps version 3 and writes the canonical settings shape. The loader also saves when its existing text-migration, legacy-applied-field cleanup, path-warning, or corrupt-voice handling calls for it. `ProjectTextIOUtil.save_book` writes the separate book payload when text itself needs persisting. Settings changes otherwise remain in memory until an explicit save. JSON artifacts are serialized before writing and saved through [`JsonSaveUtil`](../tts_audiobook_tool/app_support/JsonSaveUtil.py) with artifact-specific locking and atomic replacement; the two files are not a single cross-file transaction. Keep a copy of a project before opening it with a newer app if a reversible downgrade matters: v3 writes do not keep parallel flat fields for old app versions.

Clone/transfer, worker project transfer, and settings snapshots use the same model-scoped representation. ABR metadata snapshots omit executable `dir_path` and carry `source_dir_display` for display; import validates and normalizes snapshots (including legacy flat settings) before applying them. Project-local supporting text/voice files are copied separately where available. See [`project_transfer_util.py`](../tts_audiobook_tool/project_support/project_transfer_util.py) and [`abr-metadata-spec.md`](abr-metadata-spec.md).

## Maintenance rules and executable references

- Keep stable model IDs, group membership, owner bindings, stored types, and the semantics of persisted values stable. Renames, regrouping, or reinterpretation require an explicit versioned migration, not a JSON edit or silent reconciliation.
- A new configured model gets private settings by default; same-name parameters do not imply sharing. Extend the validated adapter vocabulary when genuinely new server behavior is required.
- Route new project model-setting consumers through `get_model_setting`/`set_model_setting` and registry-derived voice/orchestration bindings. Do not reintroduce flat model fields or duplicate storage names in `TtsModelSpec`.
- Preserve unknown whole objects when a definition is unavailable, while keeping recognized ownership and paths validated. Maintain text/book-format migrations independently of the model-settings schema.

The focused executable references are [`test_model_settings_registry.py`](../tests/test_model_settings_registry.py) (ownership, precedence, defaults, pruning and unknown-object retention), [`test_project_book_integration.py`](../tests/test_project_book_integration.py) (book and old-project migration), [`test_configured_new_model.py`](../tests/test_configured_new_model.py) (additional configured ID), [`test_configured_sgl_omni_variants.py`](../tests/test_configured_sgl_omni_variants.py) (server variants), and [`test_project_new_menu.py`](../tests/test_project_new_menu.py) (ABR import). Tests with mocked server transport do not by themselves establish live-server compatibility.
