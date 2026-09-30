# Replace named TTS model attributes with ID lookup

## Objective

Remove enum-style references such as `TtsModelType.CHATTERBOX_LOCAL`. The catalog ID should be the only model-specific name application code needs:

```python
model = TtsModelType.require_by_id("chatterbox_local")
INFO = TtsModelType.require_by_id("chatterbox_local").value
```

Keep canonical `TtsModelType` handle objects and their existing identity, equality, hashing, metadata overlay, registration, and reset behavior. This is an API cleanup, not a rewrite of model execution or project persistence.

## Lookup contract

Provide two deliberately different operations in [tts_model_type.py](<../tts_audiobook_tool/tts_models/tts_model_type.py>):

- **`require_by_id(id)` — strict:** return the registered canonical handle; raise `ValueError` with the offending ID when it is unknown. Use for hardcoded application IDs, model metadata declarations, and registry entries referencing known models. Never construct an unregistered handle to satisfy a lookup.
- **`get_by_id(id)` — tolerant:** retain its existing behavior of returning the canonical `none` placeholder for an unknown ID. Use for saved project selections and external IDs whose absence is an expected condition. Document the fallback explicitly; preserve the original saved string rather than normalizing it to `"none"`.

Strict lookup must return the same object as tolerant lookup for a known ID. Neither lookup should load a model, mutate the catalog, or perform discovery.

## Implementation steps

### 1. Introduce strict lookup

- Add `require_by_id()` before migrating callers.
- Test known IDs, unknown IDs, the placeholder, and canonical object identity.
- Keep `get_by_id()` source-compatible and tolerant; do not silently make existing external-input paths strict.

### 2. Replace model-specific references

Audit application code, tests, developer scripts, and current documentation.

- Replace named attributes used as handles with strict lookup using a literal catalog ID.
- Replace metadata declarations such as `ChatterboxBaseModel.INFO` with strict lookup followed by `.value`.
- Preserve handle-keyed registries and identity-sensitive APIs by supplying canonical lookup results. Do not convert all registry keys to strings as part of this change.
- For simple classification, compare `.id` directly instead of looking up another object unnecessarily.
- Rewrite structural pattern matching to match the ID string; function calls cannot be used as value patterns:

```python
match model.id:
    case "chatterbox_local":
        ...
```

- Remove `NONE` too, rather than retaining a special named-attribute exception. When a placeholder handle is needed, use `require_by_id("none")`; when testing classification, compare `.id` to `"none"`.
- Keep tolerant lookup at project-selection and other genuinely external-input boundaries. Review existing calls individually rather than globally replacing `get_by_id()`.
- Do not introduce a separate collection of Python ID constants or an alias map; either would recreate the second naming layer.

### 3. Simplify catalog installation

After callers have been migrated:

- Remove all per-model `ClassVar` declarations and the now-unused `ClassVar` import.
- Remove the annotation-derived required-handle check and dynamic `setattr()` installation.
- Continue installing every shipped entry in exact TOML order and retaining the initial catalog/spec snapshots for reset.
- Make tolerant fallback use the registered `"none"` entry directly, without referencing a removed class attribute or recursively invoking tolerant lookup.
- Preserve parser validation that requires the placeholder. Strict import-time lookups in metadata declarations and registries should fail clearly if their required models are missing.
- Leave iteration compatibility and unrelated catalog helpers alone.

### 4. Update tests and documentation

Update [catalog tests](<../tests/test_model_catalog.py>) to assert lookup contracts rather than uppercase attributes or annotations:

- Strict and tolerant lookup agree for every shipped model.
- Strict unknown lookup raises; tolerant unknown lookup returns the placeholder without modifying a saved selection.
- Handle identity survives overlays, repeated lookup, startup installation, and reset.
- Extra data-only models require no Python declarations and are available through lookup.
- Catalog order and backend-suffix IDs remain unchanged.
- Removed uppercase attributes, including `NONE`, are no longer installed.

Update current architecture/selection documentation and the [catalog header](<../tts_audiobook_tool/tts_models/model_catalog.toml>) to describe ID lookup, not uppercase handle generation. Keep the header concise.

## Validation

- Search application code, tests, scripts, and current docs for stale named-model attributes, dynamic uppercase access, and annotation-based catalog checks.
- Run Pyright with `venv-base` on the affected application modules.
- Run focused catalog, registry, local/remote dispatch, project-selection, voice-menu, and worker regressions. Do not default to the full suite.
- Verify missing hardcoded lookup IDs produce actionable errors and unknown saved IDs remain preserved.
- Review the diff for unintended metadata, persisted-ID, backend, or execution changes. Stop once appropriate focused validation passes.

## Non-goals and completion criteria

Do not rename catalog IDs, change their order, bump catalog schema v1, alter project schemas, add migrations, rename `TtsModelType`, or redesign backend dispatch.

Complete when application code uses literal IDs plus strict lookup or direct ID comparison, external-input paths retain tolerant behavior, and no model-named class attributes or Python declaration inventory remain.
