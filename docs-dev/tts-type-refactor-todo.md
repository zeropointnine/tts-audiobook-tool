# TTS Type Refactor TODO

> Historical design notes, not a description of current selection behavior or an up-to-date implementation checklist. Process-mode and model-selection guidance has moved to [TTS model selection rules](<tts-model-selection.md>); refer to that document for the current rules. The remaining material records earlier catalog/refactor ideas and may reference removed APIs, including historical model-named class attributes. Current code uses strict `TtsModelType.require_by_id("none")` for a known handle, tolerant `get_by_id(saved_id)` for external IDs, and direct `.id` comparisons for classification; the old uppercase names below are not aliases or supported APIs. Historical `server_*` identities and the decision below to retain them are superseded by the pre-release `<modelname>_<backend>` IDs (`_local`, `_sglomni`, `_audiocpp`); no migration for those undeployed IDs is required.

## Problem

[`TtsModelType`](../tts_audiobook_tool/tts_models/tts_model_type.py) is currently both the full catalog of supported model variants and an implicit catalog of SGL-Omni-backed variants via [`TtsModelSpec.is_sgl_omni`](../tts_audiobook_tool/tts_models/tts_model_type.py:17).

That means the list is doing double duty:

1. Model identity: which TTS variant is selected.
2. Backend grouping: whether that variant is local inference or SGL-Omni-backed.

This distinction leaks into callers such as:

- [`Tts.is_sgl_mode()`](../tts_audiobook_tool/tts.py:189)
- [`Tts.is_local_model()`](../tts_audiobook_tool/tts.py:185)
- [`Tts.init_local_model_type()`](../tts_audiobook_tool/tts.py:116)
- [`OptionsMenu.sgl_omni_type_menu()`](../tts_audiobook_tool/menus/options_menu.py:306)
- [`Prefs.sgl_omni_type`](../tts_audiobook_tool/prefs.py:471)
- generation UI branches in [`generate_menu.py`](../tts_audiobook_tool/menus/generate_menu.py)

The core smell is that SGL-Omni is a backend category, not really a model category.

**Historical, superseded identity decision:** At the time of these notes, the SGL-backed variants already carried a `server_*` id prefix in the model identity (`server_fish_s2`, `server_higgs_v3`, `server_moss_delay`, `server_moss_local`, `server_qwen3tts`, `server_zonos2`), with a matching `*_server_*.py` module for each. So "server" is already an implicit backend signal living inside model identity. Once an explicit `backend_kind` exists, that prefix becomes redundant *as a signal* — but the `id` values are a serialization contract, persisted to `prefs.json` ([`Prefs`](../tts_audiobook_tool/prefs.py:615)). Decision: keep established IDs stable and provide explicit, context-sensitive compatibility handling whenever a model must split. The former architecture-ambiguous `server_moss` preference migrates to auto-detect, while the same legacy project history stamp migrates to unknown (`NONE`). Let `backend_kind` carry classification; the `*_server_*.py` module names stay for the same reason.

## Process mode and selection

Refer to [TTS model selection rules](<tts-model-selection.md>) for the distinction between process mode, catalog backend, available types, project selection, and active runtime. The earlier diagram and selection-state guidance have been removed rather than maintained as a second authority.

## Recommended direction

Keep [`TtsModelType`](../tts_audiobook_tool/tts_models/tts_model_type.py:68) as the canonical app-level model catalog, but make backend classification explicit.

Avoid immediately splitting [`TtsModelType`](../tts_audiobook_tool/tts_models/tts_model_type.py:68) into separate unrelated enums. The app still needs one canonical selected model identity for serialization, menus, voice settings, project fields, and generation. A full split would likely increase adapter code.

Instead, use one canonical model catalog plus explicit backend classification.

## Backend mode and sentinel probe

Refer to [TTS model selection rules](<tts-model-selection.md>) for the launcher-marker probe, lifetime mode invariant, and mode-scoped availability. The old proposed startup and mode-transition behavior is superseded by that reference.

## Proposed model metadata changes

Add a backend enum, for example:

- [`TtsBackendKind.LOCAL`](../tts_audiobook_tool/tts_models/tts_model_type.py)
- [`TtsBackendKind.SGL_OMNI`](../tts_audiobook_tool/tts_models/tts_model_type.py)

Do **not** add a `TtsBackendKind.NONE` member. "No active TTS" is a *runtime* state, not a way of executing a variant: [`TtsModelType.NONE`](../tts_audiobook_tool/tts_models/tts_model_type.py:74) is a catalog placeholder, not a model with a backend. Give the `NONE` placeholder an explicit sentinel: `backend_kind: Optional[TtsBackendKind]`, set to `None` on the placeholder and documented as *not* a real backend. Do not add a third `UNSET`/`NONE` member to the enum: the placeholder is the only state that has no backend, so `None` on the `Optional` field encodes the three-state space (placeholder / local / SGL-Omni) exactly, and the type system forces every read site to handle it. This keeps "backend kind" (a property of a model variant) separate from "no active TTS" (a runtime state), which is exactly the distinction the runtime section below wants to preserve.

Semantics note: `LOCAL` means "executes in the model's local virtualenv," not "runs on a local device." Some local models do not take a torch device parameter at all (e.g., [`MIRA`](../tts_audiobook_tool/tts_models/tts_model_type.py:370), which has an empty `local_torch_devices` list) and are still local models.

Replace:

- [`TtsModelSpec.is_sgl_omni`](../tts_audiobook_tool/tts_models/tts_model_type.py:17)

With something like:

- [`TtsModelSpec.backend_kind`](../tts_audiobook_tool/tts_models/tts_model_type.py)

Then call sites can ask what backend kind a model uses instead of checking an SGL-specific boolean.

## Proposed SGL-specific metadata changes

[`TtsModelSpec.server_model_id_substring`](../tts_audiobook_tool/tts_models/tts_model_type.py:19) is currently only meaningful for SGL-Omni matching.

Options:

1. Rename it to make the scope explicit, such as [`TtsModelSpec.sgl_omni_model_id_substring`](../tts_audiobook_tool/tts_models/tts_model_type.py).
2. Move it into a nested SGL-specific metadata object, such as [`TtsModelSpec.sgl_omni`](../tts_audiobook_tool/tts_models/tts_model_type.py).
3. Move SGL model-id matching into a registry co-located with [`SglOmniUtil`](../tts_audiobook_tool/app_support/sgl_omni_util.py:17), which already centralizes SGL runtime state (base URL, model id, readiness).

Matching fragility: the current matcher, [`TtsModelType.find_tts_type_using_sgl_omni_model_id()`](../tts_audiobook_tool/tts_models/tts_model_type.py:739), does naive substring matching against short prefixes (`"fish"`, `"higgs"`, `"qwen"`, ...). Because the endpoint serves one model at a time, the realistic risk is not two models exposed simultaneously, but a *single* served model id that contains another variant's prefix:

- `fishaudio/s1-mini` (a different Fish model) matches `"fish"` and resolves to [`FISH_S2_SGLOMNI`](../tts_audiobook_tool/tts_models/tts_model_type.py:200)
- `bosonai/higgs-audio-v2-*` matches `"higgs"` and resolves to [`HIGGS_V3_SGLOMNI`](../tts_audiobook_tool/tts_models/tts_model_type.py:302) — a v2 model treated as v3
- any future `Qwen/...` LLM id matches `"qwen"` and resolves to [`QWEN3TTS_SGLOMNI`](../tts_audiobook_tool/tts_models/tts_model_type.py:600)

That mis-match risk — not just future "dynamic discovery" — is the concrete trigger for graduating to option 3. As a cheap stopgap that does not require the registry, the existing matcher can be made to prefer the *longest* matching substring among variants (first match in enum order as today's de-facto tiebreak) so that a more specific prefix wins.

For a first pass, a rename (option 1) is probably enough.

## Proposed catalog helpers

Replace scattered backend checks with named catalog queries.

The primary API should be the spec field itself, `spec.backend_kind`; class-level helpers are conveniences on top, not the main query.

Potential helpers:

- [`TtsModelType.get_items_by_backend()`](../tts_audiobook_tool/tts_models/tts_model_type.py)
- [`TtsModelType.get_local_items()`](../tts_audiobook_tool/tts_models/tts_model_type.py)
- [`TtsModelType.get_sgl_omni_items()`](../tts_audiobook_tool/tts_models/tts_model_type.py:731)
- [`TtsModelType.is_backend()`](../tts_audiobook_tool/tts_models/tts_model_type.py)
- [`TtsModelType.is_valid_sgl_omni_type()`](../tts_audiobook_tool/tts_models/tts_model_type.py) — predicate for the SGL-type validation that is currently triplicated (see plan step 2)

[`TtsModelType.get_sgl_omni_items()`](../tts_audiobook_tool/tts_models/tts_model_type.py:731) already exists, but it is currently implemented by checking [`TtsModelSpec.is_sgl_omni`](../tts_audiobook_tool/tts_models/tts_model_type.py:17). After the refactor, it should be implemented in terms of [`TtsModelSpec.backend_kind`](../tts_audiobook_tool/tts_models/tts_model_type.py).

## Runtime terminology

Refer to [TTS model selection rules](<tts-model-selection.md>) for current process-mode and runtime-binding semantics. The old predicate truth table and selection-derived mode guidance have been removed.

## Suggested incremental plan

The remaining steps below are historical catalog/refactor notes. Refer to [TTS model selection rules](<tts-model-selection.md>) and its focused test references for the current selection contract; the former preference-override and auto-detection expectations are no longer documented here.

### 1. Add backend classification

- Add [`TtsBackendKind`](../tts_audiobook_tool/tts_models/tts_model_type.py) with `LOCAL` and `SGL_OMNI` members only (no `NONE`).
- Add [`TtsModelSpec.backend_kind`](../tts_audiobook_tool/tts_models/tts_model_type.py).
- Convert local models to [`TtsBackendKind.LOCAL`](../tts_audiobook_tool/tts_models/tts_model_type.py).
- Convert server models to [`TtsBackendKind.SGL_OMNI`](../tts_audiobook_tool/tts_models/tts_model_type.py).
- Give the [`TtsModelType.NONE`](../tts_audiobook_tool/tts_models/tts_model_type.py:74) placeholder an explicit "not a real backend" sentinel.
- For startup mode and selection policy, refer to [TTS model selection rules](<tts-model-selection.md>).
- Keep the `server_*` ids (they are a serialization contract in `prefs.json` — see the Problem section). `backend_kind` carries the classification; the prefix becomes purely cosmetic.

### 2. Replace boolean checks

Replace checks like:

- [`item.value.is_sgl_omni`](../tts_audiobook_tool/tts_models/tts_model_type.py:734)
- [`Tts.get_type().value.is_sgl_omni`](../tts_audiobook_tool/menus/menu_status.py:24)

With explicit backend predicates or helper methods.

Also collapse the SGL-type validation that is currently triplicated. The guard `value == TtsModelType.NONE or not value.value.is_sgl_omni` appears in:

- [`Tts.set_sgl_omni_type()`](../tts_audiobook_tool/tts.py:178)
- the [`Prefs.sgl_omni_type` setter](../tts_audiobook_tool/prefs.py:475)
- the [`Prefs` load path](../tts_audiobook_tool/prefs.py:238)

All three should route through a single catalog predicate such as [`TtsModelType.is_valid_sgl_omni_type()`](../tts_audiobook_tool/tts_models/tts_model_type.py). The predicate itself is a plain boolean; the call sites keep their own fallback behavior: [`Tts.set_sgl_omni_type()`](../tts_audiobook_tool/tts.py:178) and the [`Prefs.sgl_omni_type` setter](../tts_audiobook_tool/prefs.py:475) both just guard and delegate to the same target (the setter's current inline check duplicates the guard — drop it), while the [`Prefs` load path](../tts_audiobook_tool/prefs.py:238) keeps its invalid-value → `None` + dirty-flag semantics.

### 3. Rename SGL-specific fields

- Rename [`TtsModelSpec.server_model_id_substring`](../tts_audiobook_tool/tts_models/tts_model_type.py:19) to something SGL-specific.
- Update [`TtsModelType.find_tts_type_using_sgl_omni_model_id()`](../tts_audiobook_tool/tts_models/tts_model_type.py:739) accordingly.

### 4. Runtime and status behavior

Refer to [TTS model selection rules](<tts-model-selection.md>). Runtime validation, interactive reconciliation, and status wording are specified there, not by this historical checklist.

### 5. Consider a separate SGL registry later

[`SglOmniUtil`](../tts_audiobook_tool/app_support/sgl_omni_util.py:17) already centralizes SGL runtime state (base URL, model id, readiness). The remaining smell is that SGL-specific *matching* metadata (`server_model_id_substring`) lives on the generic `TtsModelSpec` while SGL *runtime* behavior lives in `SglOmniUtil`. The end-state is therefore better framed as co-locating SGL-specific matching metadata with `SglOmniUtil` (e.g., in a registry) than as inventing a new registry from scratch.

Only take that step if SGL-Omni grows features such as:

- model-id match mis-fires (a single served model id containing another variant's prefix — see the matching-fragility note above; the current substring matcher cannot distinguish)
- dynamic discovery
- capabilities from the server
- endpoint-specific readiness checks
- backend-specific model aliases
- richer server model matching

A second, non-SGL trigger exists on the *duplication* axis, not the matching axis: related local/server members (e.g. `MOSS` / `SERVER_MOSS_DELAY` / `SERVER_MOSS_LOCAL`, `QWEN3TTS` / `SERVER_QWEN3TTS`, and `FISH_S2` / `SERVER_FISH_S2`) duplicate behavior knowledge (`default_output_sample_rate`, word-count limits, substitutions, streaming) by copy, and the underlying-model "family" they share is deliberately left untyped by this plan. The hardcoded sample-rate fallbacks in [`SglOmniUtil`](../tts_audiobook_tool/app_support/sgl_omni_util.py) read those catalog values directly. If those copies start diverging in a way that copy-editing cannot keep honest, that is a trigger of its own — for a family-level base spec or pairing metadata, *in addition to* (not instead of) the matching registry above.

Until then, backend classification in the main model catalog should be sufficient.

## Adjacent defects to fix along the way

Independent of the taxonomy, these live in the paths this refactor touches and should be fixed in the same pass:

- [`Tts.get_instance_if_exists()`](../tts_audiobook_tool/tts.py:399): the `SERVER_FISH_S2` slot maps to `Tts._fish_s2` (the *local* instance) instead of `Tts._fish_s2_server` ([tts.py:404](../tts_audiobook_tool/tts.py:404)). This is one of three parallel `MAP` dicts (`get_class`, `get_instance`, `get_instance_if_exists`) that would be worth consolidating into one while the area is already open.
- [`Tts.clear_tts_model()`](../tts_audiobook_tool/tts.py:640): nulls 16 of the 18 instance attributes, missing `_fish_s2_server` and `_moss_server` ([tts.py:644-659](../tts_audiobook_tool/tts.py:644)). Harmless today only because those server models are stateless (`kill()` is a `pass`); it becomes a real leak the moment a server model holds resources.
- The [`TtsModelType.NONE`](../tts_audiobook_tool/tts_models/tts_model_type.py:74) placeholder's `requirements_file_name` is set to `"requirements-sgl-omni.txt"` ([tts_model_type.py:93](../tts_audiobook_tool/tts_models/tts_model_type.py:93), flagged with a `# TODO: address entangled abstractions`). The entanglement is now nameable: one placeholder state serves two meanings that need different hints. The no-model hint at [`start.py:185`](../tts_audiobook_tool/start.py:185) becomes **mode-aware**: in SGL-Omni mode the placeholder's sgl-omni requirements file is *correct* (server not configured → install the marker venv / set the URL); in local mode it is wrong (no recognized TTS model in the venv → point at a local model's requirements file).
- The inline `is_sgl_mode` equivalent at [`menu_status.py:24`](../tts_audiobook_tool/menus/menu_status.py:24) and the heading logic at [`main_menu.py:99`](../tts_audiobook_tool/menus/main_menu.py:99) should move onto the redefined predicate from step 4, so the three-state wording lives in one place.
- The torch-flavor exit at [`start.py:131`](../tts_audiobook_tool/start.py:131) is gated on `Tts.get_type().value.is_sgl_omni` — *selection*-derived. In an SGL-Omni venv with no URL set yet, the type is `NONE` and the "wrong torch flavor" nag fires in a venv that does not need CUDA at all. Gate it on the **backend mode** instead.

## Target outcome

The app should read as:

- model catalog logic lives in [`TtsModelType`](../tts_audiobook_tool/tts_models/tts_model_type.py:68)
- backend classification lives in [`TtsModelSpec.backend_kind`](../tts_audiobook_tool/tts_models/tts_model_type.py)
- SGL-Omni-specific matching lives behind explicitly named SGL helpers
- runtime state methods say exactly what state they test
- `server_*` model ids remain the stable serialization contract; backend meaning comes from `backend_kind`, not from the id or module names
- process-mode and selection policy have one authority: [TTS model selection rules](<tts-model-selection.md>)

This should make the SGL-Omni branch legible without turning [`TtsModelType`](../tts_audiobook_tool/tts_models/tts_model_type.py:68) into an implicit subgrouping mechanism.
