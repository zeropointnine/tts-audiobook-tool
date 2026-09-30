# TTS model selection rules

This is the canonical reference for choosing `Project.tts_model_type`, reconciling it with available models, and displaying TTS status. It describes the current implementation. The edge-case policies identified below include implementation assumptions from the selection-rule discussion; documenting them does not imply they were all explicitly requested.

## Separate concepts

- **Process mode:** local generation or server generation (`TtsRuntimeMode.LOCAL` / `REMOTE_CLIENT`). Fixed for the app's lifetime.
- **Available models:** the distinct supported catalog types discovered in that process mode, independent of the project's selection.
- **Project selection:** `Project.tts_model_type`, a persisted string catalog ID. The placeholder ID is the string `"none"`, not Python `None`.
- **Active runtime:** the model successfully bound for generation. It can be the `"none"` placeholder even while the project retains a real model ID.
- **Exact server entry:** the server's inference model ID, resolved at binding time. It is ephemeral, not the value stored in `Project.tts_model_type`.

A model's backend kind (`LOCAL`, `SGL_OMNI`, or `AUDIO_CPP`) is not another process mode. A server-mode process can discover either supported server backend without becoming a local-mode process.

Every shipped entry in [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) receives a canonical `TtsModelType` handle in exact TOML order, including data-only entries. Lookup uses the stable catalog ID directly; there are no model-named class attributes, Python ID constants, or separate declaration inventory. Catalog IDs remain distinct from server inference IDs.

- **Strict:** `TtsModelType.require_by_id("auk_sglomni")` returns the registered canonical handle or raises `ValueError` naming the unknown ID. Use strict lookup for hardcoded IDs, metadata declarations, and handle-keyed registry entries.
- **Tolerant:** `TtsModelType.get_by_id(saved_id)` returns the canonical handle when known, otherwise the registered `"none"` placeholder. Use it for saved selections and external IDs whose absence is expected; preserve the original string rather than persisting the fallback ID.

Both lookups return the same object for a known ID and do not discover or load models. Metadata overlays retain that handle and update its `.value` spec. For simple classification compare `.id` to a literal (for example `model.id == "none"`); structural dispatch uses `match model.id` with literal string cases.

Canonical shipped real-model IDs use `<modelname>_<backend>`: `_local`, `_sglomni`, or `_audiocpp`; backend values remain `local`, `sgl_omni`, and `audio_cpp`. The `none` placeholder comes first, then local, SGL-Omni, and audio.cpp entries, alphabetically by model name within each backend (AuK before AuK-Flash). Storage group names, file tags, upstream names, and legacy project field prefixes are independent of these IDs. The pre-release ID rename requires no migration for undeployed IDs; existing v1/v2 flat-field migration produces the current canonical owners.

## 1. Determine process mode and availability

At startup, the launcher marker `tts_audiobook_tool_remote_client_marker` determines process mode. The historical `tts_audiobook_tool_sgl_omni_marker` is also recognized indefinitely so existing client venvs need no reinstall:

- Both markers absent: **local mode**. Probe local model libraries once. Do not discover models from a remote URL.
- Either marker present: **remote-client (server) mode**. Skip local model probing, even if a local library is also installed. Discover supported types from the configured server.

New installs use [`requirements-remote.txt`](<../requirements-remote.txt>) and [`launcher_markers/remote_client`](<../launcher_markers/remote_client/pyproject.toml>). The legacy marker source has been removed; existing non-editable installs remain supported by their installed module. Both distributions may coexist; they represent one remote-client capability, not two models. Probe errors are handled per marker, so an unreadable new marker does not hide a usable legacy marker. This compatibility supports current code with old venvs; older app releases that only recognize the legacy marker still require that legacy package.

Opening another project, changing the selected model, changing the server URL, and losing server connectivity do not change process mode. Changing modes requires relaunching in the appropriate environment.

Local mode supports at most one installed supported model library. Multiple local libraries are an environment error, rejected at startup; the multiple-model selection rule below applies to remote availability, not a new local multi-model feature.

Remote availability is a deduplicated list of catalog types, not a count of server entries. Discovery errors produce an empty available-model list. Normal checks reuse discovery TTL/backoff; opening **Project > TTS model** explicitly refreshes discovery. In server mode, entering **Options** also forces discovery before the heading, status, and controls render, bypassing TTL/backoff. This happens once per entry, not on ordinary redraws; the manual refresh action remains available.

## 2. Reconcile the project selection

Before building interactive menu items and when printing the status block, compare the raw saved ID with available model IDs:

| Available distinct types | Saved selection | Action |
| --- | --- | --- |
| Zero (local mode) | Any ID, including `"none"` or an unknown ID | Assign `"none"`; do not prompt. |
| Zero (server mode) | Any ID, including `"none"` or an unknown ID | Preserve the saved ID. See the edge-case policy below. |
| One | Matches the available type | Keep it; do not prompt. |
| One (local mode) | Does not match, including `"none"`, another model, or an unknown ID | Immediately assign the sole available type's ID. Before the first main menu, queue a nonblocking startup FYI; later changes are silent. |
| One (server mode) | Does not match, including `"none"` or an unknown ID | Immediately assign the sole available type's ID, then queue a nonblocking FYI for the end of the next complete menu. |
| More than one | Matches one of the available types | Keep it; do not prompt or choose a different type. |
| More than one | Does not match any available type | Immediately assign `"none"`; require selection from the Project menu. |

A sole-model mismatch queues this notice using the standard hint formatting: in server mode whenever reconciliation changes the selection, and in local mode only before the first main menu. A matching saved selection does not trigger a notice in either mode.

```text
🔔 FYI
This project was previously using TTS model {old_model_name}.
It will now use the currently active model, {current_name}.
```

The names are the models' UI `proper_name` values, each qualified by its backend kind (`Chatterbox TTS (local)`, `Breeze TTS 2 (audio.cpp)`; a model with no backend, such as `"none"`, is unqualified). The old name describes the previous saved selection, not necessarily a loaded runtime. `"none"` displays as `None (unselected)`; unknown IDs display as `Unknown model: {raw_id}`. The current name is the selected catalog model, not the exact server inference ID.

If runtime binding is unavailable, the second line instead reads `It is now configured to use {current_name}, but the runtime is unavailable (see TTS mode).` The selection notification does not imply successful binding or persistence; existing status errors and save-error reporting remain in effect.

Reconciliation and status printing do not display or consume the FYI. In server mode it is printed once at the bottom of the next complete menu. In local mode it is printed once at the bottom of the first main menu; earlier submenus leave it pending, and subsequent local changes do not queue a notice. Rendering happens after the existing `on_shown` callback and before normal menu input. The menu loop captures whether this is the first main menu before `on_shown` marks startup complete. The notice uses `hints.print_hint()` directly: no Enter prompt, animation, or persisted hint preference. Heading-only prompt screens leave it pending.

Pending old/new IDs live on `State`, not in global or persisted storage. Repeated checks preserve the pending notice. Multiple automatic changes before display coalesce to the first old ID and latest new ID; a net return to the original selection cancels it. Project replacement/reset, explicit model selection, and reconciliation clearing selection to `"none"` clear the pending notice. Rendering also discards a notice whose new ID no longer matches the selection. Clearing an unavailable selection to `"none"` does not itself notify.

Automatic changes are saved when the project has a directory, then the project is bound to the runtime. Save errors are reported. A directory-less placeholder project is updated in memory without writing a project file.

Selecting **None** in the Project menu is not a permanent opt-out: if exactly one model is available, the next reconciliation selects it again. Local mode shows a deferred FYI only at startup's first main menu; later local reselection is silent. Server mode shows a deferred FYI on reselection. With multiple available types, None remains unselected until the user chooses a model.

## 3. New projects, loading, and explicit selection

- **New project:** initialize to the sole available type if there is exactly one; otherwise initialize to `"none"`. Because this is the initial value, it does not separately trigger the mismatch notification.
- **Load / deserialize:** missing selection defaults to `"none"`. Reading or resolving a selection does not itself auto-select. An unknown string ID resolves through tolerant `TtsModelType.get_by_id(saved_id)` to the registered `"none"` handle for metadata access, but the raw string remains intact until interactive reconciliation changes it.
- **Project > TTS model:** offer None plus the currently available distinct types. An explicit selection updates the project, saves it, and binds it. Subsequent reconciliation still applies the table above.
- **Workers and noninteractive callers:** runtime binding does not perform interactive reconciliation, mutate the saved selection, or ask for input. They validate the selection supplied to them.

Changing the selected ID does not copy or discard other models' settings or voice references. For storage ownership and migration, refer to [Project Spec v3](<project-spec-v3.md>).

## 4. Bind selection separately from choosing it

`Tts.bind_project()` validates the current selection without changing `Project.tts_model_type`:

- Local mode requires a selected type available in the local environment.
- Server mode rejects local-only selections and requires exactly one matching server entry for the selected remote type.
- None, unknown IDs, missing models, discovery errors, or multiple matching server entries block binding. The active runtime becomes the `"none"` placeholder; the saved project ID is preserved by binding itself.

**Important:** two server entries mapping to one catalog type count as **one available type** for reconciliation, but **two matching entries** for binding. The sole type can therefore be automatically selected and still fail to bind. There is currently no exact-server-entry selector to resolve this ambiguity; the app does not arbitrarily choose an entry.

Run-start validation forces a fresh remote check. It validates binding rather than silently switching the project model immediately before generation.

## 5. TTS status line

The row label is **TTS model**:

```text
TTS model: Model name (local)
TTS model: Model name (SGL-Omni)
TTS model: Model name (audio.cpp)
```

The parenthesized backend qualifier is gray. Local model-specific text and device/loaded/force-CPU qualifiers may also appear. A local worker's loaded status is shown only when its model ID matches the project selection.

- A valid audio.cpp binding shows **`(audio.cpp, loaded)`** when the exact bound server entry has boolean `loaded: true` in cached discovery metadata. Formatting adds no network request; the qualifier may lag residency changes until normal discovery refreshes. Missing/unknown residency and other backends omit `loaded`.
- In server mode, connection failures and timeouts append red **`(server unreachable)`**, even if no model is selected.
- Other discovery errors append their actual message in red. A selected model's binding error, such as duplicate matching server entries, is also shown in red when discovery itself succeeded.
- The server qualifier uses the discovered backend when known; otherwise a saved remote model can identify its backend. If neither identifies it, the qualifier is gray **`(server)`**.

The main menu's **Project** label appends red **`(requires: TTS model selection)`** when the project's resolved type has `.id == "none"` and at least two distinct model types are available. The suffix is absent with zero or one available type, or when a model is already selected.

In the Project menu, **TTS model** shows `(currently: requires selection)` when the saved selection is `"none"` and at least two distinct types are available. Only `requires selection` is red; the surrounding qualifier retains its normal color. With zero or one available type, it remains `None (unselected)`. Known model names and unknown-ID labels are unchanged.

## 6. Edge-case policies and their provenance

These distinguish explicit selection requirements from choices made for previously unspecified cases:

| Case | Current policy | Provenance / rationale |
| --- | --- | --- |
| Zero available models in local mode | Clear the saved selection to `"none"` without prompting. | Local availability is established by probing installed libraries; no supported library means no local model can be selected. |
| Zero available models in server mode | Preserve the raw saved selection. | Implementation choice. Offline discovery is not proof that the saved model should be discarded. This also preserves the selection for a reachable server with zero supported models. |
| Server responds but is invalid, unsupported, or returns an HTTP error | Show the actual error, not `server unreachable`. | Interpretation of “unreachable” as a connection failure or timeout. |
| Multiple server entries map to one type | Reject ambiguous runtime binding. | Existing behavior retained, not a new automatic-selection rule. |
| Server backend not yet identifiable | Use gray `(server)`. | Added display fallback beyond the three explicitly requested qualifiers; do not pretend to know the protocol. |

The one-model and multiple-model mismatch rules, startup-only local model-change FYI, gray known-backend qualifiers, red selection hint, and red unreachable hint were explicit requirements. The policies in this table are current behavior, not independently confirmed product decisions.

## Implementation and focused tests

- [TTS facade](<../tts_audiobook_tool/tts.py>): process mode, availability, `reconcile_project_model()`, and `bind_project()`.
- [Menu status](<../tts_audiobook_tool/menus/menu_status.py>): persistence, pending notification capture/display, and status formatting.
- [Menu loop](<../tts_audiobook_tool/menus/menu_util.py>): reconciliation before menu items are built and FYI delivery after `on_shown`.
- [State](<../tts_audiobook_tool/state.py>) and [Project menu](<../tts_audiobook_tool/menus/project_menu.py>): project initialization and explicit selection.
- [Remote discovery](<../tts_audiobook_tool/app_support/remote_tts_discovery.py>): cached observations, backend identification, and discovery errors.
- [Selection tests](<../tests/test_project_tts_selection.py>), [status tests](<../tests/test_menu_status.py>), and [remote integration tests](<../tests/test_remote_tts_integration.py>): executable references for the rules and binding boundary.
