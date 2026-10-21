# TTS model selection rules

This is the canonical reference for choosing `Project.tts_model_type`, reconciling it with available models, and displaying TTS status. It describes the current implementation. The edge-case policies identified below include implementation assumptions from the selection-rule discussion; documenting them does not imply they were all explicitly requested.

## Separate concepts

- **Process mode:** local generation or server generation (`TtsRuntimeMode.LOCAL` / `REMOTE_CLIENT`). Fixed for the app's lifetime.
- **Available models:** the distinct supported catalog types discovered in that process mode, independent of the project's selection.
- **Project selection:** `Project.tts_model_type`, a persisted string catalog ID. The placeholder ID is the string `"none"`, not Python `None`.
- **Active runtime:** the model successfully bound for generation. It can be the `"none"` placeholder even while the project retains a real model ID.
- **Exact server entry:** `Project.audio_cpp_model_id`, a persisted opaque audio.cpp inference ID (default `""` for legacy/unselected projects). Together with `tts_model_type` it selects one configured entry. Other backends leave it empty. IDs are scoped to the currently configured server, not checkpoint fingerprints or persisted endpoint identities.
- **Remembered audio.cpp entry:** `Project.last_audio_cpp_model_type` / `last_audio_cpp_model_id`, the most recent complete audio.cpp pair (catalog type + server ID). It survives switching to a local or SGL-Omni model. Invariant: whenever the current pair is a complete audio.cpp selection, the remembered pair equals it. Selection writes go through `Project.set_tts_selection()`, and construction/load seeds it from the current pair (covering projects saved before the fields existed).

A model's backend kind (`LOCAL`, `SGL_OMNI`, or `AUDIO_CPP`) is not another process mode. A server-mode process can discover either supported server backend without becoming a local-mode process.

Every shipped entry in [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) receives a canonical `TtsModelType` handle in exact TOML order, including data-only entries. Lookup uses the stable catalog ID directly; there are no model-named class attributes, Python ID constants, or separate declaration inventory. Catalog IDs remain distinct from server inference IDs.

- **Strict:** `TtsModelType.require_by_id("auk_sglomni")` returns the registered canonical handle or raises `ValueError` naming the unknown ID. Use strict lookup for hardcoded IDs, metadata declarations, and handle-keyed registry entries.
- **Tolerant:** `TtsModelType.get_by_id(saved_id)` returns the canonical handle when known, otherwise the registered `"none"` placeholder. Use it for saved selections and external IDs whose absence is expected; preserve the original string rather than persisting the fallback ID.

Both lookups return the same object for a known ID and do not discover or load models. Metadata overlays retain that handle and update its `.value` spec. For simple classification compare `.id` to a literal (for example `model.id == "none"`); structural dispatch uses `match model.id` with literal string cases.

Canonical shipped real-model IDs use `<modelname>_<backend>`: `_local`, `_sglomni`, or `_audiocpp`; backend values remain `local`, `sgl_omni`, and `audio_cpp`. The `none` placeholder comes first, then local, SGL-Omni, and audio.cpp entries, alphabetically by model name within each backend (AuK before AuK-Flash). Historical storage group names, file tags, upstream names, and legacy project field prefixes are independent of these IDs. The pre-release ID rename requires no migration for undeployed IDs; existing v1/v2 flat-field migration produces the current canonical owners.

## 1. Determine process mode and availability

At startup, the launcher marker `tts_audiobook_tool_remote_client_marker` determines process mode. The historical `tts_audiobook_tool_sgl_omni_marker` is also recognized indefinitely so existing client venvs need no reinstall:

- Both markers absent: **local mode**. Probe local model libraries once. Do not discover models from a remote URL.
- Either marker present: **remote-client (server) mode**. Skip local model probing, even if a local library is also installed. Discover supported types from the configured server.

New installs use [`requirements-remote.txt`](<../requirements-remote.txt>) and [`launcher_markers/remote_client`](<../launcher_markers/remote_client/pyproject.toml>). The legacy marker source has been removed; existing non-editable installs remain supported by their installed module. Both distributions may coexist; they represent one remote-client capability, not two models. Probe errors are handled per marker, so an unreadable new marker does not hide a usable legacy marker. This compatibility supports current code with old venvs; older app releases that only recognize the legacy marker still require that legacy package.

Opening another project, changing the selected model, changing the server URL, and losing server connectivity do not change process mode. Changing modes requires relaunching in the appropriate environment.

Local mode supports at most one installed supported model library. Multiple local libraries are an environment error, rejected at startup; the multiple-model selection rule below applies to remote availability, not a new local multi-model feature.

The availability API is a deduplicated list of catalog types, but the audio.cpp picker exposes every distinct `(catalog type, server ID)` pair. Discovery errors produce an empty available-model list. Normal checks reuse discovery TTL/backoff; opening **Model settings > TTS model** explicitly refreshes discovery. In server mode, entering **Options** also forces discovery before the heading, status, and controls render, bypassing TTL/backoff. This happens once per entry, not on ordinary redraws; the manual refresh action remains available.

## 2. Reconcile the project selection

Before building interactive menu items and when printing the status block, reconcile the selection. **In server mode a nonempty saved server ID is kept while the server offers that exact pair, and while discovery fails (offline, no supported models):** an outage is not evidence against it. When a reachable server lacks the pair (missing, or the ID now serves another catalog type — eg, after changing the server URL), the saved entry is replaced as if the project were unselected: the sole available type is selected (with its entry, if unique), otherwise `"none"`. Another entry of the saved type is never preferred while other models are on offer. The replaced pair stays remembered (below) unless the new selection is itself a complete audio.cpp pair, so returning to the old server restores it when the project ended up `"none"` or on a non-audio.cpp model. The change always queues the model-change notice below, naming exact entries (`Fish Audio Drama 3 (audio.cpp; fish-drama3)`), including a same-type entry change. Binding alone (workers, run starts) never substitutes.

**Restoring the remembered audio.cpp entry** applies otherwise. When the server is audio.cpp, there is no saved server ID, and the selected type is not an audio.cpp type (a local, SGL-Omni, unknown, or `"none"` selection), reconciliation reselects the remembered pair if that exact `(type, server ID)` pair is among the discovered candidates. This takes precedence over the table below. A remembered ID now serving a different catalog type is not adopted. If the remembered pair is absent, the table applies, and the resulting change carries a "remembered entry unavailable" reason for the notice. Local mode and SGL-Omni servers never restore; local reconciliation leaves the remembered pair intact.

For projects without a saved server ID (and no restore), compare the raw saved catalog ID with available model IDs:

| Available distinct types | Saved selection | Action |
| --- | --- | --- |
| Zero (local mode) | Any ID, including `"none"` or an unknown ID | Assign `"none"`; do not prompt. |
| Zero (server mode) | Any ID, including `"none"` or an unknown ID | Preserve the saved ID. See the edge-case policy below. |
| One | Matches the available type | Keep it; do not prompt. |
| One (local mode) | Does not match, including `"none"`, another model, or an unknown ID | Immediately assign the sole available type's ID. If the previous ID was not `"none"`, queue a nonblocking FYI at startup or when another project is loaded; other later changes are silent. |
| One (server mode) | Does not match, including `"none"` or an unknown ID | Immediately assign the sole available type's ID. If the previous ID was not `"none"`, queue a nonblocking FYI for the end of the next complete menu. |
| More than one | Matches one of the available types | Keep it; do not prompt or choose a different type. |
| More than one | Does not match any available type | Immediately assign `"none"`; require selection from **Model settings > TTS model**. If the previous selection was an audio.cpp model, queue a nonblocking unavailable-model FYI for the end of the next complete menu; other previous types stay silent. |

After resolving the catalog type, a unique matching audio.cpp entry is assigned to `audio_cpp_model_id` and saved, including on startup/load of an older project. Multiple matching entries retain the catalog type but leave the server ID empty: binding is blocked until the user selects an entry. Filling in only the server ID does not trigger a model-change FYI. Local reconciliation clears any server ID.

A sole-model mismatch queues this notice using the standard hint formatting only when replacing a saved ID other than `"none"`: in server mode whenever reconciliation changes that selection, and in local mode before the first main menu or while checks for a newly loaded project are pending. A matching saved selection or an initially unselected project does not trigger a model-change notice in either mode.

```text
🔔 FYI
This project was last used with TTS model {old_model_name};
It will now use the sole active audio.cpp model, {current_name}.
```

Use `sole active audio.cpp model` only for audio.cpp, which can have more than one model online at a time — the notice says the single available one was chosen. Local mode and SGL-Omni say `active model` instead because they support only one model at a time.

The names are the models' UI `proper_name` values, each qualified by its backend kind (`Chatterbox TTS (local)`, `Breeze TTS 2 (audio.cpp)`). The old name describes the previous saved selection, not necessarily a loaded runtime. Unknown IDs display as `Unknown model: {raw_id}`; `"none"` is not a previous model and never queues this notice. The current name is the selected catalog model, not the exact server inference ID, except when a saved entry was replaced (both names then include their server IDs).

If runtime binding is unavailable, the second line instead reads `It is now configured to use {current_name}, but the runtime is unavailable (see Model settings).` The selection notification does not imply successful binding or persistence; existing status errors and save-error reporting remain in effect.

When multiple types are available but the saved audio.cpp model is unavailable, clearing the selection to `"none"` queues this notice instead. Previous local, SGL-Omni, unknown, or unselected types do not trigger it:

```text
🔔 FYI
This project was last used with TTS model {old_model_name},
which is currently unavailable.
Select a TTS model in Model settings > TTS model.
```

Changes involving the remembered audio.cpp entry use their own copy and are announced whenever the selection changed, including from `"none"` (then the first line is omitted and the next starts with `This project's` instead of `Its`). The remembered entry is named with its server ID:

```text
🔔 FYI
This project was last used with TTS model Breeze TTS 2 (local);
It will now use its last-used audio.cpp model, Fish Audio Drama 3 (audio.cpp; fish-drama-q8)
```

```text
🔔 FYI
This project was last used with TTS model Breeze TTS 2 (local).
Its last-used audio.cpp model, Fish Audio Drama 3 (audio.cpp; fish-drama-q8),
is currently unavailable.
Select a TTS model in Model settings > TTS model.
```

```text
🔔 FYI
This project was last used with TTS model Breeze TTS 2 (local).
Its last-used audio.cpp model, Fish Audio Drama 3 (audio.cpp; fish-drama-q8), is unavailable,
so it will now use the sole active audio.cpp model, Chatterbox TTS (audio.cpp)
```

When runtime binding fails, the restore and sole-model variants instead say the project `is now configured to use` the model `but the runtime is unavailable (see Model settings).` The change reason and the remembered pair (captured before reconciliation) travel on `PendingTtsModelChange`.

Reconciliation and status printing do not display or consume the FYI. In server mode it is printed once at the bottom of the next complete menu. In local mode it is printed once at the bottom of the first main menu, or the next complete menu after a runtime project load; earlier startup submenus leave it pending, and other subsequent local changes do not queue a notice. Rendering happens through the shared `MenuStatus.show_pending_project_hints()` function after the existing `on_shown` callback and before normal menu input. The menu loop captures whether this is the first main menu before `on_shown` marks startup complete. The notices use `hints.print_hint()` directly: no Enter prompt, animation, or persisted hint preference. Heading-only prompt screens leave them pending.

The shared function also checks voice-clone sample durations once per project load (startup or runtime) and once per explicit **Model settings > TTS model** selection, independently of whether the model changed. This check waits for the first main menu at startup, or the next complete menu after a runtime load or model selection. It uses the reconciled selection's optional `ui.voice_sample_max_duration_s` recommendation and the model-supported shared voice list, resolving files through `ProjectVoiceUtil.resolve_voice_file_path()`. Only known, unrounded durations strictly above the recommendation count; unreadable/missing samples and models without a recommendation do not trigger this FYI. It follows any model-change notice:

```text
🔔 FYI
The current model's recommended duration for voice clone samples is 15s,
but there are 2 samples for this project that exceed that value.
```

For one sample, use `there is 1 sample ... that exceeds`. This is warning-only: no sample trimming, rejection, or project mutation. Project assignment arms `State.pending_project_load_checks` for projects with a directory; replacement re-arms it, an explicit model selection re-arms it, reset clears it, and ordinary menu redraws do not repeat the check. Worker project assignment can set the flag but does not render menus or run these checks. An explicit selection never queues the model-change notice: it clears any pending one, and the FYIs re-run for the duration and segmentation checks only.

A third, independent project-load check follows the model-change and voice-duration notices. When the project has imported phrase groups and a selected model, it compares `book.segmentation_settings.max_words_per_segment` (the segmentation-time snapshot, not the current `project.max_words` import setting or actual segment lengths) against `Tts.get_model_support(project).get_max_words_range_reco(project)[1]`. It uses the reconciled model and only warns when that recommendation is positive and the snapshot strictly exceeds it. Missing voice-duration recommendations do not suppress this check. Empty text, an unselected/unknown model, an unset snapshot, and values at or below the recommendation stay silent. It shares the same once-per-load-or-selection timing and never modifies settings or resegments text:

```text
🔔 FYI
This project's text was segmented with a maximum of 80 words per segment,
exceeding the current model's recommended maximum of 60.
Output accuracy on longer prompts may be degraded.
```

Pending old/new IDs live on `State`, not in global or persisted storage. Repeated checks preserve the pending notice. Multiple automatic changes before display coalesce to the first old ID and latest new ID; a net return to the original selection cancels it. Project replacement/reset, explicit model selection, and local reconciliation clearing selection to `"none"` clear the pending notice. Rendering also discards a notice whose new ID no longer matches the selection. Remote reconciliation clearing an unavailable audio.cpp selection to `"none"` queues the unavailable-model notice above; clearing other previous types discards the pending notice.

Automatic changes are saved when the project has a directory, then the project is bound to the runtime. Save errors are reported. A directory-less placeholder project is updated in memory without writing a project file.

`"none"` is an internal state only (new projects with several choices, unknown IDs, and reconciliation clearing an unmatched selection); the model picker does not offer it. It could never persist as a choice anyway: with exactly one model available reconciliation reselects it, and with several it counts as requiring selection.

## 3. New projects, loading, and explicit selection

- **New project:** initialize to the sole available type if there is exactly one; otherwise initialize to `"none"`. Because this is the initial value, it does not separately trigger the mismatch notification.
- **Load / deserialize:** missing selection defaults to `"none"`. Reading or resolving a selection does not itself auto-select. An unknown string ID resolves through tolerant `TtsModelType.get_by_id(saved_id)` to the registered `"none"` handle for metadata access, but the raw string remains intact until interactive reconciliation changes it.
- **Model settings > TTS model:** (server mode only) offer each currently available audio.cpp entry (or distinct type for other backends); there is no None item. When there are no choices (discovery error or no supported models), show the discovery error instead of a menu and leave the selection unchanged. Labels include the server ID and checkpoint filename when available. An explicit selection updates both saved fields, saves the project, and binds it. Choosing a non-audio.cpp type clears the saved server ID. It clears any pending model-change notice and re-arms the duration and segmentation FYI checks for the next complete menu. Subsequent reconciliation still applies the table above.
- **Workers and noninteractive callers:** runtime binding does not perform interactive reconciliation, mutate the saved selection, or ask for input. They validate the selection supplied to them.

Changing the selected ID does not copy or discard other models' settings or the shared project voice list. All models use the same ordered filename/transcript pairs within that project; model-specific clone/transcript capability still controls how they are consumed. For storage ownership and migration, refer to [Project Spec v5](<project-spec-v5.md>).

## 4. Bind selection separately from choosing it

`Tts.bind_project()` validates the current selection without changing `Project.tts_model_type`:

- Local mode requires a selected type available in the local environment.
- Server mode rejects local-only selections. For audio.cpp, bind the **saved exact server ID only if it still matches the saved catalog type**, independent of response order or residency. A missing/incompatible ID blocks binding without fallback (interactive reconciliation replaces it beforehand when the server is reachable; see section 2). With no saved ID, binding accepts only a unique matching entry and does not mutate the project; interactive reconciliation separately persists that unique choice. Multiple matches require selection. SGL-Omni still requires a unique matching entry and advertises exactly one served model.
- None, unknown IDs, missing models, or discovery errors block binding. The active runtime becomes the `"none"` placeholder; the saved project ID is preserved by binding itself.

**Important:** several audio.cpp entries mapping to one catalog type still count as **one available type** in the capability API, but are **separate choices** in the picker. The saved server ID appears in the picker/current label and the bound ID in the main-menu heading. Workers receive the live project selection, including unsaved exact-ID edits, and refresh/bind at command start. Audiobook requests within that command keep using the bound adapter's exact model ID without per-line rediscovery or fallback.

Server IDs are configuration names: renaming one makes the saved selection unavailable; changing the checkpoint behind the same ID changes what will run. Changing the configured server URL can resolve the same saved ID to a different checkpoint. The project stores neither endpoint identity nor a checkpoint fingerprint. Model parameters and voice instructions remain owned by the catalog model, not separately by server entry.

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
- When discovery finds no supported catalog variants, append red **`(server mode; audio.cpp has no supported models)`** (using the discovered server backend). Other discovery errors append their actual message in red. A selected model's server-entry binding error is shown as a terse red qualifier when discovery itself succeeded: `(unavailable)`, `(requires entry selection)`, or `(not on server)` (`TtsBindingIssue.status`); the main-menu heading uses the same text. The full explanation is a one-time FYI instead (below). Other binding issues without a status keep their message.

**One-time binding FYI:** when such a binding issue first appears in server mode, `prepare_tts` queues a notice printed once at the end of the next complete menu (after any model-change notice; no Enter prompt). It is keyed on selection, server URL and issue, so redraws do not repeat it; it re-arms when the issue clears or a project is loaded. Examples:

```text
🔔 FYI
Several server entries match this project's TTS model, Chatterbox TTS (audio.cpp):
one, two.
Select one in Model settings > TTS model.
```

```text
🔔 FYI
This project's TTS model, Fish Audio Drama 3 (audio.cpp; fish-drama3), is not available on this server.
Select a TTS model in Model settings > TTS model.
``` A saved local-only selection instead shows **`(unavailable in server mode)`** when discovery succeeded.
- A known selected model's gray backend qualifier always identifies that model, even when it differs from the connected server. For example, a preserved `moss_local` selection shows **`MOSS-TTS (local) (server mode; audio.cpp has no supported models)`**. None/unknown selections fall back to the discovered backend, or gray **`(server)`** when unidentified. Backend links to the server's model list are omitted when the selected backend differs from the discovered server backend. All of this remains on the single **TTS model** row.

The main menu's **Model settings** label appends red **`(requires: TTS model selection)`** when a project directory is set and the shared selection-required policy is true: an unselected project with multiple choices, a legacy selected audio.cpp type with multiple entries and no saved ID, or a saved exact entry absent from successful discovery. Offline discovery does not prompt replacement.

In server mode, the picker is the first item of **Model settings**, followed by the selected model's controls; the menu opens even when no model is selected (local mode still requires one). Its label, **TTS model**, shows `(currently: requires selection)` when an unselected project has multiple choices. A legacy audio.cpp type with several matching entries retains its model name; the shared policy still opens the picker automatically. Only `requires selection` is red; the surrounding qualifier retains its normal color. Otherwise an unselected project remains `None (unselected)`. Known audio.cpp selections include their saved server ID, even when unavailable.

When that shared selection-required condition holds on entry (`ModelSelectMenu.is_selection_required`), **Model settings** opens the picker immediately via `MenuUtil.menu(initial_hotkey="1")`. The picker is activated before Model settings first renders, so its breadcrumb is `Main > Model >`. After a choice is made, or the user backs out, Model settings renders normally.

## 6. Edge-case policies and their provenance

These distinguish explicit selection requirements from choices made for previously unspecified cases:

| Case | Current policy | Provenance / rationale |
| --- | --- | --- |
| Zero available models in local mode | Clear the saved selection to `"none"` without prompting. | Local availability is established by probing installed libraries; no supported library means no local model can be selected. |
| Zero available models in server mode | Preserve the raw saved selection. | Implementation choice. Offline discovery is not proof that the saved model should be discarded. This also preserves the selection for a reachable server with zero supported models. |
| Server responds but is invalid, unsupported, or returns an HTTP error | Show the actual error, not `server unreachable`. | Interpretation of “unreachable” as a connection failure or timeout. |
| Multiple audio.cpp model entries map to one type | Expose separate choices and persist the exact ID; legacy ambiguous projects require selection. | Stable entry selection must not depend on response order or loaded state. |
| Saved audio.cpp entry disappears or changes type on a reachable server | Replace it as if unselected (sole type, else `"none"`) with a model-change FYI; the old pair stays remembered for a later restore unless replaced by a new audio.cpp pair. Binding alone never substitutes. | Explicit requirement. |
| Server goes offline with a saved entry | Preserve both saved fields and block binding. | Offline discovery is not evidence against the selection. |
| Reconnecting to audio.cpp after a local/SGL-Omni model was used | Restore the remembered exact pair if offered; otherwise apply the ordinary rules with an unavailable-remembered-entry notice. | Explicit requirement. |
| Server backend not yet identifiable | Use gray `(server)`. | Added display fallback beyond the three explicitly requested qualifiers; do not pretend to know the protocol. |

The one-model and multiple-model mismatch rules, startup/project-load local model-change FYI, gray known-backend qualifiers, red selection hint, and red unreachable hint were explicit requirements. Apart from the explicitly requested audio.cpp exact-entry policy, the policies in this table are current behavior, not independently confirmed product decisions.

## Implementation and focused tests

- [TTS facade](<../tts_audiobook_tool/tts.py>): process mode, availability, `reconcile_project_model()`, and `bind_project()`.
- [Menu status](<../tts_audiobook_tool/menus/menu_status.py>): persistence, pending notification capture/display, and status formatting.
- [Menu loop](<../tts_audiobook_tool/menus/menu_util.py>): reconciliation before menu items are built and FYI delivery after `on_shown`.
- [State](<../tts_audiobook_tool/state.py>) and [model picker](<../tts_audiobook_tool/menus/model/model_select_menu.py>): project initialization and explicit selection.
- [Remote discovery](<../tts_audiobook_tool/app_support/remote_tts_discovery.py>): cached observations, backend identification, and discovery errors.
- [Selection tests](<../tests/models/test_project_tts_selection.py>), [status tests](<../tests/ui/test_menu_status.py>), and [remote integration tests](<../tests/models/test_remote_tts_integration.py>): executable references for the rules and binding boundary.
