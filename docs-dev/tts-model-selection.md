# TTS model selection rules

This is the canonical reference for choosing `Project.tts_model_type`, reconciling it with available models, and displaying TTS status. It describes the current implementation, including active-runtime status and simplified model-selection FYIs. The edge-case policies identified below include implementation assumptions from the selection-rule discussion; documenting them does not imply they were all explicitly requested.

## Separate concepts

- **Process mode:** local generation or server generation (`TtsRuntimeMode.LOCAL` / `REMOTE_CLIENT`). Fixed for the app's lifetime.
- **Available models:** the distinct supported catalog types discovered in that process mode, independent of the project's selection.
- **Project selection:** `Project.tts_model_type`, a persisted string catalog ID. The placeholder ID is the string `"none"`, not Python `None`.
- **Active runtime:** the model successfully bound for generation. It can be the `"none"` placeholder even while the project retains a real model ID. Binding is distinct from loading weights and from job-specific readiness (voices, text, etc.); a bound model need not be loaded yet.
- **Remembered remote selection:** historical remote configuration used for restoration and relevant one-time remote selection notices, not a menu-status row. It is not evidence of connectivity, availability, successful binding, or successful generation. The uniform FYI uses **last used** to describe saved configuration; these selection fields do not separately record successful inference.
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

Before building interactive menu items and when printing the status block, reconcile the selection. **In server mode a nonempty saved server ID is kept while the server offers that exact pair, and while discovery fails (offline, no supported models):** an outage is not evidence against it. When a reachable server lacks the pair (missing, or the ID now serves another catalog type — eg, after changing the server URL), the saved entry is replaced as if the project were unselected: the sole available type is selected (with its entry, if unique), otherwise `"none"`. Another entry of the saved type is never preferred while other models are on offer. The replaced pair stays remembered (below) unless the new selection is itself a complete audio.cpp pair, so returning to the old server restores it when the project ended up `"none"` or on a non-audio.cpp model. The change queues the single saved-model FYI below, including for a same-type entry change; exact entry IDs affect selection/coalescing but are not included in that FYI. Binding alone (workers, run starts) never substitutes.

**Restoring the remembered audio.cpp entry** applies otherwise. When the server is audio.cpp, there is no saved server ID, and the selected type is not an audio.cpp type (a local, SGL-Omni, unknown, or `"none"` selection), reconciliation reselects the remembered pair if that exact `(type, server ID)` pair is among the discovered candidates. This takes precedence over the table below. A remembered ID now serving a different catalog type is not adopted. If the remembered pair is absent, the table applies, and the resulting change retains its "remembered entry unavailable" reason internally but uses the same single-sentence FYI. Local mode and SGL-Omni servers never restore; local reconciliation leaves the remembered pair intact.

For projects without a saved server ID (and no restore), compare the raw saved catalog ID with available model IDs:

| Available distinct types | Saved selection | Action |
| --- | --- | --- |
| Zero (local mode) | Any ID, including `"none"` or an unknown ID | Assign `"none"`; do not prompt. |
| Zero (server mode) | Any ID, including `"none"` or an unknown ID | Preserve the saved ID. See the edge-case policy below. |
| One | Matches the available type | Keep it; do not prompt. |
| One (local mode) | Does not match, including `"none"`, another model, or an unknown ID | Immediately assign the sole available type's ID. If the previous ID was not `"none"`, queue a nonblocking FYI at startup or when another project is loaded; other later changes are silent. |
| One (server mode) | Does not match, including `"none"` or an unknown ID | Immediately assign the sole available type's ID. If the previous ID was not `"none"`, queue a nonblocking FYI for the end of the next complete menu. |
| More than one | Matches one of the available types | Keep it; do not prompt or choose a different type. |
| More than one | Does not match any available type | Immediately assign `"none"`; require selection from **Model settings > TTS model**. If the previous selection was an audio.cpp model, queue a nonblocking saved-model FYI for the end of the next complete menu; other previous types stay silent. |

After resolving the catalog type, a unique matching audio.cpp entry is assigned to `audio_cpp_model_id` and saved, including on startup/load of an older project. Multiple matching entries retain the catalog type but leave the server ID empty: binding is blocked until the user selects an entry. Filling in only the server ID does not trigger a model-change FYI. Local reconciliation clears any server ID.

### Model-selection FYIs

All ordinary model-selection FYIs use exactly one sentence, regardless of process mode, replacement/restoration reason, server connectivity, or binding success:

```text
🔔 FYI
This project was last used with TTS model {old_model} ({backend}).
```

- `{old_model}` is the catalog's `ui.proper_name` for the previous saved model. `{backend}` is exactly `local`, `sgl-omni`, or `audio.cpp` (lowercase). Do not append an exact server entry ID.
- This includes a saved local model in an offline remote process. It uses the same sentence, not a separate mismatch/relaunch explanation.
- When an audio.cpp restoration/unavailable-entry transition starts from `"none"`, use the known remembered audio.cpp model as the historical model. Otherwise use the previous saved selection, including local selections in remote mode.
- If that historical model is unknown or unselected, omit the FYI instead of inventing a model/backend or announcing `None`.
- No follow-up sentence describes the current environment, replacement model, restoration, availability, configuration, or relaunch advice. Status and generation blockers retain current-state/error details separately.
- Ordinary model-history FYIs are shown **at most once per project load**, shared across model-change and binding notices. `State.has_shown_tts_model_hint` is set only when a known-model sentence is actually displayed and is reset on project assignment/load/reset (including reloading the same project), never by URL changes, reconnection, automatic reconciliation, or re-armed voice/segmentation checks. The flag is process-local, not persisted.
- Existing initial queue eligibility, coalescing before display, startup timing, and explicit-selection suppression remain unchanged. A sole-model mismatch can queue a notice for a previous real selection; matching or initially unselected projects do not gain a new notice merely because they were loaded.
- A model-change FYI suppresses an additional ordinary binding FYI for that menu. The sole model-selection wording exception is the ambiguous server-entry notice (section 5.4).

The sentence's `last used` wording refers to saved model configuration, not a newly recorded successful generation. Save-error reporting remains separate.

Reconciliation and status printing do not display or consume the FYI. In server mode it is printed once at the bottom of the next complete menu. In local mode it is printed once at the bottom of the first main menu, or the next complete menu after a runtime project load; earlier startup submenus leave it pending, and other subsequent local changes do not queue a notice. Rendering happens through the shared `MenuStatus.show_pending_project_hints()` function after the existing `on_shown` callback and before normal menu input. The menu loop captures whether this is the first main menu before `on_shown` marks startup complete. The notices use `hints.print_hint()` directly: no Enter prompt, animation, or persisted hint preference. Heading-only prompt screens leave them pending.

The shared function also checks voice-clone sample durations once per project load (startup or runtime) and once per explicit **Model settings > TTS model** selection, independently of whether the model changed. In remote mode with an incompatible saved local selection, both the duration and segmentation recommendations are skipped; local recommendations cannot make this remote process runnable. This check waits for the first main menu at startup, or the next complete menu after a runtime load or model selection. It uses the reconciled selection's optional `ui.voice_sample_max_duration_s` recommendation and the model-supported shared voice list, resolving files through `ProjectVoiceUtil.resolve_voice_file_path()`. Only known, unrounded durations strictly above the recommendation count; unreadable/missing samples and models without a recommendation do not trigger this FYI. It follows any model-change notice:

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

Pending old/new IDs live on `State`, not in global or persisted storage. Repeated checks preserve the pending notice. Multiple automatic changes before display coalesce to the first old ID and latest new ID; a net return to the original selection cancels it. Project replacement/reset, explicit model selection, and local reconciliation clearing selection to `"none"` clear the pending notice. Rendering also discards a notice whose new ID no longer matches the selection. Remote reconciliation clearing an unavailable audio.cpp selection to `"none"` queues the same single-sentence saved-model notice; clearing other previous types discards the pending notice.

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

## 5. TTS status and messaging

### 5.1 Status describes the current runtime, not remembered configuration

The **TTS model** row describes the active runtime, not merely `Project.tts_model_type`. A model name is current only when binding succeeds in the launched process mode. Binding does not imply that weights are loaded or that all job-specific readiness checks pass.

Process mode comes from the launched environment and remains fixed. Neither a saved selection, a model's backend qualifier, nor server connectivity changes it. Never silently fall back from remote generation to local generation.

| Situation | Primary status example |
| --- | --- |
| Local mode, valid local binding | `Breeze TTS 2 (local)` |
| Local mode, no supported local model | `None active (local mode; no supported model installed)` |
| Remote mode, valid remote binding | `FireRedTTS3 (audio.cpp)` or `AuK (SGL-Omni)` |
| Remote server unreachable or timed out | `None active (remote mode; server unreachable)` |
| Remote server reachable, model/entry choice required | `None active (remote mode; requires selection)` |
| Remote server reachable, selected entry unavailable | `None active (remote mode; selected entry unavailable)` |
| Remote server reachable, no supported models | `None active (remote mode; no supported models)` |
| Remote mode, local-only saved selection still blocks binding after reconciliation | `None active (remote mode; local model unavailable)` |

Normal interactive reconciliation still runs before rendering. Some failure states are therefore replaced by a valid sole-model binding before the status is shown. The table does not introduce new selection rules or disable that reconciliation.

Other discovery/configuration errors retain their specific explanation with `None active (remote mode; ...)`; do not label HTTP errors, unsupported protocols, or invalid responses as unreachable. Unknown saved IDs never appear as active models. Local binding errors other than a missing installation also retain their actual explanation with an explicit local-mode context.

Use the same active-versus-remembered distinction in the main-menu heading and other generation-facing summaries. A backend qualifier on a valid binding identifies how that model runs; when no model binds, explicitly identify the process mode instead of attaching an error to an incompatible model's name.

### 5.2 Saved local selections are not remote runtime status

In remote mode:

- Never display a saved local model as the current model or a last-used fallback.
- Retained local model settings, voice samples, or LoRAs do not imply that local generation is available. For a saved local selection, the **Voices** summary shows only `1 saved voice` / `N saved voices`, not local-only LoRA/preset details.
- Do not expose local model settings as controls for the current remote runtime merely because the saved ID is local. Preserve those settings for a later local launch.
- The primary row identifies remote mode and any outage; the generic saved-model FYI names the saved local model. Generation blockers, not additional FYI paragraphs, explain the environment mismatch.

This misleading presentation must no longer occur:

```text
TTS model: Breeze TTS 2 (local) (server unreachable)
```

Regardless of remembered remote history, the replacement is simply:

```text
TTS model: None active (remote mode; server unreachable)
```

### 5.3 No remembered-model row in menu status

Do not display a **Last-selected remote model** row, in any process mode or connectivity state. The status block describes the current runtime only; neither a saved remote selection nor the remembered audio.cpp pair adds a historical row.

Remembered remote configuration remains available for restoration and the single-sentence saved-model FYI. That FYI includes only the catalog model name and lowercase backend, not a server entry ID or a claim about current availability. Saved server IDs remain configuration, not endpoint identities or checkpoint fingerprints.

An outage must not erase saved selections to make the display simpler. Reconnection revalidates the saved entry and applies the existing restoration/reconciliation rules. Local reconciliation continues to select the installed local model without deleting the remembered audio.cpp pair. No new general remote-history persistence is implied by this display adjustment.

### 5.4 Sole model-selection FYI exception: ambiguous server entries

Ordinary model changes and binding failures use only the single sentence in section 2. Remove the former environment/replacement/restoration/unavailable/relaunch FYI paragraphs. The ambiguous server-entry notice alone retains its existing wording and entry list:

```text
🔔 FYI
Several server entries match this project's TTS model, {model}:
{server_ids}.
Select one in Model settings > TTS model.
```

Notices remain nonblocking and deferred to the end of a complete menu. Ordinary model-history notices share the once-per-project-load guard in section 2, regardless of which path queued them. In particular, displaying the saved Breeze/local sentence while offline consumes that load's ordinary FYI; fixing the server URL and successfully selecting a remote model updates status without repeating the sentence. Only ambiguous-entry notices are keyed on selection, server URL, and issue and re-arm when the issue clears or a project is loaded. Showing ordinary history neither consumes nor disables ambiguity notices, which can still appear or reappear independently.

Persistent status and generation blockers retain current-state/error details. Independent voice-duration and segmentation FYIs are unaffected by this model-selection wording simplification.

### 5.5 Generation blockers and model controls agree with status

- Offline remote mode explains that the server is unreachable, even with an unselected or unknown saved ID; a saved local selection additionally explains the environment mismatch where relevant. Cached model/binding/discovery messages are deduplicated. Run starts force discovery even when binding would reject the selection before checking the server.
- A reachable server requiring a model/entry choice directs the user to **Model settings > TTS model**.
- A local-only selection in remote mode directs the user to relaunch in the appropriate local environment if local generation is intended. Do not imply that loading local weights in the remote process will fix it.
- Metadata-only settings access may retain the saved selection, but generation-facing controls and summaries must not represent an incompatible selection as the active model.
- Continue validating at run start. These rules do not authorize model substitution immediately before generation.

### 5.6 Retained qualifiers and scope

Backend qualifiers remain gray and failure explanations red. Local model-specific text and device/loaded/force-CPU qualifiers may appear only for the corresponding valid local binding. A local worker's loaded status must match the selected and bound model.

A valid audio.cpp binding may show `(audio.cpp, loaded)` when the exact bound server entry has boolean `loaded: true` in cached discovery metadata, plus the existing checkpoint filename qualifier when available. Formatting adds no network request; residency may lag until normal discovery refresh. Missing/unknown residency and other remote backends omit `loaded`. Links to a server model list must refer to that server/backend, not to an incompatible saved local model.

This adjustment concerns presentation and explanation. It does not discard project settings or LoRAs, change process mode, change the selection/binding boundary, introduce local fallback, or establish generation success. Existing selection, persistence, exact-entry restoration, and menu-picker behavior remain in force.

### 5.7 Model picker behavior (existing implementation)

The main menu's **Model settings** label appends red **`(requires: TTS model selection)`** when a project directory is set and the shared selection-required policy is true: an unselected project with multiple choices, a legacy selected audio.cpp type with multiple entries and no saved ID, or a saved exact entry absent from successful discovery. Offline discovery does not prompt replacement.

In server mode, the picker is the first item of **Model settings** only when multiple selectable entries are available, followed by the selected model's controls; the menu opens even when no model is selected (local mode still requires one). Multiple audio.cpp entries of the same catalog type count as separate choices. With zero or one choice (including SGL-Omni's sole served model), the picker and its separating blank line are omitted; the existing TTS status line still identifies the model. Its label, **TTS model (N available)**, shows `(currently: requires selection)` when an unselected project has multiple choices. A legacy audio.cpp type with several matching entries retains its model name; the shared policy still opens the picker automatically. Only `requires selection` is red; the surrounding qualifier retains its normal color. Otherwise an unselected project remains `None (unselected)`. Known audio.cpp selections include their saved server ID, even when unavailable.

When multiple selectable entries exist and that shared selection-required condition holds on entry (`ModelSelectMenu.is_selection_required`), **Model settings** opens the picker immediately via `MenuUtil.menu(initial_hotkey="1")`. The picker is activated before Model settings first renders, so its breadcrumb is `Main > Model >`. After a choice is made, or the user backs out, Model settings renders normally.

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
| Reconnecting to audio.cpp after a local/SGL-Omni model was used | Restore the remembered exact pair if offered; otherwise apply the ordinary rules with the same single-sentence saved-model FYI. | Explicit requirement. |
| Server backend not yet identifiable | Use explicit `remote mode` when no model binds, without guessing the server protocol. | Explicit adjusted UX requirement. |
| Remote mode retains a local selection while the server is offline | Preserve configuration, but show `None active (remote mode; server unreachable)` and use the ordinary single-sentence saved-model FYI; never present the local model as current. | Explicit adjusted UX requirement. |
| No active remote binding, but a remote selection is remembered | Keep remembered configuration for restoration/notices, but do not add a history row to the menu-status block. | Explicit adjusted UX requirement. |

The one-model and multiple-model mismatch rules, startup/project-load local model-change FYI, gray known-backend qualifiers, red selection hint, and red unreachable hint were explicit requirements. Section 5's adjusted presentation rules are also explicit requirements. Other selection-policy assumptions retain the provenance above; this UX adjustment does not change their behavior.

## Implementation and focused tests

- [TTS facade](<../tts_audiobook_tool/tts.py>): process mode, availability, `reconcile_project_model()`, and `bind_project()`.
- [Menu status](<../tts_audiobook_tool/menus/menu_status.py>): persistence, pending notification capture/display, and status formatting.
- [Menu loop](<../tts_audiobook_tool/menus/menu_util.py>): reconciliation before menu items are built and FYI delivery after `on_shown`.
- [State](<../tts_audiobook_tool/state.py>) and [model picker](<../tts_audiobook_tool/menus/model/model_select_menu.py>): project initialization and explicit selection.
- [Remote discovery](<../tts_audiobook_tool/app_support/remote_tts_discovery.py>): cached observations, backend identification, and discovery errors.
- [Selection tests](<../tests/models/test_project_tts_selection.py>), [status tests](<../tests/ui/test_menu_status.py>), and [remote integration tests](<../tests/models/test_remote_tts_integration.py>): executable references for selection, status, and the binding boundary.
- [Runtime-status lifecycle tests](<../tests/ui/test_tts_runtime_status.py>): offline remote mode with saved Breeze/LoRA settings, absence of historical status rows, one-time/rearmed saved-model notices, reconnection, and local relaunch without remote discovery or deletion of remembered state. These use real selection/binding logic with fake discovery/worker observations, not live inference.
- [Readiness tests](<../tests/ui/test_readiness.py>), [model-menu tests](<../tests/models/test_model_menu_shared.py>), and [voice-menu tests](<../tests/voices/test_voice_menu_shared.py>): incompatible-control suppression and cached/forced discovery blockers, including unselected/unknown IDs and deduplication. No fabricated successful-use history is used to justify a `last good` label.
