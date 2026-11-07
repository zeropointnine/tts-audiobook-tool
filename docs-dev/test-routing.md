# Choosing tests for a change

Tests are grouped by their **primary behavioral owner**, not strictly by source directory or test technique. Start with the relevant file or test node; a directory is a useful search boundary, not a mandatory test bundle.

## Where to look

| Directory | Owns |
|---|---|
| [text](<../tests/text/>) | EPUB import, segmentation, headings, normalization, text equivalence and edit-session logic. |
| [project](<../tests/project/>) | Book/project/preferences persistence, migrations, paths, transfers, staging and export metadata. |
| [models](<../tests/models/>) | Catalog/settings, model selection, backend discovery/adapters, local-model caches, model menus and environment/bootstrap checks. |
| [workers](<../tests/workers/>) | Worker IPC/runtime roles, events, session controls/logs/Find, generation retries/timeouts/validation and sleep inhibition. |
| [voices](<../tests/voices/>) | Voice references, selection, preflight, sample actions/cropping, voice menus, pan assignment and spatial policy. |
| [audio](<../tests/audio/>) | DSP, silence/gaps/pauses, reverb/room tone, concat/export, device streams and sample playback. |
| [ui](<../tests/ui/>) | Shared widgets/editors/dialogs, general console menus, headers, generation/playback screens and browser-player tests. |
| [conversation](<../tests/conversation/>) | Chat/LLM/response sessions, microphone transcription, startup and conversation-specific UI. |
| [enhance](<../tests/enhance/>) | Enhancement text/alignment/artifacts, orchestration and feature-specific UI. |
| [speaker_assign](<../tests/speaker_assign/>) | Speaker assignment: character roster, dialog utterances, chunking, LLM reply parsing and speaker-to-voice mapping. |

Domain-specific menus and screens stay with their feature; shared UI stays in **ui**. Mixed-responsibility files have not been split. [Shared fixtures](<../tests/conftest.py>) and helper modules remain at the test root.

## How much to run

1. Find the changed behavior's owner above; inspect the matching test's imports and assertions. If names are ambiguous, search tests for the edited symbol.
2. Start with direct coverage. Add a consumer test **only when its contract changes**—inputs/outputs, stored data, events or user-visible behavior. Do not follow every transitive import.
3. Briefly explain the selection, inspect failures/skips, and stop once appropriate checks pass. Broaden to a directory for a feature-wide change, not automatically for a one-function edit.

A few important boundaries:

- **Voice policy vs audio vs playback:** channel/pan assignment is in **voices**; signal processing is in **audio**. Add relevant cases from the [playback tests](<../tests/ui/test_real_time_playback_app.py>) when streaming application, device fallback or saved playback changes—not for every DSP helper edit.
- **Worker vs screen:** shared event/session behavior starts in **workers**. Add the affected host in **ui** or **conversation** when dispatch, cancellation or terminal-result handling changes. A local widget change does not require every host.
- **Audio file I/O:** writers, decoder and conversion tests are in **audio** and run under the no-FFmpeg guard (`pytest.mark.no_ffmpeg_exe`; `--no-ffmpeg-exe` applies it to every test). For changes to that stack, its callers, or audio dependencies, also run the end-to-end [no-FFmpeg smoke script](<../testx/no_ffmpeg_smoke.py>) (`./venv-base/bin/python -m testx.no_ffmpeg_smoke`); see [audio I/O](<audio-io.md>).
- **Persistence:** schemas and atomic writes start in **project**, model-setting ownership in **models**. Add the affected import/editor/generation flow only if the stored-data contract reaches it.

## Commands and environments

Run from the repository root, using `venv-base` by default:

```bash
# Focused rule coverage.
./venv-base/bin/python -m pytest -q -p no:cacheprovider tests/voices/test_voice_spatial_policy.py

# Preview a feature's tests without executing them.
./venv-base/bin/python -m pytest --collect-only -q tests/text
```

Use `path/to/test_file.py::test_name` for a narrower case, or pass a directory for a justified feature-wide check. Preserve all relevant boundary cases when narrowing parameterized tests.

Browser-player JavaScript tests also live in **ui**. Run them with `npm run test:browser`; pytest does not collect them.

Local-model library/cache tests may require the environment in the [venv-to-model mapping](<../.agents/venv-local-models.md>):

```bash
./venv-cb/bin/python -m pytest -q -rs -p no:cacheprovider tests/models/test_chatterbox_voice_clone_cache.py
```

Inspect skips with `-rs`: skipped/zero selected tests are not coverage. A missing pytest runner or collection/import failure is an environment limitation, not a pass; report it without silently installing dependencies. A full base-environment run does not replace affected model-environment checks.

## When broad validation is justified

Shared autouse fixtures, import/bootstrap machinery or foundational schemas can warrant a full run after focused checks. State why; it is not a compulsory final step. Documentation-only edits normally need link/path checks, not pytest.

Run the full suite in parallel with [pytest-xdist](https://pytest-xdist.readthedocs.io/) (installed in `venv-base`):

```bash
./venv-base/bin/python -m pytest tests -q -p no:cacheprovider -n auto --dist worksteal
```

This takes about 40 s, compared with about 220 s serially. Each worker is a separate process, so class-level state and the model-worker singleton stay per-process. Use `-n` where the selection is slow per test: the full suite, `tests/ui` (about 106 s → 19 s), `tests/workers`, or a large Textual file (eg `test_quick_gen_modal.py`, 9 s → 5 s). Skip it for fast logic tests such as `tests/text`, where worker startup cancels out the gain. Drop `-n` (or pass `-n 0`) to use `--pdb`/breakpoints or to check whether a failure only occurs in parallel. The suite's real-subprocess and real-timeout tests (`test_model_worker`, `test_gen_timeout`, `test_loaded_ffmpeg_libs`) are the most load-sensitive. Model venvs don't need xdist; their runs are small.

Report the files/nodes, environment, result and relevant limitations. Many tests use fake workers/models/transports/devices; “integration” in a filename does not establish live-server compatibility, real inference, encoded output or audible quality. Scope any real smoke check explicitly. For a suspected flake, capture the failing state and use bounded focused reproduction—not repeated full-suite runs until green.

When adding or moving tests, follow these ownership boundaries and update affected references. Keep this guide directory-level rather than rebuilding an exhaustive file index.

## Writing tests

- **Parametrize only over axes the code branches on.** Don't stack `parametrize` decorators into a Cartesian product. List the cases that reach distinct branches, plus boundary values. Independent axes can rotate through each other's values (see the path styles in [qwen3 selection](<../tests/models/test_qwen3_audio_cpp_selection.py>)) rather than multiply. Don't parametrize over models or backends that share one code path; one representative is enough. Only enumerate models whose adapter or support code actually differs.
- **Derive expected values from the loaded catalog**, eg `TtsModelType.require_by_id(model_id).value.max_random_seed`, rather than retyping `model_catalog.toml` values. A test that mirrors shipped data breaks on every catalog edit and protects nothing that parsing tests don't.
- **Test shared behavior once, in its owner.** Add one wiring test per consumer, not a copy of the owner's cases. For example, the worker-session ESC/cancel/hard-reset sequence lives in [worker session keys](<../tests/workers/test_worker_session_keys.py>), parametrized over the generation and realtime apps. Host files test only host-specific behavior.
- **Wait for a condition, not a duration.** In Textual tests, use `wait_until(pilot, predicate)` from [textual_editor_stubs](<../tests/textual_editor_stubs.py>) for state reached through timers, threads or worker polling. Keep a fixed pause only to show that something does *not* happen.
- **Don't pin UI wording or styling** (exact paragraphs, colours, ANSI codes, label positions) unless the text is the behavior being tested.
- **When reducing cases, check coverage.** A before/after comparison with `coverage run --branch` should show no lost executed lines or branch arcs. Keep value boundaries even where coverage alone wouldn't notice them.
