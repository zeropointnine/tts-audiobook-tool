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

Domain-specific menus and screens stay with their feature; shared UI stays in **ui**. Mixed-responsibility files have not been split. [Shared fixtures](<../tests/conftest.py>) and helper modules remain at the test root.

## How much to run

1. Find the changed behavior's owner above; inspect the matching test's imports and assertions. If names are ambiguous, search tests for the edited symbol.
2. Start with direct coverage. Add a consumer test **only when its contract changes**—inputs/outputs, stored data, events or user-visible behavior. Do not follow every transitive import.
3. Briefly explain the selection, inspect failures/skips, and stop once appropriate checks pass. Broaden to a directory for a feature-wide change, not automatically for a one-function edit.

A few important boundaries:

- **Voice policy vs audio vs playback:** channel/pan assignment is in **voices**; signal processing is in **audio**. Add relevant cases from the [playback tests](<../tests/ui/test_real_time_playback_app.py>) when streaming application, device fallback or saved playback changes—not for every DSP helper edit.
- **Worker vs screen:** shared event/session behavior starts in **workers**. Add the affected host in **ui** or **conversation** when dispatch, cancellation or terminal-result handling changes. A local widget change does not require every host.
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

Shared autouse fixtures, import/bootstrap machinery or foundational schemas can warrant `./venv-base/bin/python -m pytest tests -q -p no:cacheprovider` after focused checks. State why; it is not a compulsory final step. Documentation-only edits normally need link/path checks, not pytest.

Report the files/nodes, environment, result and relevant limitations. Many tests use fake workers/models/transports/devices; “integration” in a filename does not establish live-server compatibility, real inference, encoded output or audible quality. Scope any real smoke check explicitly. For a suspected flake, capture the failing state and use bounded focused reproduction—not repeated full-suite runs until green.

When adding or moving tests, follow these ownership boundaries and update affected references. Keep this guide directory-level rather than rebuilding an exhaustive file index.
