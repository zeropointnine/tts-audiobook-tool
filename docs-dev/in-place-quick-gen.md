# In-place quick TTS generation

The Generate editor (`Q` on a line, or `Q` in the segment info dialog) and the
word-substitutions editor (`Q` on a row) can run a one-off TTS job without
leaving the editor. The job runs in a modal over the live editor; the editor is
never torn down, so scroll position, filter, selection and staged edits all
survive.

Before this existed, a quick job exited the editor, ran a full-screen
`GenerationApp` / `TtsPreviewApp`, then rebuilt the editor and "deep-linked" back
to the selected line. That round trip is gone.

## Pieces

- `textual/worker_session.py` — `WorkerSessionMixin`: the worker-job lifecycle,
  host-agnostic. Submission, the `operation_id`-filtered event drain, console
  plumbing into a `WorkerLogContentArea`, the cancel -> hard-reset ladder, and the
  terminal-summary flow. It needs only the scheduling/query methods that both a
  Textual `App` and a `Screen` provide.
  The hard reset runs on a thread worker and reports back through the owning
  `App` (`call_from_thread`), captured on the UI thread before the thread starts.
- `textual/worker_app.py` — `WorkerTextualApp`: the full-screen host
  (`GenerationApp`, `RealTimePlaybackApp`). Adds bindings, chrome, the find bar and
  `App.exit`.
- `textual/quick_gen_modal.py` — `QuickGenModal` (a `ModalScreen` hosting the same
  mixin), its two-row `QuickGenHeader`, and the job specs
  (`make_generation_job`, `make_preview_job`). `QuickGenResult` is the dismissal
  value.
- `VoiceMenuShared.get_voice_problems` — the console-free voice pre-flight the
  modal uses (`validate_voices` keeps the interactive version for menus).

## Modal behavior

- Header: row 1 is the title plus `Status: ...`; row 2 is the key prompt
  (`[ESC] to cancel`, then the kill-process prompt while a cancel is pending, then
  `[ESC] to close`). No memory or elapsed info. Body: a divider and a live
  `WorkerLog`. No find bar.
- Pre-flight (`collect_preflight_problems`): readiness blockers, voice problems,
  and a busy worker are all collected and listed in red inside the modal. Nothing
  is submitted when any exist.
- A cleanly completed job dismisses the modal immediately. Anything else
  (cancelled, failed, aborted, reset, or completed with word errors / exhausted
  retries) stays open until ESC/ENTER so the output can be reviewed. A running job
  cannot be dismissed; ESC cancels, a second ESC hard-resets.

## Editor integration

- While a modal session is active the editor sets `modal_session_active`. Textual
  resolves *priority* bindings (find, select all, ...) before a modal sees a key,
  so `ContentTextualApp.check_action` disables those while the flag is set.
  Non-priority editor bindings (`q`, `x`, space, ...) stop at the modal
  automatically; the quick-gen actions also refuse to start a second session.
- Generate editor, on modal close: reconcile the project with what the worker
  wrote (`reconcile_generation_state`), reload `original_queued_indices` /
  `staged_queued_indices` from the persisted range, refresh classifications, update
  the rows in place (no rebuild), and preserve the surviving selection and its
  anchor. When the requested line leaves the active filter, the highlight moves
  to a nearby phrase instead of the first row. A produced segment autoplays on
  completion even when its line is no longer visible.
- Word-substitutions editor: the preview returns an in-memory `Sound`, played
  after the modal closes. Nothing is persisted. Replacing a still-playing preview
  stops its old status timer before installing the new one.
- Unexpected editor exits (including command-palette Quit and UI exceptions)
  unmount the modal without a dismissal callback. Its teardown cancels only its
  owned operation, finishes a hard reset off the UI thread, and reconciles any
  generated files before returning to menus. An already-running reset is joined,
  not repeated; normally dismissed sessions leave reconciliation to the editor.

## Known limitations / deferred

These are deliberate simplifications carried over from the old flow. Revisit
later.

- **Delete-first.** The line's existing sound is deleted before the job is
  submitted. A cancelled or failed job therefore leaves the line without audio.
  Keeping the old segment until the new one succeeds would need the worker to skip
  its redundant-segment pruning in regen mode (the "best" segment is the one with
  the fewest word errors, so a worse new result would otherwise be pruned instead
  of the old one) and an exact-filename-collision guard on cleanup.
- **Queue persistence side effect.** Quick gen saves the editor's staged (unsaved)
  queue to `project.json` before submitting, because the worker loads its own
  project from disk and writes the range back. Quick gen is therefore not purely
  "not a Save" with respect to queue edits.
- **No transcript auto-fix.** A voice sample with a missing transcript blocks quick
  gen with a message. The full generation flow still auto-transcribes it.
