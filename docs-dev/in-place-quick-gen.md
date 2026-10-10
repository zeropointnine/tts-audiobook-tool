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
  mixin), its two-row `QuickGenHeader`, the review panel, and the job specs
  (`make_generation_job`, `make_preview_job`). `QuickGenResult` is the dismissal
  value.
- `project_support/segment_staging_util.py` — the `segments_temp/` staging area
  for reviewable output: per-job directory paths, pruning to the best take,
  accepting a take into `segments/`, and housekeeping deletion.
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
- A cleanly completed preview dismisses the modal immediately. A line
  regeneration never auto-dismisses. Anything without a staged candidate
  (cancelled, failed, aborted, reset) stays open until ESC/ENTER so the output can
  be reviewed. A running job cannot be dismissed; ESC cancels, a second ESC
  hard-resets.

## Staged regeneration and review

A line regeneration is reviewed before it replaces anything:

- `make_generation_job` picks a per-job directory
  `{project}/segments_temp/<uuid>/` and passes it as `GenerateCommand.staging_dir`.
  With a staging dir, `GenerateUtil.generate_files` saves the `.flac` and its STT
  sidecar there under their normal segment names, prunes each line's retries to
  its best take *within the staging dir* (per line, ranked like the project
  catalog; `segments/` is never pruned), and does not update or
  save `generate_range_string`. The project is untouched while the job runs; the
  line's old audio stays in place.
- When a *completed* job leaves a staged take (clean or with word errors), the
  modal shows a review panel below the log (divider + `[1] Play original sound`,
  `[2] Play new sound`, each with its word-error count read from the file name,
  then `[ENTER]` keep / `[ESC]` discard). The new sound autoplays; 1/2 switch
  between sounds and toggle the playing one off. The header's prompt row is blank
  while the panel carries the prompt.
- The modal only reports the decision (`QuickGenResult.staged_path`, `accepted`).
  The editor applies it: on keep, `accept_staged_segment` moves the take and its
  sidecar into `segments/` (`os.replace`, same volume). Normal segment names have
  no timestamp, so the new take can have exactly the old take's name; such an old
  pair is first moved aside into the staging dir, and any failure rolls every move
  back, so the audio/sidecar pair in `segments/` is entirely old or entirely new.
  Only on success are the line's other catalog segments deleted and the line
  removed from the editor's staged queue (in memory; queue edits are saved on exit
  as usual). A failed keep is reported and leaves the line as it was. The job's
  staging directory is deleted on every outcome.
- Debug sound files (`save_debug_files` pref) are not staged; they still go to
  `segments/` and are not removed on discard.
- Housekeeping: `GenerateMenu.run_editor` deletes the whole `segments_temp/` after
  the editor exits (normally or not). By then the modal's teardown has cancelled
  or reset any owned job, so nothing is still writing there. Quitting the editor
  while a review is pending is a discard.
- The worker log's "Saved: ..." link points at the staging path, which is gone
  after the decision.

## Editor integration

- While a modal session is active the editor sets `modal_session_active`. Textual
  resolves *priority* bindings (find, select all, ...) before a modal sees a key,
  so `ContentTextualApp.check_action` disables those while the flag is set.
  Non-priority editor bindings (`q`, `x`, space, ...) stop at the modal
  automatically; the quick-gen actions also refuse to start a second session.
- Generate editor, on modal close with a kept take: accept it (see above),
  refresh classifications, update the rows in place (no rebuild), and preserve
  the surviving selection and its anchor. When the requested line leaves the
  active filter, the highlight moves to a nearby phrase instead of the first row.
  Nothing autoplays after the modal closes; the review panel already played it.
- Word-substitutions editor: the preview returns an in-memory `Sound`, played
  after the modal closes. Nothing is persisted. Replacing a still-playing preview
  stops its old status timer before installing the new one.
- Unexpected editor exits (including command-palette Quit and UI exceptions)
  unmount the modal without a dismissal callback. Its teardown cancels only its
  owned operation and finishes a hard reset off the UI thread before returning to
  menus (a job's `reconcile` hook, if any, runs then; staged jobs have none). An
  already-running reset is joined, not repeated.

## Known limitations / deferred

- **No transcript auto-fix.** A voice sample with a missing transcript blocks quick
  gen with a message. The full generation flow still auto-transcribes it.
