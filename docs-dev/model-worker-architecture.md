# Model Worker Architecture

Last updated: 2026-09-29

## Scope

This document describes the long-lived **model worker** process: why it exists, how the main process talks to it, which code and state belong to it, and the rules to follow when changing anything on either side of that boundary. TTS generation is the dominant workload and gets the most attention here, but the same mechanism serves every heavyweight inference the app performs.

Primary files:

- `tts_audiobook_tool/model_worker.py` — worker entry point, per-command handlers, and the `ModelWorker` manager used by the main process
- `tts_audiobook_tool/model_worker_protocol.py` — the typed command and event dataclasses
- `tts_audiobook_tool/model_runtime.py` — process-role tracking and the model-ownership guard
- `tts_audiobook_tool/worker_reset.py` — hard-reset causes, requests, and outcomes
- `tts_audiobook_tool/generation_events.py`, `tts_audiobook_tool/gen_timeout_util.py` — in-worker progress events and the hang watchdog
- `tts_audiobook_tool/textual/worker_app.py`, `tts_audiobook_tool/textual/worker_content.py` — the shared full-screen session shell and worker log

Related: [tts-model-architecture.md](tts-model-architecture.md) (model and venv layering), [sound-segment-and-regen-architecture.md](sound-segment-and-regen-architecture.md) (the on-disk artifacts the worker produces).

## Why the worker exists

Before the worker (everything up to and including `e46ad10`, 2026-08-22) one process did it all: menus, editors, TTS, STT, YAMNet, LavaSR, and the PortAudio playback stream. That is simple, and it kept the inference code trivially synchronous. The app paid for the simplicity in ways that could not be fixed inside one process:

- **Inference owned the thread that owned the UI.** A generate run is one long blocking loop (`GenerateUtil.generate_files()`). While it ran there was no event loop left to render a header, respond to keys, or draw anything except what the inference loop itself printed. A richer progress UI had to be smuggled into the inference loop, which is exactly where it does not belong.
- **A wedged inference wedged the app.** Native runtimes occasionally stop returning — the `gen_timeout_util.py` docstring names `dots.tts` specifically. In one process, a call that never returns also prevents any watchdog from being *noticed*, because the detection loop shares the wedged thread. The only control left was Ctrl-C, which cooperative `Interrupts().did_interrupt` checks cannot honor if control never returns to one. The user lost the app, the loaded project, and any unsaved edits.
- **Model memory could not be reliably freed.** Model libraries retain references, monkeypatch globals, start threads, and hold CUDA allocator caches. Dropping our own references could still leave VRAM allocated, so `Options > Unload models` was best-effort. Process exit is the only cleanup boundary that reliably releases driver allocations, native state, and library global state.
- **A crash in third-party model code was an app crash.** A segfault or abort inside a model library took the menu loop and project state down with it.
- **Model global state leaked into the interactive process** permanently: atexit handlers, signal handlers, CUDA contexts, import-time patches.

One long-lived child process buys:

1. **A responsive UI.** The main process keeps its event loop; worker output arrives as events. The full-screen Textual sessions (Generate, realtime playback, TTS preview) exist because nothing outside the UI writes to the terminal during a job.
2. **A deterministic unload/reset boundary.** Unloading models and recovering from a bad model state are process lifecycle operations, not reference juggling.
3. **Hang recovery without losing the app.** A watchdog inside the worker plus parent-side timeouts can terminate and restart the worker while the app, project, and preferences survive.
4. **Crash isolation.** A dead worker surfaces as one `WorkerExited` event: the session fails, the app continues, the next command runs in a fresh process.
5. **No rewrite of the inference code.** The worker calls the existing warm-up, generate, validate, retry, save, and summary code unchanged. Blocking is not a problem when it happens in another process. This is the constraint that shaped the whole design: the worker is a *process* boundary, not an async rewrite.

What it deliberately does **not** buy is parallelism. There is one worker and one in-flight command. The worker is an ownership and isolation boundary, not a concurrency mechanism.

## Process model and lifetime

`ModelWorker` is a process manager, not a service: a class of statics holding one `multiprocessing` spawn-context process, a command queue, an event queue, a cancellation event, and a continue event.

- The process is created with `multiprocessing.get_context("spawn")` and named `model-worker`. `spawn` is not a preference: CUDA, threads, and model libraries make `fork` unsafe, and `spawn` re-imports modules in the child rather than copying the parent's half-initialized state.
- It is **non-daemonic**. A daemon process may not create children, and the worker does create children (the LavaSR CUDA worker, and nested helpers inside model libraries). Non-daemonic also means the parent must reap it explicitly, which is why `App` starts it after application state is initialized and shuts it down in a `finally` around the main menu loop (`app.py`).
- Startup is bounded by `WORKER_START_TIMEOUT_SECONDS` (20s); graceful shutdown is bounded by `WORKER_SHUTDOWN_TIMEOUT_SECONDS` (5s), after which the manager escalates to terminate/kill.
- The worker's own bootstrap (`_model_worker_main`) marks the process role, sets `SIGINT` to `SIG_IGN`, initializes the local TTS model type, installs console capture, and then reports `WorkerReady(process_id)`. No model is loaded at that point; models load lazily on the first command that needs them and stay resident afterwards.

The main process tracks an explicit `WorkerStatus` — `ABSENT` / `STARTING` / `RUNNING` / `DEAD` — that transitions only under the `ModelWorker` lock. The event drainers (`get_event` / `drain_events`) are the single place that detects death: a closed event queue, or a process that is no longer alive. On detection they move to `DEAD` exactly once and synthesize one `WorkerExited` terminal event, so every consumer finalizes on one event type instead of polling `is_alive()`. `is_busy()` and `status()` answer lifecycle questions without process polling.

A dead worker is resurrected by the next `start()`. Most `submit_*` and `*_blocking` helpers call `start()` first, so a crashed worker reports `WorkerExited` for the in-flight operation and the next command transparently runs in a fresh process. `reset()` terminates and replaces the process; `shutdown()` terminates and leaves the status `ABSENT`. `clear_models_if_running_blocking()` deliberately does *not* start a worker — there is nothing to clear in a process that does not exist.

The worker lifetime is independent of any screen:

```text
application starts
  -> spawned worker becomes ready (no model loaded yet)
  -> normal synchronous menus
  -> run_generation_app() blocks the menu caller
       -> submit GenerateCommand
       -> worker lazily loads/retains models
       -> Textual session drains console + structured events
       -> terminal summary remains visible until Enter
  -> normal synchronous menus resume
  -> later sessions reuse the same worker and model instances
application exits
  -> graceful shutdown, then terminate/kill escalation if needed
```

## Ownership boundary: what lives where

The rule the codebase now enforces: **the interactive main process never constructs a model.** It owns configuration, project state, presentation, and cancellation intent; the worker owns model instances, CUDA contexts, audio devices, and inference threads.

Enforcement lives in `model_runtime.py`. Every process sets a `ModelRuntimeRole` at startup:

- `MODEL_WORKER` — set by `_model_worker_main` via `mark_model_worker()`. Model construction allowed.
- `INTERACTIVE_MAIN` — set by `start.py` before `App()` is created. Model construction forbidden.
- `STANDALONE` — the default for anything that never marks itself: the REST server (`--server`), `testx/` scripts, and most tests. Model construction allowed.

Model accessors call `require_model_owner("<name>")` before constructing anything (`Tts`, `Stt`, `ModelManager` registries, YAMNet, LavaSR). In `INTERACTIVE_MAIN` that raises `RuntimeError`, so an accidental in-process load is a loud failure at the call site rather than a silent VRAM leak. Two consequences worth knowing:

- The guard is **role-based, not import-based**. Importing `tts.py` in the main process is fine and normal — configuration setters, metadata, and validation live there. Only instantiation is policed.
- `STANDALONE` is the permissive default, so the server and scripts keep their in-process inference paths. Worker-only ownership is a property of the interactive app, not of the library.

Configuration still flows through the main process's statics, and the worker reconciles its own copy:

- `Tts.set_type()`, `Stt.set_variant()`, and the model-parameter setters run in the main process to record intent. Those setters only clear resident model instances when the current role is *not* `INTERACTIVE_MAIN` — the interactive main has no instances to clear, and the worker applies the same setters per command.
- The worker applies the settings that matter to inference from a frozen `GenerationSettings` snapshot carried by the command (`stt_variant_id`, `stt_config_id`, `tts_force_cpu`, `tts_model_type_id`, `remote_tts_url`, `save_debug_files`), then uses the normal setters so worker-local statics stay authoritative inside the worker.

**The filesystem is the cross-process data plane.** Generated audio, segment metadata, `project.json`, and `gen_logs/` transcripts are written by whichever process owns them and re-read by the other. After a terminal event the main process copies back `remaining_range_string` and invalidates its sound-segment catalog; if the worker died mid-run, the main process re-derives the range string from the on-disk catalog so stored state always matches the audio that exists.

## The IPC protocol

`model_worker_protocol.py` defines frozen dataclasses moved over two `multiprocessing.Queue`s. Commands carry an `operation_id`; every event carries one too (except `WorkerReady`, which carries the worker pid).

Commands (main to worker):

| Command | What runs in the worker | Driven by |
| --- | --- | --- |
| `GenerateCommand` | the full warm-up / generate / validate / retry / save loop | Generate, quick Generate |
| `TtsPreviewCommand` | one diagnostic inference for a prompt | text-menu preview |
| `RealTimePlaybackCommand` | realtime generation plus PortAudio playback | realtime playback session |
| `SynthesizeChatCommand`, `ResetChatSessionCommand` | one chat utterance; chat session reset | voice chat |
| `TranscribeAudioCommand` | Whisper transcription of a sample buffer | enhance chunking, realtime microphone |
| `InspectTtsCommand` | model metadata, warm-up, blocking issues | chat init, voice/project menus |
| `GetModelStateCommand` | resident-model inventory | menu status block, unload flow |
| `ProbeLavaSrCommand`, `UpsampleFileCommand` | LavaSR availability; file-to-file enhancement | concatenation |
| `ClearModelsCommand` | soft unload of resident models | unload, upsampler handoff |
| `ShutdownCommand` | graceful worker stop | app exit, parent-side timeouts |

Events (worker to main): `WorkerReady`, `ConsoleOutput`, `ConsoleFlush`, `GenerationUpdate`, `GenerationFinished`, `TtsPreviewFinished`, `RealTimePlaybackUpdate`, `RealTimePlaybackFinished`, `ModelsCleared`, `ModelStateReported`, `ChatSessionReset`, `ChatAudioChunk`, `ChatSynthesisFinished`, `AudioTranscribed`, `TtsInspected`, `LavaSrProbed`, `AudioFileUpsampled`, `WorkerCommandFailed`, `WorkerCommandCancelled`, `WorkerStopped`. `WorkerExited` is **main-synthesized** and never sent by the worker.

`GenerationUpdate.update` holds the `GenerationEvent` union from `generation_events.py` (`GenerationPhase | GenerationStarted | GenerationProgress | GenerationStats | GenerationTimedOut | ModelUnhealthy`); consumers dispatch with `isinstance` rather than parsing console text.

**One command in flight.** A submit claims `_active_operation_id` under the lock and refuses if one is already running; the id is released when a terminal event for that operation is observed. Consumers must ignore events whose `operation_id` is not theirs — a stale terminal event must never finalize a newer session. `request_cancel(operation_id)` likewise only sets the cancellation event if the id still matches, so a cancel cannot leak into a successor operation.

**What may cross:** primitives, tuples, paths, JSON-shaped dicts, enums, frozen dataclasses from the protocol module, and NumPy arrays when audio must move (`TranscribeAudioCommand.audio`, `ChatAudioChunk.data`, `TtsPreviewFinished.sound`). `RealTimePlaybackCommand` carries phrase groups as `phrase_groups_json`, i.e. serialized dicts, not objects.

**What must not cross:** `State`, `Project`, model instances, CUDA tensors, file handles, locks, and anything owning a thread or OS resource. `Project` owns a watchdog `Observer`; `State` owns the interactive app's mutable world. Both are rebuilt inside the worker from a project directory instead.

## Worker-side state construction

Each command that needs project context builds its own process-local state:

- `State.for_worker(prefs)` mirrors the instance attributes that `State.__init__` sets, without interactive startup prompts. It is a deliberate duplicate, and a test asserts the attribute sets match — if `State.__init__` gains an attribute, `for_worker` must gain it too.
- `ProjectLoadUtil.load_using_dir_path(..., prompt_on_warnings=False)` loads the project from disk; assigning it through the `State.project` setter also applies `Tts.set_model_params_using_project` and the language whitelist inside the worker.
- Projects that are torn down at the end of a command are `kill()`ed so their watchdog observers stop. Chat is the exception: `_get_or_load_chat_project` caches the loaded project keyed by `(project_dir, st_mtime_ns, st_size)` so a multi-sentence conversation does not re-read and re-validate the book per utterance, and the cached project is the one project the per-command teardown leaves alive.

## Driving a worker job

Two client shapes exist, and both are legitimate:

- **Full-screen sessions** (`WorkerTextualApp` in `textual/worker_app.py`) own the terminal for the duration of a job. `on_mount` submits the job, then a `set_interval(EVENT_POLL_SECONDS)` timer calls `_drain_worker_events()` — the queue is polled, never drained on a worker thread, so the Textual event loop is never blocked. The base class owns the key bindings, console plumbing, the cancel/hard-reset ladder, and the terminal summary; concrete apps (`GenerationApp`, `RealTimePlaybackApp`, `TtsPreviewApp`) supply job submission, session-event dispatch, and result formatting.
- **Blocking helpers** (`*_blocking`) serve menu and flow code that is itself synchronous: `transcribe_audio_blocking`, `upsample_file_blocking`, `probe_lava_sr_blocking`, `inspect_tts_blocking`, `get_model_state_blocking`, `clear_models_if_running_blocking`, `synthesize_chat_blocking`. They loop over `get_event`, relay console output to an optional handler, honor a `cancel_check`, and can enforce `timeout_seconds` — on timeout the parent stops the worker rather than waiting forever. Enhancement's chunked transcription and realtime microphone transcription both use this shape, including a per-chunk retry with a fresh worker after a timeout.

Both shapes must treat the worker as shared: a session that ends without draining leaves a live worker holding a command nobody awaits.

## Cancellation, watchdog, hard reset

Cancellation is **cooperative**. The worker ignores terminal `SIGINT`, so Textual and the main process remain the sole keyboard owner. The parent's `multiprocessing.Event` is installed into the worker's existing `Interrupts` singleton via `set_external_event()`, which means every pre-existing `Interrupts().did_interrupt` check is also a cancellation check — no model call had to be made asynchronous.

The ladder in a full-screen session (`action_cancel_or_reset`):

1. First Ctrl-C calls `request_cancel(operation_id)` and sets the cooperative event.
2. In local (non-SGL-Omni) backend mode, a second Ctrl-C hard-resets: terminate the worker and its descendants, start a fresh one, discard resident model state.
3. In SGL-Omni mode the hard reset is not offered — inference is remote and the worker holds no local TTS memory to dump — so further presses are ignored and the session waits for the cooperative cancel.

The hard-reset action runs on a Textual thread worker so process termination never blocks rendering.

The watchdog covers the case cooperation cannot: a call that never returns. `gen_timeout_util.py` runs `GenTimeoutTracker` on a helper thread inside the worker, because the inference loops run in the worker and a call that never returns would block detection on the calling thread. When it fires it emits `GenerationTimedOut` through the contextvar sink; the session turns that into a hard reset. The first generation step of a run is exempt because it may carry model load or a download. The TTS preview has no "later step", so `_should_watch_preview_inference()` decides per call and `backend_gen_timeout_scope()` supplies an always-armed scope: the first preview of a worker process and any preview whose model is not resident run unwatched; every other preview runs armed. STT has its own story — a parent-side `timeout_seconds` plus a stack dump to `STT_WORKER_STACK_LOG_PATH` after `STT_DIAGNOSTIC_THRESHOLD_SECONDS`, and a chunk-level retry budget in `enhance/enhance_alignment.py`.

Every reset carries an explicit `HardResetCause` (`USER_ESCALATION`, `GENERATION_TIMEOUT`, `MODEL_UNHEALTHY`, `INTERFACE_FAILURE`) so alert and policy behavior derives from the cause rather than from message text. Trigger text says a reset is *required*, never that it succeeded; the terminal summary appends any replacement-startup error from `ModelWorker.reset()`.

## Console capture and crash diagnostics

The worker's output has to reach the parent's UI without ever touching the parent's terminal. Capture happens at two levels:

- Python `sys.stdout` / `sys.stderr` are replaced with queue relays (`_QueueTextStream`). Writes stay chunk-oriented, preserving partial output and stream identity; flushes are explicit `ConsoleFlush` events. The wrappers preserve the original `isatty()` answer so progress libraries keep their terminal-oriented behavior.
- File descriptors 1 and 2 are redirected to child-local pipes read by daemon threads. This catches native extensions, pre-existing logging handlers, and subprocesses that bypass Python stream objects.

If capture itself breaks (event queue closed, pipe broken), the worker makes one best-effort attempt per stream to ship `[worker console capture lost: <stream>]` so a transcript shows where output stopped.

When the worker dies without a reset being attempted, sessions finalize as `FAILED` and name the worker's own log file (`<temp dir>/tts-audiobook-tool-worker.log`, from `init_logging(f"{APP_NAME}-worker")`; `make_worker_log_file_path()` computes the same path in the main process). `WORKER_RESET` is reserved for an intentional termination whose replacement startup was attempted.

## Presentation shell

`WorkerTextualApp` is a temporary `App`, not an async conversion of the outer application; its blocking `.run(inline=False)` follows the same lifecycle as the existing full-screen editors. The screen is a header, an optional app band, a read-only `WorkerLog`, and a find bar.

`WorkerLog` (`textual/worker_content.py`) is a `ScrollView`, so only visible rows render; there is no widget per console line, and the rendered history is capped at `VISIBLE_HISTORY_LINES` (50,000). Up/PageUp/Home and mouse-up suspend tail following; End resumes it. The stream assembler treats newline as a committed history line and carriage return / line-home as in-place replacement of the document's current line, so a progress bar updates on the document's last line and never resizes the log area. ANSI styling converts through `Text.from_ansi`; unsupported cursor controls are normalized rather than emulated.

Each job optionally writes a complete plain-text transcript into the project's `gen_logs/` directory (`PROJECT_GEN_LOG_SUBDIR`); carriage-return progress that replaced the current line is retained as individual transcript lines.

If the terminal cannot host Textual (`can_textual()` / `can_use_full_screen_terminal()`), the job still runs in the worker: the main process synchronously drains worker events to ordinary stdout/stderr, writes the same transcript, handles Ctrl-C through the shared cancellation event, and returns the same typed result. It never falls back to in-process inference.

## Process-safety rules

These are the invariants that keep the worker from becoming an orphan, a zombie, a semaphore leak, or a hang at app exit. Preserve them.

- **Signals.** The worker sets `SIGINT` to `SIG_IGN` at bootstrap. Never install a SIGINT handler in worker code, and never call `signal.signal()` off the main thread. The worker's SIGTERM handler raises `SystemExit(0)` so `atexit` finalizers run and multiprocessing's semaphores are unregistered — killing the worker without that cleanup can leak named semaphores on POSIX.
- **Descendants.** `_force_stop_process()` walks `psutil.Process(pid).children(recursive=True)`, terminates, joins, kills, then calls `psutil.wait_procs`. The worker legitimately spawns children (LavaSR's `lava-sr-cuda` worker, nested CUDA helpers inside model libraries); terminating only the worker would orphan them while they still hold VRAM. Any new nested process must be reachable by that walk.
- **The worker is non-daemonic on purpose.** If that ever changes, the LavaSR CUDA worker and any library-spawned helper breaks.
- **IPC resources are closed deterministically.** `_discard_process_state()` calls `close()` and `join_thread()` on both queues before dropping references. Skipping `join_thread()` can hang interpreter exit waiting on the feeder thread.
- **Blocking helpers must be bounded.** Any new `*_blocking` helper needs a `cancel_check`, and anything that can wait on native code needs `timeout_seconds`; on timeout the parent stops the worker instead of waiting.
- **Handoff discipline inside the worker.** Per-command `finally` blocks clear `Interrupts()`, detach the external event (`set_external_event(None)`), release transient allocator state, and `kill()` non-cached projects. Do not clear `cancellation_event` inside a command handler — only the parent's submit path clears it, before enqueueing.
- **Process exit is the unload boundary.** `unload_models_blocking()` is a shutdown, not a `gc` pass. `ClearModelsCommand` is the soft variant, for handoffs such as concat: `ModelManager.clear_all_models(except_lava_sr=True)` before loading LavaSR, and an explicit clear afterwards even on interruption, so the upsampler never outlives its phase.
- **Do not create a CUDA context in the main process.** `gc_ram_vram()` only touches CUDA when `torch.cuda.is_initialized()` is already true, precisely so a memory-clear helper cannot create an unused context in the UI process.
- **Audio devices belong to the worker.** `SoundDeviceStream` is created inside `RealTimePlaybackCommand`, so the PortAudio stream and its callback thread die with the worker. Where a PortAudio stream is created on POSIX, SIGINT is blocked during stream creation so the spawned audio thread inherits the mask and Ctrl-C stays on the main thread.
- **Sleep inhibition is scoped to the job.** `SystemSleepLock` is released when the worker job reaches a terminal result, not when the session's summary prompt ends.

## Rules for contributors touching worker-owned code

- **Add the guard, not the workaround.** Any new code path that constructs a model, a CUDA context, or an audio device must sit behind `require_model_owner()`. If it must run in the main process, it is not worker-owned and should not be heavyweight.
- **Anything crossing IPC must be small, frozen, and picklable.** Add fields as primitives or protocol dataclasses. Never put `State`, `Project`, a model instance, GPU-resident audio, or a lock in a command or event.
- **New protocol messages are a three-part change**: the dataclass, membership in the `ModelWorkerCommand` / `ModelWorkerEvent` union, and a handler or dispatch branch. A new *terminal* event must also join the manager's terminal-event set, or the worker stays "busy" forever and every later submit is refused.
- **Keep `operation_id` on everything** and filter by it on the consumer side.
- **Defer heavy imports inside worker command handlers.** Under `spawn`, module-level imports run in the child; a module-level `torch` or model import makes every worker start slow and pulls the same code into the main process.
- **Mirror `State.__init__` in `State.for_worker`.** There is a test for the attribute-set match; it fails on purpose when either drifts.
- **Any preference that changes inference must be added to `GenerationSettings`** and applied by the worker handler. The worker does not see the main process's `Prefs` object.
- **Do not make the business logic asynchronous.** `generate_files()` stays a blocking call. Progress is added by emitting a `GenerationEvents` event at a natural boundary — with no sink installed those calls are no-ops — not by threading callbacks through the generation code.
- **Emit from helper threads through the copied context.** The sink is a `contextvar`; a watchdog thread emits via `contextvars.copy_context()` so the event reaches the operation's sink instead of leaking into another thread's.
- **Never print from the main process during a worker session.** Output reaches the screen only as `ConsoleOutput` events.
- **Cancellation must be observable at a safe boundary.** Add `Interrupts().did_interrupt` checks in any new long loop inside worker-owned code; a loop with no check is uncancellable and will be hard-reset instead.
- **Give every new blocking inference path a timeout story** — the in-worker watchdog for TTS steps, a parent-side `timeout_seconds` for anything else — and say which one applies.
- **Add a test per protocol message.** `tests/test_model_worker.py` covers lifecycle (start/shutdown, synthesized `WorkerExited`, SIGTERM handling, deterministic IPC teardown, cancel-then-acknowledge, timeout-stops-worker, `for_worker` parity); `tests/test_worker_reset.py` covers reset causes. Run that file, not the whole suite, when touching the worker.

## Weaknesses and things worth revisiting

None of these are bugs; they are the costs of the current shape, and the places where a future change is most likely to pay off.

- **Serial by construction.** One worker, one in-flight command. A long generation blocks a chat utterance, a preview, or a status query, and `*_blocking` callers simply refuse or wait. A second worker (per GPU, or one for STT and one for TTS) is the obvious scaling step, but it turns `_active_operation_id`, the shared cancellation event, and the resident-model statics into per-worker state, which is the hard part.
- **Worker-wide signals, not per-operation signals.** `cancellation_event` and `continue_event` are single process-wide events. Identity checks at the boundary prevent most leakage, but the vocabulary does not scale to concurrent operations, and a handler that cleared the event would corrupt its successor.
- **Ownership enforcement is opt-in by entry point.** `STANDALONE` is the default role, so the server, `testx/`, and tests keep in-process inference and never exercise the worker. A regression in the worker path is invisible to those paths, and a new entry point silently gets the permissive role. Inverting the default (strict unless explicitly standalone), or an import-time guard for model libraries, would close that gap.
- **Configuration is duplicated by hand.** `GenerationSettings` is a manual projection of the preferences that affect inference, and the main process's statics must be re-applied inside the worker per command. Forgetting a field is silent. A mechanically derived snapshot, or a test asserting that every inference-affecting preference is carried, would remove the class of bug.
- **`GetModelStateCommand` starts a worker.** The menu status block calls `get_model_state_blocking()`, which calls `start()`. A read-only query can therefore spawn a process — and if the worker crashes at startup, every menu redraw respawns it. A `peek` variant that never spawns, plus a startup-failure counter that parks the worker in a disabled state, would avoid a crash loop.
- **Unbounded queues, no backpressure.** A chatty model library at debug verbosity can push `ConsoleOutput` faster than the UI drains it; the 50,000-line cap is UI-side only, so the parent's memory grows. Coalescing console events per drain tick, or a bounded queue with an "output dropped" marker, is cheap insurance.
- **Audio crosses IPC inconsistently — on purpose, but worth revisiting.** Realtime playback keeps PortAudio in the worker to avoid shipping a continuous stream; chat ships per-sentence `ChatAudioChunk`s, enhance ships whole Whisper chunks, and a preview ships an entire `Sound` back. All acceptable at current sizes, but a shared-memory buffer or a temp-file handoff is the right shape for long audio, and the asymmetry is easy to get wrong when adding a feature.
- **Death detection lives in the drainers.** A path that stops draining without observing a terminal event can leave a dead worker in `DEAD` until the next command. A monitor thread, or use of the process sentinel, would make detection independent of consumer behavior.
- **Error handling is string-shaped.** `WorkerCommandFailed.message` is a formatted exception, and callers compare against constants such as `STT_TRANSCRIPTION_TIMEOUT_ERROR` by equality. An error-code enum in the protocol would make retry policy robust and testable.
- **No retry or resume for a crashed command.** A worker death fails the operation; only enhance's chunk loop implements its own retry with a fresh worker. Anything else loses in-progress work, which is acceptable for generation (completed segments are already on disk) but not obviously for other commands.
- **No protocol versioning or log correlation.** Parent and worker always run the same code today, so this is latent. A `protocol_version` in `WorkerReady`, and the `operation_id` on worker log lines, would make a future mismatch or a cross-command timing question answerable from the log alone.
- **The console-relay layer is a compatibility shim we would rather not own.** Capturing Python streams and fds, preserving `isatty()`, and normalizing ANSI exists so unmodified third-party code can run somewhere its output is not a terminal. The `GenerationEvents` path is the direction of travel: if progress, warnings, and diagnostics became structured events, the capture layer could shrink to a safety net.
- **No observability of reset frequency.** Hard resets are the recovery mechanism for the worst failure mode, and nothing records how often they happen, by cause, per model. That data would tell us which runtimes deserve a narrower timeout or an upstream bug report.
- **Log retention beyond the cap is unmeasured.** If `WorkerLog` retention becomes the bottleneck under sustained output, replace only the log widget with a deque-backed `ScrollView`; the protocol and the generation engine do not change.
