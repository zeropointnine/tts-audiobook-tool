# Generation TUI and Model Worker Architecture

## Scope

The first rollout moves the conventional and quick **Generate** paths into a long-lived spawned model worker and presents each run in a temporary full-screen Textual application. Automatic concatenation/upscaling remains outside the modal and resumes in the conventional console flow.

This began as a Generate-first migration. Realtime playback now also uses the long-lived worker and a temporary full-screen app; voice chat, enhancement, and a few model-aware menus still have legacy in-process paths. They must be migrated in later phases before the main process can be considered a strict no-model process for every feature.

## Process and UI lifetimes

`App` starts `ModelWorker` once after normal application state is initialized and shuts it down in a `finally` block around the main menu loop. The worker uses the multiprocessing `spawn` context and is non-daemonic so existing model code which creates a child process can continue to work.

The worker lifetime is independent of a generation screen:

```text
application starts
  -> spawned worker becomes ready (no model loaded yet)
  -> normal synchronous menus
  -> GenerationApp.run() blocks the menu caller
       -> submit GenerateCommand
       -> worker lazily loads/retains models
       -> Textual drains console and structured events
       -> terminal summary remains visible until Enter
  -> normal synchronous menus resume
  -> later GenerationApp reuses the same worker/model instances
application exits
  -> graceful worker shutdown, then terminate/kill escalation if needed
```

A hard reset terminates the worker process and starts a fresh one. Before forced termination, the manager captures and terminates nested helper processes (for example the per-segment LavaSR CUDA worker) so they cannot become orphaned. Process exit is the definitive cleanup boundary for Python objects, native state, and CUDA allocations.

## IPC protocol

The protocol in `model_worker_protocol.py` consists of small frozen dataclasses sent through two multiprocessing queues:

- main -> worker: `GenerateCommand`, `ClearModelsCommand`, `ShutdownCommand`
- worker -> main: `WorkerReady`, `ConsoleOutput`, `ConsoleFlush`, `GenerationUpdate`, `GenerationFinished`, `ModelsCleared`, `WorkerCommandFailed`, `WorkerStopped`
- main-synthesized: `WorkerExited` — never sent by the worker; the main process synthesizes it when a client drainer detects that the worker process has died

`GenerationUpdate.update` is typed as the `GenerationEvent` union from `generation_events.py` (`GenerationPhase | GenerationStarted | GenerationProgress | GenerationStats`); consumers dispatch on it with `isinstance` instead of parsing console text. The optional sink around `generate_files()` lives in a `contextvar` so the active sink cannot leak between threads or operations.

## Worker lifecycle

The main process tracks an explicit `WorkerStatus` (absent / starting / running / dead) that transitions only under the `ModelWorker` lock. The event drainers (`get_event` / `drain_events`) are the single place that detects worker death: a closed event queue (EOF/OSError/ValueError) or a process that is no longer alive. On detection they transition to dead exactly once and synthesize one `WorkerExited` terminal event into the pending events, so both UI modes finalize on that one event type instead of polling `ModelWorker.is_alive()`. Busy-ness and lifecycle are queryable via `ModelWorker.is_busy()` / `ModelWorker.status()` without process polling.

A dead worker is resurrected by the next `start()` call: `submit_generation` and `clear_models_blocking` both start first, so a crashed worker reports a `WorkerExited` result for the in-flight operation and the next command runs in a fresh process. `reset()` / `shutdown()` discard all process state (status returns to absent).

Every operation event carries an `operation_id` (the startup `WorkerReady` notification carries the worker pid instead). The current implementation permits one in-flight command, matching the blocking main menu and modal generation flow.

A `GenerateCommand` contains a project directory, selected indices, batch size, regeneration flag, and an explicit snapshot of the preferences that affect generation. `State` and `Project` do not cross IPC: `Project` owns a watchdog observer and is not safely picklable. The worker loads its own `Project`, constructs a process-local `State` without invoking interactive startup prompts, and applies the snapshot through the normal state setters so `Tts`/`Stt` static configuration remains authoritative inside the worker.

After a terminal event, the main process copies back `remaining_range_string` and explicitly invalidates its sound-segment catalog. If the worker was hard-reset or exited without completing, its in-memory range update may never have reached disk; in that case the main process re-derives the range string from the (just invalidated) file-based segment catalog and persists it, so the stored range always matches the audio on disk. Generated audio and metadata remain filesystem-owned, as before.

## Preserving synchronous generation

`GenerateUtil.generate_files()` remains a normal blocking function. The worker calls the existing warm-up, generation, validation, retry, save, and summary code directly.

`generation_events.py` adds a process-local optional sink around that synchronous call. A handful of existing natural boundaries emit:

- `GenerationPhase`
- `GenerationStarted`
- `GenerationProgress`
- `GenerationStats`

With no sink installed these calls are no-ops. Console text is not parsed to reconstruct header state.

## Cancellation and reset

The worker ignores terminal `SIGINT`; Textual/main remains the sole keyboard owner. A multiprocessing cancellation event is installed as an additional source in the worker's existing `Interrupts` singleton. Existing `did_interrupt` checks therefore stop at the same safe boundaries without making model calls asynchronous.

In the generation screen:

1. First Ctrl-C sets the cooperative cancellation event.
2. A second Ctrl-C opens a confirmation dialog.
3. Confirming terminates and restarts the worker, discarding resident model state.

The UI remains responsive while a blocking inference call finishes. The hard-reset action runs on a Textual thread worker so process termination does not block rendering.

## Console capture

At worker bootstrap, Python `sys.stdout` and `sys.stderr` are replaced with file-like queue relays. Writes remain chunk-oriented, preserving partial output and stream identity. Flushes are represented explicitly.

File descriptors 1 and 2 are also redirected to child-local pipes read by daemon threads. This fallback captures native extensions, previously-created logging handlers, and subprocesses which bypass Python's stream objects; such output cannot write directly into and corrupt the parent's full-screen terminal.

The worker stream wrappers preserve the original `isatty()` answer so progress libraries retain their normal terminal-oriented behavior.

## Crash diagnostics

If the worker process dies, the main process' WORKER_RESET messages name the worker's own log file (`<temp dir>/tts-audiobook-tool-worker.log`, the file the worker writes through `init_logging(f"{APP_NAME}-worker")`), so a crash can be diagnosed from the terminal summary in both UI modes. If the worker's console capture itself fails (event queue closed, or the fd-level capture pipe breaks), the worker makes one best-effort attempt per capture to ship a `[worker console capture lost: <stream>]` marker before giving up, so the transcript shows where output stopped.

## Textual presentation

`GenerationApp` is a separate temporary `App`, not a conversion of the outer application to async. Its blocking `.run(inline=False)` call follows the same lifecycle as the existing full-screen editors.

The screen contains:

- three fixed header lines for phase, counts, elapsed time, current indices, speed, and realtime factor;
- a read-only reflowing scrolling output area (`WorkerLog`, a `ScrollView` subclass shared with realtime playback);
- a prompt/status line.

`WorkerLog` is a `ScrollView`, so only visible rows are rendered. It provides a normal scrollbar without creating one widget per console line. The rendered history is capped at 50,000 logical lines. Up/PageUp/Home and mouse-up scrolling suspend tail following; End resumes it. The stream assembler treats newline as a committed history line and carriage return / line-home controls as in-place replacement of the document's current line. ANSI styling is converted with `Text.from_ansi`; unsupported cursor controls are normalized rather than implementing a full terminal emulator.

The log's document always ends with the *current line*: partial/carriage-return progress (a dynamic progress bar, a status) lives on the document's last line and updates in place, exactly as it would in a conventional console. Committing the line (a newline) turns it into a normal history line and starts a fresh empty current line below it, so a commit advances the document by one line in the normal document flow — and because the in-progress line is part of the document, a progress bar starting or completing never resizes the log area.

Each job also writes a complete plain-text transcript into the project's `gen_logs/` directory (the subdirectory name is defined by the constant `PROJECT_GEN_LOG_SUBDIR`). Carriage-return progress updates that replace the log's current line are retained as individual transcript lines rather than discarded. The UI remains on its terminal summary, including the transcript path, until Enter is pressed.

## Non-Textual fallback

If the terminal cannot host Textual, generation still executes in `ModelWorker`. The main process synchronously drains worker events back to ordinary stdout/stderr, writes the same transcript, handles Ctrl-C through the shared cancellation event, and returns a typed terminal result to the existing menu flow. It does not silently fall back to in-process model inference.

## Model clearing and phased ownership

The Options menu's soft-unload action clears both domains during the phased rollout:

- legacy in-process models used by features not yet migrated;
- worker-owned generation models through `ClearModelsCommand`.

A separate hard-reset item restarts the worker. Model-parameter changes do not need a special clear command: assigning each newly loaded project through the normal `State.project` setter updates worker-local `Tts` model parameters, whose existing dirty check unloads an incompatible resident instance.

To avoid accidentally creating an otherwise unused CUDA context in the main process, `gc_ram_vram()` only synchronizes/clears CUDA when PyTorch reports that CUDA is already initialized.

## Known follow-up work

- Realtime playback is worker-backed, with `SoundDeviceStream` deliberately owned by the worker to keep generated audio off IPC. Revisit main-process audio/chunk IPC only if worker-owned PortAudio proves unstable on supported platforms.
- Route streaming chat through the worker; it still needs audio/chunk IPC beyond the current protocols.
- Route enhancement/STT-only operations and model-instantiating menu actions through the worker.
- Once those paths move, remove the transitional main-process model clearing and enforce worker-only heavyweight ownership globally.
- Benchmark sustained output beyond the current 50,000-line cap. If `RichLog` retention pruning becomes the bottleneck, replace only the log widget with a deque-backed custom `ScrollView`; the worker protocol and generation engine do not change.
