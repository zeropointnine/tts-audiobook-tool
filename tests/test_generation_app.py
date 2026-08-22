import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from textual.widgets import Static

from tts_audiobook_tool import util
from tts_audiobook_tool.constants import COL_ERROR
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    ConsoleOutput,
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    WorkerExited,
)
from tts_audiobook_tool.generation_events import GenerationProgress, GenerationStarted
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import generation_app as generation_app_module
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual.worker_content import WorkerLogContentArea
from tts_audiobook_tool.textual.generation_app import (
    ConsoleLineAssembler,
    GenerationApp,
    GenerationModalResult,
    GenerationTranscript,
    _present_console_result,
    _read_persisted_range_string,
    _reconcile_generation_result,
    _run_generation_console,
)


def run(coroutine) -> None:
    asyncio.run(coroutine)


def make_state() -> State:
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    return cast(State, SimpleNamespace(project=project))


def test_console_line_assembler_handles_partial_lines_and_carriage_replacement() -> None:
    assembler = ConsoleLineAssembler()

    lines, live = assembler.feed("loading 1")
    assert lines == []
    assert live == "loading 1"

    lines, live = assembler.feed("\rloading 2\x1b[K\x1b[")
    assert lines == []
    assert live == "loading 2"

    # CSI controls may be split across worker queue chunks.
    lines, live = assembler.feed("1Gdone\nnext")
    assert lines == ["done"]
    assert live == "next"
    assert assembler.finish() == ["next"]


def test_transcript_strips_ansi_but_keeps_all_lines(tmp_path) -> None:
    path = tmp_path / "generation.log"
    transcript = GenerationTranscript(str(path))
    transcript.write_lines([f"{COL_ERROR}failed", "plain"])
    transcript.write_chunk("progress 1\rprogress 2\n")
    transcript.write_chunk("\x1b[")
    transcript.write_chunk("31mred\x1b[0m\n")
    transcript.close()

    assert path.read_text(encoding="utf-8") == (
        "failed\nplain\nprogress 1\nprogress 2\nred\n"
    )


def test_console_line_assembler_strips_non_hyperlink_osc_and_holds_split_osc() -> None:
    assembler = ConsoleLineAssembler()

    # A complete OSC sequence (terminal title, BEL-terminated) is discarded.
    lines, live = assembler.feed("\x1b]2;job title\x07done\n")
    assert lines == ["done"]
    assert live == ""

    # A complete OSC sequence terminated by ST (ESC \).
    lines, live = assembler.feed("\x1b]0;name\x1b\\ok\n")
    assert lines == ["ok"]
    assert live == ""

    # An OSC split across chunks is held until its terminator arrives.
    assembler = ConsoleLineAssembler()
    lines, live = assembler.feed("\x1b]2;titl")
    assert lines == []
    assert live == ""
    lines, live = assembler.feed("e\x07after\n")
    assert lines == ["after"]
    assert live == ""

    # Even a split ST terminator must not leak the OSC into the live line.
    lines, live = assembler.feed("\x1b]2;x\x1b")
    assert lines == []
    assert live == ""
    lines, live = assembler.feed("\\next\n")
    assert lines == ["next"]
    assert live == ""


def test_console_line_assembler_preserves_complete_and_split_osc_8_links() -> None:
    link = "\x1b]8;;file:///tmp/book.flac\x1b\\book.flac\x1b]8;;\x1b\\"
    assembler = ConsoleLineAssembler()

    lines, live = assembler.feed(f"Saved: {link}\n")
    assert lines == [f"Saved: {link}"]
    assert live == ""

    assembler = ConsoleLineAssembler()
    split_at = len("\x1b]8;;file:///tmp/bo")
    lines, live = assembler.feed("Saved: " + link[:split_at])
    assert lines == []
    assert live == "Saved: "
    lines, live = assembler.feed(link[split_at:] + "\n")
    assert lines == [f"Saved: {link}"]
    assert live == ""


def test_transcript_strips_osc_and_holds_split_osc(tmp_path) -> None:
    path = tmp_path / "generation.log"
    transcript = GenerationTranscript(str(path))
    transcript.write_chunk("\x1b]2;title\x07hello\n")
    transcript.write_chunk("\x1b]0;name\x1b\\world\n")
    # An OSC split across chunks must not leak into the transcript.
    transcript.write_chunk("\x1b]2;sp")
    transcript.write_chunk("lit\x07tail\n")
    transcript.close()

    assert path.read_text(encoding="utf-8") == "hello\nworld\ntail\n"


def test_disabled_transcript_creates_no_file_and_writes_nothing(tmp_path) -> None:
    path = tmp_path / "gen_logs" / "generation.log"
    transcript = GenerationTranscript(str(path), enabled=False)

    assert transcript.path == ""
    transcript.write_lines(["hello"])
    transcript.write_chunk("world\n")
    transcript.close()

    assert not path.exists()


def test_generation_result_reconciles_range_and_sound_catalog(tmp_path) -> None:
    project_file = tmp_path / "project.json"
    project_file.write_text('{"generate_range_string": "4-6"}', encoding="utf-8")
    invalidations: list[bool] = []
    state = cast(
        State,
        SimpleNamespace(
            project=SimpleNamespace(
                dir_path=str(tmp_path),
                generate_range_string="all",
                phrase_groups=[],
                sound_segments=SimpleNamespace(
                    force_invalidate=lambda: invalidations.append(True),
                    sound_segments_map={},
                ),
                save=lambda: "",
            )
        ),
    )
    assert _read_persisted_range_string(state) == "4-6"
    result = GenerationModalResult(
        GenerationTerminalStatus.WORKER_RESET,
        "4-6",
        str(tmp_path / "generation.log"),
    )

    _reconcile_generation_result(state, result)

    assert state.project.generate_range_string == "4-6"
    assert invalidations == [True]


def test_reconcile_rederives_range_string_from_segment_catalog(tmp_path) -> None:
    saved: list[str] = []
    project = SimpleNamespace(
        dir_path=str(tmp_path),
        generate_range_string="all",
        phrase_groups=[0, 1, 2],
        sound_segments=SimpleNamespace(
            force_invalidate=lambda: None,
            sound_segments_map={0: "00001.flac", 1: "00002.flac"},
        ),
        save=lambda: (saved.append(project.generate_range_string) or ""),
    )
    state = cast(State, SimpleNamespace(project=project))
    # The (killed) worker never persisted its own range update, so the
    # copy-back value still says "all" even though segments 1-2 were written.
    result = GenerationModalResult(
        GenerationTerminalStatus.WORKER_RESET,
        "all",
        str(tmp_path / "generation.log"),
    )

    _reconcile_generation_result(state, result)

    # The re-derived range excludes the segments that exist on disk.
    assert project.generate_range_string == "3"
    assert saved == ["3"]


def test_non_textual_fallback_relays_worker_output(monkeypatch, tmp_path, capsys) -> None:
    events = iter(
        [
            ConsoleOutput("job", "stdout", "progress 1\rprogress 2\n"),
            GenerationFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                "none",
            ),
        ]
    )
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        staticmethod(lambda timeout=0.1: next(events)),
    )
    transcript = GenerationTranscript(str(tmp_path / "generation.log"))

    try:
        result = _run_generation_console(
            make_state(),
            {0},
            1,
            False,
            transcript,
        )
    finally:
        transcript.close()

    assert result.status == GenerationTerminalStatus.COMPLETED
    assert "progress 2" in capsys.readouterr().out
    assert Path(transcript.path).read_text(encoding="utf-8") == (
        "progress 1\nprogress 2\n"
    )


def test_console_fallback_finalizes_on_worker_exited(monkeypatch, tmp_path) -> None:
    events = iter(
        [
            ConsoleOutput("job", "stdout", "progress 1\n"),
            WorkerExited(
                "job",
                "Model worker process exited unexpectedly. Worker log: /tmp/worker.log",
            ),
        ]
    )
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "get_event",
        staticmethod(lambda timeout=0.1: next(events)),
    )
    transcript = GenerationTranscript(str(tmp_path / "generation.log"))

    try:
        result = _run_generation_console(
            make_state(),
            {0},
            1,
            False,
            transcript,
        )
    finally:
        transcript.close()

    assert result.status == GenerationTerminalStatus.WORKER_RESET
    assert "Worker log: /tmp/worker.log" in result.message


def test_generation_app_finalizes_on_worker_exited(monkeypatch, tmp_path) -> None:
    queued_events = [
        [
            ConsoleOutput("job", "stdout", "worker output\n"),
            WorkerExited(
                "job",
                "Model worker process exited unexpectedly. Worker log: /tmp/worker.log",
            ),
        ]
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: queued_events.pop(0) if queued_events else []),
    )
    monkeypatch.setattr(
        generation_app_module,
        "make_worker_log_file_path",
        lambda: "/tmp/fake-worker.log",
    )

    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(make_state(), {0}, 1, False, transcript)

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.status == GenerationTerminalStatus.WORKER_RESET
            assert "Worker log: /tmp/worker.log" in app.terminal_result.message
            await pilot.press("enter")
        assert app.return_value is not None
        assert app.return_value.status == GenerationTerminalStatus.WORKER_RESET

    try:
        run(exercise())
    finally:
        transcript.close()


def test_generation_app_waits_for_enter_after_terminal_summary(monkeypatch, tmp_path) -> None:
    queued_events = [
        [
            ConsoleOutput("job", "stdout", "worker output\n"),
            GenerationUpdate("job", GenerationStarted(total=2)),
            GenerationUpdate(
                "job",
                GenerationProgress(processed=1, remaining=1, total=2),
            ),
            GenerationFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                "2",
            ),
        ]
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: queued_events.pop(0) if queued_events else []),
    )
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))

    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(make_state(), {0, 1}, 1, False, transcript)

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.status == GenerationTerminalStatus.COMPLETED
            assert app.return_value is None
            # The bottom prompt row was removed; the terminal phase now
            # surfaces in the header status line instead.
            assert "Generation completed" in str(
                app.query_one("#generation-status", Static).render()
            )
            # The header title is the fixed accent-colored line.
            assert "Generating audio..." in str(
                app.query_one("#generation-title", Static).render()
            )
            # Once the job has stopped, the header prompt line switches to
            # the continue hint.
            assert "to continue" in str(
                app.query_one("#generation-hotkey", Static).render()
            )
            await pilot.press("enter")
        assert app.return_value is not None
        assert app.return_value.status == GenerationTerminalStatus.COMPLETED

    try:
        run(exercise())
    finally:
        transcript.close()

    text = Path(transcript.path).read_text(encoding="utf-8")
    assert "worker output" in text
    assert "Generation completed." in text
    # The transcript stores plain text (ANSI is stripped), so compare
    # against the uncolored hotkey label.
    assert "Press [ENTER] to continue" in text


def test_generation_app_plays_done_sound_for_completion_and_automatic_abort(
    monkeypatch,
) -> None:
    app = GenerationApp(
        make_state(),
        {0},
        1,
        False,
        GenerationTranscript("", enabled=False),
    )
    sound_calls: list[None] = []
    monkeypatch.setattr(
        generation_app_module.app_support,
        "play_done_sound",
        lambda: sound_calls.append(None),
    )

    statuses_that_alert = {
        GenerationTerminalStatus.COMPLETED,
        GenerationTerminalStatus.ABORTED,
    }
    for status in GenerationTerminalStatus:
        sound_calls.clear()

        app._post_terminal_summary(GenerationModalResult(status, "", ""))

        assert bool(sound_calls) is (status in statuses_that_alert)


def test_quick_generation_auto_returns_without_concatenation_message() -> None:
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=True)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        True,
        GenerationTranscript("", enabled=False),
    )
    result = GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")

    app._pre_terminal_summary(result)
    app.terminal_result = result

    assert app.auto_continue
    assert app.prompt_mode == "auto_return"
    assert app.terminal_summary_extra_lines(result) == []


def test_quick_generation_auto_returns_without_auto_concat() -> None:
    """A quick generation that completed skips the ENTER wait even when
    auto-concat is disabled: the editor flow resumes on its own."""
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        True,
        GenerationTranscript("", enabled=False),
    )
    result = GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")

    app._pre_terminal_summary(result)
    app.terminal_result = result

    assert app.auto_continue
    assert app.prompt_mode == "auto_return"
    assert app.terminal_summary_extra_lines(result) == []


def test_quick_generation_failed_item_still_waits_for_enter() -> None:
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        True,
        GenerationTranscript("", enabled=False),
    )
    result = GenerationModalResult(
        GenerationTerminalStatus.COMPLETED,
        "",
        "",
        failed_items=1,
    )

    app._pre_terminal_summary(result)
    app.terminal_result = result

    assert not app.auto_continue
    assert app.prompt_mode == "finished"
    assert app.terminal_label(result) == "Generation completed."
    assert any("ENTER" in line for line in app.terminal_summary_extra_lines(result))


def test_quick_generation_interrupted_still_waits_for_enter() -> None:
    """Only an uninterrupted completion bypasses ENTER: cancelled, aborted,
    failed, and reset quick generations still hold the summary for review."""
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=True)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        True,
        GenerationTranscript("", enabled=False),
    )
    interrupted_statuses = set(GenerationTerminalStatus) - {
        GenerationTerminalStatus.COMPLETED
    }

    for status in interrupted_statuses:
        result = GenerationModalResult(status, "", "")

        app._pre_terminal_summary(result)
        app.terminal_result = result

        assert not app.auto_continue
        assert app.prompt_mode == "finished"
        # The banner is only suppressed for completions: interrupted quick
        # generations still announce why they stopped.
        assert app.terminal_label(result)
        assert app.terminal_display_label(result)
        assert any("ENTER" in line for line in app.terminal_summary_extra_lines(result))


def test_regular_generation_still_waits_for_enter_without_auto_concat() -> None:
    """The auto-concat preference still governs the regular generation
    flow: without it, a completed batch waits for ENTER."""
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        False,
        GenerationTranscript("", enabled=False),
    )
    result = GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")

    app._pre_terminal_summary(result)
    app.terminal_result = result

    assert not app.auto_continue
    assert app.prompt_mode == "finished"
    # Regular generation always shows the completion banner.
    assert app.terminal_label(result) == "Generation completed."
    assert app.terminal_display_label(result) == "Generation completed."
    assert any("ENTER" in line for line in app.terminal_summary_extra_lines(result))


def test_quick_generation_suppresses_completion_banner() -> None:
    """A quick generation that completed shows no completion banner: it
    returns straight to the editor, so the banner would only flash for the
    brief auto-return delay."""
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)),
        {0},
        1,
        True,
        GenerationTranscript("", enabled=False),
    )
    result = GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")

    assert app.terminal_label(result) == ""
    assert app.terminal_display_label(result) == ""


def test_quick_generation_app_exits_without_enter_keypress(
    monkeypatch, tmp_path
) -> None:
    queued_events = [
        [ConsoleOutput("job", "stdout", "worker output\n")],
        [
            GenerationFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                "none",
            )
        ],
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: queued_events.pop(0) if queued_events else []),
    )
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))
    # Park the poll interval so the test drives each event batch manually;
    # the header can then be checked while the session is still live.
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)

    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)), {0}, 1, True, transcript
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            # Only the console output so far: the session is still running.
            app._drain_worker_events()
            await pilot.pause()
            assert app.terminal_result is None
            # The header title announces the quick-generate flow.
            title = str(app.query_one("#generation-title", Static).render())
            assert "Quick generate" in title
            assert "Generating audio" not in title

            # The finish event ends the session immediately: no ENTER wait,
            # no banner, and no continue pause.
            app._drain_worker_events()
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.status == GenerationTerminalStatus.COMPLETED
            assert app.auto_continue

    try:
        run(exercise())
    finally:
        transcript.close()

    # The app exited on its own within that pause: no ENTER keypress and
    # no continue delay were involved.
    assert app.return_value is not None
    assert app.return_value.status == GenerationTerminalStatus.COMPLETED

    text = Path(transcript.path).read_text(encoding="utf-8")
    assert "worker output" in text
    assert "Generation completed." not in text
    assert f"Press {util.make_hotkey_string('ENTER')} to continue" not in text
    assert "Proceeding to concatenation..." not in text


def test_quick_generation_console_omits_concatenation_message(capsys) -> None:
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=True)
    state = cast(State, SimpleNamespace(project=project))
    result = GenerationModalResult(GenerationTerminalStatus.COMPLETED, "", "")

    _present_console_result(
        state,
        result,
        GenerationTranscript("", enabled=False),
        is_regen=True,
    )

    assert "Proceeding to concatenation" not in capsys.readouterr().out


def test_quick_generation_console_failed_item_waits_for_enter(
    monkeypatch, capsys
) -> None:
    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=False)
    state = cast(State, SimpleNamespace(project=project))
    result = GenerationModalResult(
        GenerationTerminalStatus.COMPLETED,
        "",
        "",
        failed_items=1,
    )
    enter_prompts: list[None] = []
    monkeypatch.setattr(generation_app_module.ask, "can_hotkey", True)
    monkeypatch.setattr(
        generation_app_module.ask,
        "ask_enter_to_continue",
        lambda: enter_prompts.append(None),
    )

    _present_console_result(
        state,
        result,
        GenerationTranscript("", enabled=False),
        is_regen=True,
    )

    assert enter_prompts == [None]
    assert "Generation completed." in capsys.readouterr().out


def test_generation_app_auto_continues_when_auto_concat_enabled(monkeypatch, tmp_path) -> None:
    queued_events = [
        [
            ConsoleOutput("job", "stdout", "worker output\n"),
            GenerationUpdate("job", GenerationStarted(total=2)),
            GenerationUpdate(
                "job",
                GenerationProgress(processed=1, remaining=1, total=2),
            ),
            GenerationFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                "2",
            ),
        ]
    ]
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(
        ModelWorker,
        "drain_events",
        staticmethod(lambda max_events=1000: queued_events.pop(0) if queued_events else []),
    )
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))

    project = SimpleNamespace(generate_range_string="all", gen_auto_concat=True)
    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(
        cast(State, SimpleNamespace(project=project)), {0, 1}, 1, False, transcript
    )

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.3)
            assert app.terminal_result is not None
            assert app.terminal_result.status == GenerationTerminalStatus.COMPLETED
            assert app.auto_continue

    try:
        run(exercise())
    finally:
        transcript.close()

    # The app exited on its own within that pause: no ENTER keypress and
    # no continue delay were involved.
    assert app.return_value is not None
    assert app.return_value.status == GenerationTerminalStatus.COMPLETED

    text = Path(transcript.path).read_text(encoding="utf-8")
    assert "worker output" in text
    assert "Generation completed." in text
    assert "Proceeding to concatenation..." in text
    assert f"Press {util.make_hotkey_string('ENTER')} to continue" not in text


def test_progress_bar_updates_never_resize_the_log_area(monkeypatch, tmp_path) -> None:
    """The model library's carriage-return progress bar updates the
    document's current line in place, and committing it advances the
    document by one line in the normal document flow. Across the bar's
    whole lifecycle (start, ticks, commit) the log area's height must
    stay constant, so the content area never resizes or flickers when
    a progress bar starts or finishes."""
    queued_events = [
        [ConsoleOutput("job", "stdout", "worker output\n")],
        [ConsoleOutput("job", "stdout", "\rbar 10%")],
        [ConsoleOutput("job", "stdout", "\rbar 50%")],
        [ConsoleOutput("job", "stdout", "\rbar 100%\n")],
        [
            GenerationFinished(
                "job",
                GenerationTerminalStatus.COMPLETED,
                "2",
            )
        ],
    ]

    def fake_drain(max_events=1000):
        # One batch per call, so a manual drain delivers exactly one.
        return queued_events.pop(0) if queued_events else []

    # Park the poll interval; the test drives the drain manually so each
    # phase boundary (idle feed, bar start, bar ticks, bar commit,
    # finished) is sampled deterministically.
    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    # Park the settle timer so the terminal summary (which would add
    # lines) cannot fire mid-test.
    monkeypatch.setattr(
        generation_app_module, "FINAL_OUTPUT_SETTLE_SECONDS", 3600.0
    )
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain))
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))

    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(make_state(), {0}, 1, False, transcript)

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            log = app.query_one(WorkerLogContentArea).worker_log
            # One sample per phase: the initial idle state, then the
            # state after each drained event batch. The virtual height
            # tracks the document: the empty current line; the committed
            # output line; the bar ticking in place on the current line
            # (no growth); the committed bar plus a fresh current line.
            virtual_heights: list[int] = [log.virtual_size.height]
            area_heights: list[int] = [log.scrollable_content_region.height]
            for _ in range(len(queued_events)):
                app._drain_worker_events()
                await pilot.pause()
                virtual_heights.append(log.virtual_size.height)
                area_heights.append(log.scrollable_content_region.height)
            assert virtual_heights == [1, 2, 2, 2, 3, 3]
            # The area itself never resized: no layout change, no flicker.
            assert len(set(area_heights)) == 1
            assert not queued_events

    try:
        run(exercise())
    finally:
        transcript.close()


def test_ctrl_c_snaps_scrolled_log_to_bottom(monkeypatch, tmp_path) -> None:
    """CTRL-C forces the worker log to jump to the bottom and resume tail
    following: a user who scrolled up to read earlier output immediately
    sees the cancellation notice and the latest worker lines."""
    queued_events = [
        [ConsoleOutput("job", "stdout", f"worker line {i}\n") for i in range(60)],
    ]

    def fake_drain(max_events=1000):
        return queued_events.pop(0) if queued_events else []

    monkeypatch.setattr(worker_app_module, "EVENT_POLL_SECONDS", 3600.0)
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(fake_drain))
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))
    monkeypatch.setattr(
        ModelWorker,
        "request_cancel",
        staticmethod(lambda operation_id: True),
    )

    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(make_state(), {0}, 1, False, transcript)

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            log = app.query_one(WorkerLogContentArea).worker_log
            app._drain_worker_events()
            await pilot.pause()
            assert log.max_scroll_y > 0
            assert log.scroll_offset.y == log.max_scroll_y

            # Scrolling up detaches from the tail.
            log.action_history_up()
            await pilot.pause()
            assert not log.follow_tail
            assert log.scroll_offset.y < log.max_scroll_y

            # CTRL-C snaps the log back to the bottom, and the appended
            # cancellation notice is visible there.
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert log.follow_tail
            assert log.scroll_offset.y == log.max_scroll_y
            document = "\n".join(line.text.plain for line in log._lines)
            assert "Cancellation requested" in document
            assert "worker line 59" in document

    try:
        run(exercise())
    finally:
        transcript.close()


def test_second_ctrl_c_hard_resets_worker_immediately(monkeypatch, tmp_path) -> None:
    cancel_calls: list[str] = []
    reset_calls: list[int] = []

    def fake_reset() -> str:
        reset_calls.append(1)
        return ""
    monkeypatch.setattr(
        ModelWorker,
        "submit_generation",
        staticmethod(lambda **_: "job"),
    )
    monkeypatch.setattr(ModelWorker, "drain_events", staticmethod(lambda **_: []))
    monkeypatch.setattr(ModelWorker, "is_alive", staticmethod(lambda: True))
    monkeypatch.setattr(
        ModelWorker,
        "request_cancel",
        staticmethod(lambda operation_id: cancel_calls.append(operation_id) or True),
    )
    monkeypatch.setattr(ModelWorker, "reset", staticmethod(fake_reset))

    transcript = GenerationTranscript(str(tmp_path / "generation.log"))
    app = GenerationApp(make_state(), {0}, 1, False, transcript)

    async def exercise() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.press("ctrl+c")
            assert cancel_calls == ["job"]
            assert app.cancel_requested
            await pilot.pause()
            # While the worker waits for a safe boundary, the header prompt
            # switches to the kill-process hint.
            assert "to kill process and stop immediately" in str(
                app.query_one("#generation-hotkey", Static).render()
            )

            # The second CTRL-C hard-resets the worker immediately, with no
            # confirmation dialog.
            await pilot.press("ctrl+c")
            await pilot.pause(0.2)

            assert reset_calls == [1]
            assert type(app.screen).__name__ == "Screen"
            assert app.terminal_result is not None
            assert app.terminal_result.status == GenerationTerminalStatus.WORKER_RESET
            # The WORKER_RESET message points at the worker's own log file.
            assert "Worker log" in app.terminal_result.message
            # Once the job has stopped, the header prompt line switches to
            # the continue hint.
            assert "to continue" in str(
                app.query_one("#generation-hotkey", Static).render()
            )
            await pilot.press("enter")

    try:
        run(exercise())
    finally:
        transcript.close()
