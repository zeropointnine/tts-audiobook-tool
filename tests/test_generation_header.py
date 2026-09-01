import asyncio

from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Static

from tts_audiobook_tool.textual.generation_header import GenerationHeader


def run(coroutine) -> None:
    asyncio.run(coroutine)


class HeaderTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield GenerationHeader()


class QuickGenHeaderTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield GenerationHeader(title="Quick generate", id="generation-header")


def test_header_composes_five_sub_widgets() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            app = pilot.app
            title = app.query_one("#generation-title", Static)
            memory = app.query_one("#generation-memory", Static)
            status = app.query_one("#generation-status", Static)
            stats = app.query_one("#generation-stats", Static)
            hotkey = app.query_one("#generation-hotkey", Static)

            assert "Generating audio..." in str(title.render())
            assert str(memory.render()) == ""
            assert str(status.render()) == ""
            assert str(stats.render()) == ""
            assert "to interrupt" in str(hotkey.render())

    run(exercise())


def test_header_accepts_session_title() -> None:
    """The quick-generate flow passes its own title; the default otherwise."""
    async def exercise() -> None:
        async with QuickGenHeaderTestApp().run_test() as pilot:
            await pilot.pause()
            title = pilot.app.query_one("#generation-title", Static)
            assert "Quick generate" in str(title.render())
            assert "Generating audio" not in str(title.render())

    run(exercise())


def test_update_status_prefixes_label() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_status("Loading model")
            render = header.query_one("#generation-status", Static).render()
            assert str(render) == "Status: Loading model"
            assert isinstance(render, Content)
            assert len(render.spans) == 1

    run(exercise())


def test_update_stats_formats_processed_total_elapsed() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_stats(1, 2, 30)
            assert (
                str(header.query_one("#generation-stats", Static).render())
                == "Processed: 1/2  Elapsed: 30s"
            )

    run(exercise())


def test_update_stats_renders_eta_before_elapsed() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_stats(37, 512, 305, eta_seconds=5405.4)
            assert (
                str(header.query_one("#generation-stats", Static).render())
                == "Processed: 37/512  ETA: 1h30m10s  Elapsed: 5m05s"
            )

    run(exercise())


def test_update_stats_rounds_eta_to_nearest_10() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            # 84s rounds down to 80s.
            header.update_stats(1, 4, 30, eta_seconds=84.0)
            assert (
                str(header.query_one("#generation-stats", Static).render())
                == "Processed: 1/4  ETA: 1m20s  Elapsed: 30s"
            )
            # 306s rounds up to 310s.
            header.update_stats(1, 4, 30, eta_seconds=306.0)
            assert (
                str(header.query_one("#generation-stats", Static).render())
                == "Processed: 1/4  ETA: 5m10s  Elapsed: 30s"
            )

    run(exercise())


def test_update_stats_renders_zero_eta() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_stats(2, 2, 30, eta_seconds=0.0)
            assert (
                str(header.query_one("#generation-stats", Static).render())
                == "Processed: 2/2  ETA: 0s  Elapsed: 30s"
            )

    run(exercise())


def test_update_memory_text(monkeypatch) -> None:
    monkeypatch.setattr(
        "tts_audiobook_tool.app_support.make_memory_string",
        lambda **kwargs: "VRAM: 1.0GB/8.0GB, RAM: 4.0GB/16.0GB",
    )

    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_memory_text()
            render = header.query_one("#generation-memory", Static).render()
            assert str(render) == "VRAM: 1.0GB/8.0GB, RAM: 4.0GB/16.0GB"

    run(exercise())


def test_update_memory_text_noop_when_empty(monkeypatch) -> None:
    monkeypatch.setattr("tts_audiobook_tool.app_support.make_memory_string", lambda **kwargs: "")

    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_memory_text()
            assert str(header.query_one("#generation-memory", Static).render()) == ""

    run(exercise())


def test_update_hotkey_auto_return_is_blank() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_hotkey("auto_return")
            assert str(header.query_one("#generation-hotkey", Static).render()) == ""

    run(exercise())


def test_update_hotkey_auto_continue_announces_concatenation() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_hotkey("auto_continue")
            assert "Proceeding to concatenation..." in str(
                header.query_one("#generation-hotkey", Static).render()
            )

    run(exercise())


def test_update_hotkey_renders_mode_as_ansi() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(GenerationHeader)
            header.update_hotkey("default")
            render = header.query_one("#generation-hotkey", Static).render()
            assert "CTRL-C" in str(render)
            # The ANSI codes became styled spans rather than literal escapes.
            assert isinstance(render, Content)
            assert len(render.spans) > 0

    run(exercise())
