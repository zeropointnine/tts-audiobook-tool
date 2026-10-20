from textual.app import App, ComposeResult
from textual.content import Content
from textual.geometry import Size
from textual.widgets import Rule, Static

from tts_audiobook_tool.textual.real_time_playback_header import (
    RealTimePlaybackHeader,
    RealTimePlaybackSourceText,
)
from textual_editor_stubs import run


class HeaderTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield RealTimePlaybackHeader()


class SourceTextTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield RealTimePlaybackSourceText()


def test_header_composes_and_updates_realtime_stats() -> None:
    async def exercise() -> None:
        async with HeaderTestApp().run_test() as pilot:
            await pilot.pause()
            header = pilot.app.query_one(RealTimePlaybackHeader)
            assert "Realtime audiobook playback" in str(
                header.query_one("#realtime-title", Static).render()
            )

            header.update_status("Generating audio")
            header.update_stats(2, 5, 12.3)
            status = header.query_one("#realtime-status", Static).render()
            assert str(status) == "Status: Generating audio"
            assert isinstance(status, Content)
            assert len(status.spans) == 1
            stats = str(header.query_one("#realtime-stats", Static).render())
            assert "Processed: 2/5" in stats
            assert "Buffer: 12.3s" in stats
            assert "Elapsed:" not in stats

            header.update_stats(2, 5, 0.0)
            zero_buffer = header.query_one("#realtime-stats", Static).render()
            assert "Buffer: 0.0s" in str(zero_buffer)
            assert isinstance(zero_buffer, Content)
            assert len(zero_buffer.spans) > 0

            header.update_stats(2, 5, 0.0, zero_buffer_is_error=False)
            finished_buffer = header.query_one("#realtime-stats", Static).render()
            assert isinstance(finished_buffer, Content)
            # A finished (expected) zero buffer shows the same text but is
            # rendered in the OK color rather than being flagged as an error.
            assert str(finished_buffer) == str(zero_buffer)
            assert len(finished_buffer.spans) == 1
            assert len(zero_buffer.spans) == 1
            assert finished_buffer.spans[0].style != zero_buffer.spans[0].style

            header.update_hotkey("awaiting_continue")
            assert "ENTER" in str(
                header.query_one("#realtime-hotkey", Static).render()
            )

    run(exercise())


def test_source_text_band_layout() -> None:
    async def exercise() -> None:
        async with SourceTextTestApp().run_test() as pilot:
            await pilot.pause()
            band = pilot.app.query_one(RealTimePlaybackSourceText)
            # 2-line text area plus the 1-line closing rule (margin zeroed).
            assert band.size == Size(80, 3)
            # The widget size excludes the 1-cell side padding; the outer
            # box fills the band width.
            text = band.query_one("#realtime-source-text", Static)
            assert text.size == Size(78, 2)
            assert text.outer_size == Size(80, 2)
            assert band.query_one("#realtime-source-divider", Rule).size == Size(80, 1)

    run(exercise())


def test_source_text_wraps_to_two_lines_and_ellipsizes_overflow() -> None:
    async def exercise() -> None:
        async with SourceTextTestApp().run_test() as pilot:
            await pilot.pause()
            band = pilot.app.query_one(RealTimePlaybackSourceText)
            text = band.query_one("#realtime-source-text", Static)
            # The band's text area is 78 cells wide (80 minus 1-cell padding
            # on each side); content is colored (COL_ACCENT) so spans exist.
            assert text.content_size.width == 78

            band.update_playing_text("The quick brown fox")
            active = text.render()
            assert str(active) == "The quick brown fox"
            assert isinstance(active, Content)
            assert len(active.spans) > 0
            assert str(active.spans[0].style) == "rgb(255,170,68)"  # COL_ACCENT #ffaa44

            # Two wrapped lines fit with no ellipsis.
            band.update_playing_text("a" * 100)
            assert str(text.render()) == "a" * 78 + "\n" + "a" * 22

            # Overflow past the second line ends in an ellipsis on it.
            band.update_playing_text("a" * 200)
            assert str(text.render()) == "a" * 78 + "\n" + "a" * 75 + "..."

            band.update_playing_text("")
            assert str(text.render()) == ""

    run(exercise())


def test_source_text_single_line_is_bottom_aligned() -> None:
    async def exercise() -> None:
        async with SourceTextTestApp().run_test() as pilot:
            await pilot.pause()
            band = pilot.app.query_one(RealTimePlaybackSourceText)
            text = band.query_one("#realtime-source-text", Static)

            def rows() -> list[str]:
                return [text.render_line(y).text.strip() for y in range(2)]

            band.update_playing_text("The quick brown fox")
            await pilot.pause()
            assert rows() == ["", "The quick brown fox"]

            band.update_playing_text("a" * 100)
            await pilot.pause()
            assert rows() == ["a" * 78, "a" * 22]

    run(exercise())
