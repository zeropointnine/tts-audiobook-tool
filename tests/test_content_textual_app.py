import asyncio
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from tts_audiobook_tool.textual import content_textual_app
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.textual_shared import HangingIndentText


@dataclass
class StubPhraseGroup:
    presentable_text: str


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup]


class StubContentApp(ContentTextualApp):
    def __init__(self, project: StubProject) -> None:
        self.changed_phrase_indices: set[int] = set()
        self.refreshed_indices: list[int] = []
        self.refresh_batches: list[list[int]] = []
        self.committed = False
        super().__init__(
            project,  # type: ignore[arg-type]
            ["Content editor"],
            phrase_indices=[2, 0],
        )

    @property
    def has_changes(self) -> bool:
        return bool(self.changed_phrase_indices)

    def format_line(self, index: int) -> Text:
        phrase_index = self.phrase_indices[index]
        style = "#888888 reverse" if index in self.selected_indices else ""
        return Text(
            self.project.phrase_groups[phrase_index].presentable_text,
            style=style,
        )

    def action_mutate(self) -> None:
        def mutate(_visible_index: int, phrase_index: int) -> bool:
            if phrase_index in self.changed_phrase_indices:
                return False
            self.changed_phrase_indices.add(phrase_index)
            return True

        self.mutate_selected_items(mutate)

    def refresh_line(self, index: int) -> None:
        self.refreshed_indices.append(index)
        super().refresh_line(index)

    def refresh_lines(self, indices, *, reflow: bool = True) -> None:
        index_list = list(indices)
        self.refresh_batches.append(index_list)
        super().refresh_lines(index_list, reflow=reflow)

    def make_confirmation_dialog(self) -> SaveChangesDialog:
        return SaveChangesDialog("Apply content changes?")

    def commit_changes_and_exit(self) -> None:
        self.committed = True
        self.exit()


def make_app() -> tuple[StubContentApp, StubProject]:
    project = StubProject(
        [
            StubPhraseGroup("first"),
            StubPhraseGroup("hidden"),
            StubPhraseGroup("third needle"),
        ]
    )
    return StubContentApp(project), project


def test_terminal_support_failure_prints_error_feedback(monkeypatch) -> None:
    feedback_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(content_textual_app, "can_textual", lambda: False)
    monkeypatch.setattr(
        content_textual_app,
        "print_feedback",
        lambda message, **kwargs: feedback_calls.append(
            (message, kwargs.get("is_error", False))
        ),
    )

    assert ContentTextualApp.check_terminal_support() is False
    assert feedback_calls == [
        (
            "The current terminal environment does not support full-screen editor",
            True,
        )
    ]


def run(coroutine) -> None:
    asyncio.run(coroutine)


def test_base_retains_project_and_maps_options_to_phrase_groups() -> None:
    app, project = make_app()

    assert app.project is project
    assert app.phrase_indices == [2, 0]
    assert str(app.format_line(0)) == "third needle"
    assert app.find_match_indices("needle") == [0]
    assert app.find_match_indices("hidden") == []


def test_base_css_contains_shared_and_app_specific_rules() -> None:
    assert "#textual-shared-css-test" in ContentTextualApp.CSS
    assert "#line-list" in ContentTextualApp.CSS


def test_hanging_indent_ignores_non_printing_ansi_prefix_characters() -> None:
    prefix = "\x1b[31mLabel:\x1b[0m "
    text = HangingIndentText.from_ansi(
        prefix + "one two three four",
        content_start=len(prefix),
    )
    console = Console(width=14, force_terminal=False, color_system=None)

    rendered_lines = console.render_lines(text, console.options, pad=False)
    rendered = ["".join(segment.text for segment in line) for line in rendered_lines]

    assert rendered == ["Label: one two", "       three", "       four"]


def test_base_composes_header_list_status_and_superseding_find_bar() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.query_one("#header", Vertical)
            assert app.query_one("#line-list")
            status_bar = app.query_one("#status-bar", Horizontal)
            find_bar = app.query_one("#find-bar", Horizontal)
            selection_status = app.query_one("#selection-status", Static)
            assert status_bar.display is True
            assert find_bar.display is False

            await pilot.press("ctrl+f")
            assert status_bar.display is False
            assert selection_status.display is False
            assert find_bar.display is True

            await pilot.press("escape")
            assert status_bar.display is True
            assert selection_status.display is True
            assert find_bar.display is False

    run(exercise())


def test_base_parses_ansi_header_strings_when_composing_and_updating() -> None:
    app, _ = make_app()
    app.header_lines = ["\x1b[31mInitial header\x1b[0m"]

    async def exercise() -> None:
        async with app.run_test():
            header = app.query_one("#header-line-0", Static)
            initial_renderable = header.content
            assert isinstance(initial_renderable, Text)
            assert initial_renderable.plain == "Initial header"
            assert initial_renderable.spans

            app.update_header(["\x1b[32mUpdated header\x1b[0m"])
            updated_renderable = header.content
            assert isinstance(updated_renderable, Text)
            assert updated_renderable.plain == "Updated header"
            assert updated_renderable.spans

    run(exercise())


def test_base_mutates_selected_mapped_items_and_confirms_before_commit() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down")
            assert app.selected_indices == {0, 1}

            app.action_mutate()
            assert app.changed_phrase_indices == {0, 2}
            assert app.selected_indices == {1}
            assert app.refresh_batches == [[0, 1]]

            await pilot.press("escape")
            assert isinstance(app.screen, SaveChangesDialog)
            assert app.committed is False

            await pilot.press("escape")
            assert app.is_running is True
            assert app.committed is False

            await pilot.press("escape", "y")
            await pilot.pause()
            assert app.committed is True
            assert app.is_running is False

    run(exercise())


def test_plain_navigation_does_not_replace_prompts() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.refreshed_indices.clear()
            await pilot.press("down")
            assert app.selected_indices == {1}
            assert app.refreshed_indices == []

            await pilot.press("shift+up")
            assert app.selected_indices == {0, 1}
            assert app.refreshed_indices == []

    run(exercise())


def test_base_clean_escape_exits_without_confirmation() -> None:
    app, _ = make_app()

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running is False
            assert not isinstance(app.screen, SaveChangesDialog)

    run(exercise())
