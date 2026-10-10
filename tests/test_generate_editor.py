import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from rich.style import Style
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Input, Static

import tts_audiobook_tool.textual.generate_editor as generate_editor_module
from tts_audiobook_tool.app_types import Book, BookSection, SttVariant
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool.project_support.segment_transcript_util import (
    SegmentTranscriptUtil,
)
from tts_audiobook_tool.state import State
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.textual.filter_dialog import FilterDialog
from tts_audiobook_tool.textual.content_textual_app import (
    EditorClosed,
    EditorSaveFailed,
)
from tts_audiobook_tool.textual.generate_editor import (
    FilterType,
    GenerateEditor,
    GeneratePhraseGroupItem,
    GenerateSectionItem,
)
from tts_audiobook_tool.textual.manual_selection_dialog import ManualSelectionDialog
from tts_audiobook_tool.textual.save_changes_dialog import SaveChangesDialog
from tts_audiobook_tool.textual.segment_info_dialog import SegmentInfoDialog
from tts_audiobook_tool.textual.textual_shared import NonWrappingOptionList
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.textual.alert_dialog import AlertDialog
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
)
from tts_audiobook_tool.textual import worker_app as worker_app_module
from tts_audiobook_tool.textual.quick_gen_modal import (
    QuickGenJob,
    QuickGenModal,
    QuickGenResult,
    QuickGenReview,
)
from textual_editor_stubs import (
    make_phrase_group,
    run,
    StubPhraseGroup,
    StubProject,
    StubSoundSegment,
    StubSoundSegments,
)


def make_state(project: StubProject) -> State:
    return cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(
                stt_variant=SttVariant.DISABLED,
            ),
        ),
    )


def make_app(
    num_lines: int = 12,
    generated_indices: set[int] | None = None,
) -> tuple[GenerateEditor, StubProject]:
    if generated_indices is None:
        generated_indices = set(range(num_lines))
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)],
        StubSoundSegments(
            {
                index: [StubSoundSegment(f"segment-{index}.flac")]
                for index in generated_indices
            }
        ),
    )
    app = GenerateEditor(make_state(project))
    app.load_content()
    return app, project


def make_sectioned_app(
    sections: list[BookSection],
    generated_indices: set[int] | None = None,
) -> tuple[GenerateEditor, StubProject]:
    phrase_groups = [
        phrase_group
        for section in sections
        for phrase_group in section.phrase_groups
    ]
    if generated_indices is None:
        generated_indices = set(range(len(phrase_groups)))
    project = StubProject(
        cast(list[StubPhraseGroup | PhraseGroup], phrase_groups),
        StubSoundSegments(
            {
                index: [StubSoundSegment(f"segment-{index}.flac")]
                for index in generated_indices
            }
        ),
        book=Book(sections=sections),
    )
    app = GenerateEditor(make_state(project))
    app.load_content()
    return app, project


def test_generate_editor_projects_sections_and_suppresses_single_section() -> None:
    single_app, _ = make_sectioned_app(
        [BookSection(title="Only", phrase_groups=[make_phrase_group("One.")])]
    )
    app, _ = make_sectioned_app(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Three.")]),
        ]
    )

    assert all(isinstance(item, GeneratePhraseGroupItem) for item in single_app.list_items)
    assert [type(item) for item in app.list_items] == [
        GenerateSectionItem,
        GeneratePhraseGroupItem,
        GeneratePhraseGroupItem,
        GenerateSectionItem,
        GeneratePhraseGroupItem,
    ]
    assert str(app.format_line(0)) == "\nSection 1/2: Opening (2 lines)\n\n"
    assert str(app.format_line(3)) == "\nSection 2/2: Middle (1 line)\n\n"
    assert app.format_line(0).spans == []


def test_generate_filter_omits_empty_sections_and_counts_visible_matches() -> None:
    app, _ = make_sectioned_app(
        [
            BookSection(
                title="Opening",
                phrase_groups=[make_phrase_group("One."), make_phrase_group("Two.")],
            ),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Three.")]),
            BookSection(title="Empty", phrase_groups=[]),
        ],
        generated_indices={1},
    )

    app.filter_type = FilterType.GENERATED
    app.phrase_indices = app.get_filtered_phrase_indices()

    assert [type(item) for item in app.list_items] == [
        GenerateSectionItem,
        GeneratePhraseGroupItem,
    ]
    assert str(app.format_line(0)) == "\nSection 1/3: Opening (1 line)\n\n"
    assert app.content_line_index(app.phrase_indices[1]) == 1


def test_generate_section_rows_are_searchable_but_non_actionable() -> None:
    app, project = make_sectioned_app(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Needle", phrase_groups=[make_phrase_group("Two.")]),
        ],
        generated_indices=set(),
    )

    assert app.find_match_indices("section 2/2: needle") == [2]
    assert app.find_match_indices("00002") == [3]
    assert app.find_match_indices("00003") == []

    async def exercise() -> None:
        async with app.run_test() as pilot:
            assert app.highlighted_content_line_index() is None
            await pilot.press("space", "p", "q", "i", "x")
            assert app.staged_queued_indices == set()
            assert project.sound_segments.deleted_index_batches == []

            # A selection spanning a section row keeps the row out of the staged set
            await pilot.press("home", "down", "shift+down", "shift+down")
            await pilot.press("space")
            assert app.staged_queued_indices == {0, 1}
            assert app.selected_indices == {3}

    run(exercise())


def test_queue_applies_to_selected_phrase_when_section_is_highlighted() -> None:
    app, _ = make_sectioned_app(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ],
        generated_indices=set(),
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "shift+up")
            assert app.selected_index == 0
            assert app.selected_indices == {0, 1}
            assert app.highlighted_content_line_index() is None

            await pilot.press("space")
            assert app.staged_queued_indices == {0}
            assert app.selected_indices == {0}

    run(exercise())


def test_delete_with_section_highlight_applies_to_selected_generated_phrase() -> None:
    app, project = make_sectioned_app(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(title="Middle", phrase_groups=[make_phrase_group("Two.")]),
        ],
        generated_indices={0},
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "shift+up", "x")
            assert isinstance(app.screen, SaveChangesDialog)
            assert app.screen.copy_lines == ["Delete 1 generated sound segment?"]

            await pilot.press("y")
            await pilot.pause()
            assert project.sound_segments.deleted_index_batches == [{0}]

    run(exercise())


def test_initial_rows_reuse_single_segment_status_snapshot() -> None:
    num_lines = 1_000
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(num_lines)],
        StubSoundSegments({}),
    )
    app = GenerateEditor(make_state(project))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert project.sound_segments.best_item_call_count == num_lines

    run(exercise())


@pytest.mark.parametrize(
    ("line_count", "range_string", "expected_indices"),
    [
        pytest.param(6, "2, 4-5", {1, 3, 4}, id="list-and-range"),
        pytest.param(3, "all", {0, 1, 2}, id="all"),
        pytest.param(3, "none", set(), id="none"),
    ],
)
def test_initial_queue_is_loaded_from_project_generation_range(
    line_count: int, range_string: str, expected_indices: set[int]
) -> None:
    app, project = make_app(line_count)
    project.generate_range_string = range_string

    app = GenerateEditor(make_state(project))
    app.load_content()

    assert app.original_queued_indices == expected_indices
    assert app.staged_queued_indices == expected_indices


def patch_quick_job(
    project: StubProject,
    *,
    problems: tuple[str, ...] = (),
    produce_segment: bool = True,
    status: GenerationTerminalStatus = GenerationTerminalStatus.COMPLETED,
):
    """Patch the editor's job factory with a fake staged worker job.

    The job's submit stands in for the model worker: it writes a staged
    segment file into a real per-job staging directory, then a canned finish
    event is delivered when the modal polls the worker. Accepting it is faked
    against the stub catalog, and review playback is stubbed out.
    """
    submitted: list[int] = []
    staging_root = Path(tempfile.mkdtemp(prefix="quick-gen-test-"))

    def make_job(_state, phrase_index: int) -> QuickGenJob:
        staging_dir = staging_root / f"job-{len(submitted)}"

        def submit() -> str:
            submitted.append(phrase_index)
            if produce_segment:
                staging_dir.mkdir(parents=True)
                write_staged_segment(staging_dir, phrase_index)
            return "job"

        return QuickGenJob(
            title=f"Quick generate - line {phrase_index + 1}",
            submit=submit,
            problems=problems,
            review=QuickGenReview(str(staging_dir), phrase_index),
        )

    def accept(project_arg, phrase_index: int, staged_path: Path) -> str:
        accepted.append((phrase_index, staged_path.name))
        project_arg.sound_segments.sound_segments_map[phrase_index] = [
            StubSoundSegment("quick.flac")
        ]
        return ""

    accepted: list[tuple[int, str]] = []
    events = [[GenerationFinished("job", status, "")]]
    patches = [
        patch.object(generate_editor_module, "make_generation_job", make_job),
        patch.object(generate_editor_module, "accept_staged_segment", accept),
        patch.object(
            ModelWorker,
            "drain_events",
            staticmethod(lambda max_events=1000: events.pop(0) if events else []),
        ),
        patch.object(worker_app_module, "EVENT_POLL_SECONDS", 0.02),
        patch.object(
            PlaySoundUtil, "play_sound_file_async", return_value=("review-sound", "")
        ),
        patch.object(PlaySoundUtil, "current_sound_id", return_value="review-sound"),
        patch.object(PlaySoundUtil, "stop_sound_async"),
    ]
    submitted_info = SimpleNamespace(
        indices=submitted, accepted=accepted, staging_root=staging_root
    )
    return submitted_info, patches


def write_staged_segment(staging_dir: Path, phrase_index: int) -> Path:
    path = staging_dir / (
        f"[{phrase_index + 1:05d}] [0123456789abcdef] [none] [voice] new.flac"
    )
    path.write_bytes(b"flac")
    return path


def run_with_patches(patches, coroutine) -> None:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        run(coroutine)


def test_segment_actions_are_ignored_before_deferred_content_loads() -> None:
    project = StubProject(
        [StubPhraseGroup("Line 1")],
        StubSoundSegments({0: [StubSoundSegment("segment-0.flac")]}),
    )
    app = GenerateEditor(make_state(project))

    app.action_toggle_queued()
    app.action_show_filter()
    app.action_play_sound()
    app.action_quick_generate()
    app.action_show_info()
    app.action_delete_generated()

    assert app.all_phrase_indices == []
    assert app.staged_queued_indices == set()
    assert app.filter_type == FilterType.ALL
    assert project.save_calls == []
    assert app.staged_queued_indices == app.original_queued_indices


def test_q_preflight_problems_show_in_modal_without_saving_or_deleting() -> None:
    # Problems (readiness, voices, busy worker) are listed in the modal itself
    # and leave the project untouched: no queue save, no deleted segment.
    app, project = make_app(2)
    info, patches = patch_quick_job(
        project, problems=("Choose a voice", "Configure the model")
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "q")
            await pilot.pause(0.2)
            assert isinstance(app.screen, QuickGenModal)
            assert project.save_calls == []
            assert project.sound_segments.deleted_index_batches == []
            assert app.is_running is True
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, QuickGenModal)
            assert app.modal_session_active is False

    run_with_patches(patches, exercise())
    assert info.indices == []


def test_q_leaves_project_untouched_while_job_runs() -> None:
    # Quick gen is staged and in place: the editor stays running underneath
    # the modal, the line's old sound survives, and the staged queue edit is
    # not saved behind the user's back.
    app, project = make_app(3, {1, 2})
    info, patches = patch_quick_job(project, status=GenerationTerminalStatus.FAILED)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("space", "down", "q")
            await pilot.pause(0.3)
            assert isinstance(app.screen, QuickGenModal)
            assert app.is_running is True
            assert project.save_calls == []
            assert project.sound_segments.deleted_index_batches == []
            assert 1 in app.generated_indices

    run_with_patches(patches, exercise())
    assert info.indices == [1]


def test_q_uses_highlighted_project_index_under_filter() -> None:
    app, project = make_app(5, {1, 3})
    info, patches = patch_quick_job(project, status=GenerationTerminalStatus.FAILED)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f", "3", "down", "q")
            await pilot.pause(0.3)

    run_with_patches(patches, exercise())
    assert info.indices == [3]


def test_quick_gen_review_enter_keeps_new_sound_and_refreshes_row() -> None:
    # A staged result waits in the review panel; ENTER keeps it, and the same
    # editor instance resumes with the selection intact and the row refreshed.
    app, project = make_app(4, {0, 1, 2, 3})
    info, patches = patch_quick_job(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "down", "q")
            await pilot.pause(0.6)
            assert isinstance(app.screen, QuickGenModal)
            assert app.screen.review_pending
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, QuickGenModal)
            assert app.is_running is True
            assert app.selected_index == 2
            assert app.selected_indices == {2}
            assert app.modal_session_active is False
            assert 2 in app.generated_indices

    run_with_patches(patches, exercise())
    assert info.indices == [2]
    assert [index for index, _ in info.accepted] == [2]
    # The job's staging directory is cleaned up once the decision is applied.
    assert not (info.staging_root / "job-1").exists()


def test_quick_gen_review_escape_discards_new_sound() -> None:
    # ESC in the review panel discards the staged take; the old audio stays.
    app, project = make_app(3, {0, 1, 2})
    info, patches = patch_quick_job(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "q")
            await pilot.pause(0.6)
            assert app.screen.review_pending  # type: ignore[attr-defined]
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, QuickGenModal)
            assert 1 in app.generated_indices

    run_with_patches(patches, exercise())
    assert info.accepted == []
    assert project.sound_segments.deleted_index_batches == []
    assert not any(info.staging_root.iterdir())


def test_quick_gen_preserves_multiple_selection_anchor_and_scroll() -> None:
    # An unchanged row mapping must not collapse a backwards selection or
    # move the viewport while the regenerated row is refreshed.
    app, project = make_app(40)
    _, patches = patch_quick_job(project)

    async def exercise() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("end", "up", "shift+up", "shift+up")
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            old_scroll_offset = option_list.scroll_offset
            assert old_scroll_offset.y > 0
            assert app.selected_index == 36
            assert app.selected_indices == {36, 37, 38}
            assert app.selection_anchor_index == 38

            await pilot.press("q")
            await pilot.pause(0.6)
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, QuickGenModal)
            assert app.selected_index == 36
            assert app.selected_indices == {36, 37, 38}
            assert app.selection_anchor_index == 38
            assert option_list.inactive_selection_indices == {37, 38}
            assert option_list.scroll_offset == old_scroll_offset

            await pilot.press("shift+up")
            assert app.selected_indices == {35, 36, 37, 38}
            assert app.selection_anchor_index == 38

    run_with_patches(patches, exercise())


@pytest.mark.parametrize(
    ("filter_type", "expected_highlight"),
    [
        (FilterType.UNGENERATED, 3),
        (FilterType.FAILED, 4),
        (FilterType.GENERATED_WITH_ERRORS, 4),
    ],
)
def test_kept_quick_gen_selects_nearest_remaining_line_when_filtered_out(
    filter_type: FilterType, expected_highlight: int
) -> None:
    # A kept clean segment leaves these filters; the highlight moves nearby
    # instead of jumping to row 0.
    generated_indices = set() if filter_type == FilterType.UNGENERATED else {0, 2, 4}
    app, project = make_app(5, generated_indices)
    for index in generated_indices:
        segment = project.sound_segments.sound_segments_map[index][0]
        segment.num_errors = 1
        project.sound_segments.failed_segment_files.add(segment.file_name)
    app.refresh_phrase_classifications()
    _, patches = patch_quick_job(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.handle_filter_selection(filter_type)
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            option_list.highlighted = app.phrase_indices.index(2)
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause(0.6)
            await pilot.press("enter")
            await pilot.pause()

            assert app.filter_type == filter_type
            assert 2 not in app.phrase_indices
            assert app.highlighted_content_line_index() == expected_highlight
            assert app.selected_indices == {app.selected_index}
            assert app.selection_anchor_index == app.selected_index

    run_with_patches(patches, exercise())


def test_quick_gen_preserves_surviving_selection_across_section_row_removal() -> None:
    # Dropping the first section shifts every backing row; retain phrase and
    # section identities, including the surviving anchor, rather than positions.
    app, project = make_sectioned_app(
        [
            BookSection(title="Opening", phrase_groups=[make_phrase_group("One.")]),
            BookSection(
                title="Middle",
                phrase_groups=[
                    make_phrase_group("Two."),
                    make_phrase_group("Three."),
                    make_phrase_group("Four."),
                    make_phrase_group("Five."),
                ],
            ),
        ],
        generated_indices=set(),
    )
    _, patches = patch_quick_job(project)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            app.handle_filter_selection(FilterType.UNGENERATED)
            await pilot.press(
                "end", "shift+up", "shift+up", "shift+up", "shift+up", "shift+up"
            )
            assert app.selected_content_line_indices() == {0, 1, 2, 3, 4}
            assert app.selection_anchor_index == 6
            assert app.highlighted_content_line_index() == 0

            await pilot.press("q")
            await pilot.pause(0.6)
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.list_items) == 5
            assert isinstance(app.list_items[0], GenerateSectionItem)
            assert app.highlighted_content_line_index() == 1
            assert app.selected_content_line_indices() == {1, 2, 3, 4}
            assert app.selected_indices == {0, 1, 2, 3, 4}
            assert app.selection_anchor_index == 4

    run_with_patches(patches, exercise())


def test_failed_keep_leaves_line_queued_and_ungenerated() -> None:
    # A keep that fails (and rolls back) reports the error but must not drop
    # the still-ungenerated line from the queue saved on exit.
    app, _project = make_app(2, set())
    app.staged_queued_indices = {0}
    with (
        patch.object(
            generate_editor_module,
            "accept_staged_segment",
            return_value="Couldn't keep new sound: OSError: disk full",
        ),
        patch.object(app, "notify") as notify,
    ):
        app.handle_quick_gen_result(
            QuickGenResult(
                GenerationTerminalStatus.COMPLETED,
                staged_path="/tmp/staging/new.flac",
                accepted=True,
            ),
            0,
        )
    notify.assert_called_once()
    assert app.staged_queued_indices == {0}
    assert 0 in app.ungenerated_indices


def test_kept_quick_gen_unqueues_line_when_no_filtered_rows_remain() -> None:
    # An empty filtered view has no highlight at all; keeping a staged take
    # still marks the line generated and drops it from the staged queue.
    app, project = make_app(1, set())
    app.staged_queued_indices = {0}
    app.handle_filter_selection(FilterType.UNGENERATED)

    def accept(project_arg, phrase_index: int, staged_path: Path) -> str:
        project_arg.sound_segments.sound_segments_map[phrase_index] = [
            StubSoundSegment("quick.flac")
        ]
        return ""

    with patch.object(generate_editor_module, "accept_staged_segment", accept):
        app.handle_quick_gen_result(
            QuickGenResult(
                GenerationTerminalStatus.COMPLETED,
                staged_path="/tmp/staging/new.flac",
                accepted=True,
            ),
            0,
        )
    assert app.phrase_indices == []
    assert app.selected_index is None
    assert app.selected_indices == set()
    assert app.selection_anchor_index is None
    assert 0 in app.generated_indices
    assert app.staged_queued_indices == set()


def test_failed_quick_gen_keeps_original_sound_and_shows_no_review() -> None:
    # Without a staged candidate there is nothing to review: the modal shows
    # its summary, ESC closes it, and the line keeps its original audio.
    app, project = make_app(3, {0, 1, 2})
    info, patches = patch_quick_job(
        project, produce_segment=False, status=GenerationTerminalStatus.FAILED
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "q")
            await pilot.pause(0.6)
            assert isinstance(app.screen, QuickGenModal)
            assert not app.screen.review_pending
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, QuickGenModal)
            assert 1 in app.generated_indices
            PlaySoundUtil.play_sound_file_async.assert_not_called()  # type: ignore[attr-defined]

    run_with_patches(patches, exercise())
    assert info.accepted == []


def test_editor_find_and_select_all_are_blocked_while_modal_is_open() -> None:
    # The editor's priority bindings resolve before a modal sees the key, so
    # they are disabled for the duration of the session.
    app, project = make_app(2)
    _, patches = patch_quick_job(project, problems=("Choose a voice",))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause(0.2)
            assert isinstance(app.screen, QuickGenModal)
            await pilot.press("ctrl+f", "ctrl+a")
            await pilot.pause()
            assert app.find_active is False
            assert app.selected_indices == {0}

    run_with_patches(patches, exercise())


def test_x_ignores_ungenerated_selection() -> None:
    app, _ = make_app(2, set())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x")
            assert not isinstance(app.screen, SaveChangesDialog)

    run(exercise())


def test_x_counts_generated_selected_rows_and_no_preserves_segments() -> None:
    app, project = make_app(4, {0, 2})
    project.sound_segments.sound_segments_map[0].append(
        StubSoundSegment("redundant-segment-0.flac")
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+a", "x")
            assert isinstance(app.screen, SaveChangesDialog)
            assert app.screen.copy_lines == [
                "Delete 2 generated sound segments?"
            ]
            await pilot.press("n")
            assert project.sound_segments.deleted_index_batches == []
            assert set(project.sound_segments.sound_segments_map) == {0, 2}

    run(exercise())


def test_x_confirm_deletes_all_files_for_generated_rows_and_refreshes_state(
    monkeypatch,
) -> None:
    app, project = make_app(3, {0, 2})
    project.sound_segments.sound_segments_map[0].append(
        StubSoundSegment("redundant-segment-0.flac")
    )

    async def exercise() -> None:
        async with app.run_test() as pilot:
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            retained_option = option_list.get_option("generate-phrase-1")
            formatted_indices: list[int] = []
            original_format_line = app.format_line

            def record_format_line(index: int):
                formatted_indices.append(index)
                return original_format_line(index)

            monkeypatch.setattr(app, "format_line", record_format_line)
            await pilot.press("x")
            assert isinstance(app.screen, SaveChangesDialog)
            assert app.screen.copy_lines == ["Delete 1 generated sound segment?"]
            await pilot.press("y")
            await pilot.pause()

            assert project.sound_segments.deleted_index_batches == [{0}]
            assert project.sound_segments.invalidation_count == 1
            assert formatted_indices == [0]
            assert str(app.format_line(0)).startswith("00001 [         ]")
            assert option_list.get_option("generate-phrase-1") is retained_option
            assert str(app.query_one("#status-right", Static).render()) == (
                "0 lines queued for generation"
            )

    run(exercise())


def test_x_escape_cancels_deletion() -> None:
    app, project = make_app(1)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("x", "escape")
            assert project.sound_segments.deleted_index_batches == []
            assert isinstance(project.sound_segments.get_best_item_for(0), StubSoundSegment)

    run(exercise())


def test_confirmed_delete_reapplies_active_filter_near_selection() -> None:
    app, project = make_app(5, {0, 2, 4})
    for index in (0, 2, 4):
        project.sound_segments.sound_segments_map[index][0].num_errors = 1

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f", "4", "down", "x", "y")
            await pilot.pause()

            assert app.filter_type == FilterType.GENERATED_WITH_ERRORS
            assert app.phrase_indices == [0, 4]
            assert app.selected_index == 1
            assert app.phrase_indices[app.selected_index] == 4

    run(exercise())


def test_confirmed_delete_stops_playback_for_an_affected_line() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ),
            patch.object(PlaySoundUtil, "current_sound_id", return_value="sound-id"),
            patch.object(PlaySoundUtil, "stop_sound_async") as stop_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("p", "x", "y")
                await pilot.pause()

                stop_sound.assert_called_once_with()
                assert app.playing_sound_id == ""
                assert app.playing_phrase_index is None

    run(exercise())


def test_all_phrases_are_displayed_once_in_project_order() -> None:
    app, project = make_app(6, {5, 1})
    project.sound_segments.sound_segments_map[5].append(
        StubSoundSegment("second-generation-for-line-6.flac")
    )
    project.sound_segments.sound_segments_map[-1] = [StubSoundSegment("invalid.flac")]
    project.sound_segments.sound_segments_map[12] = [StubSoundSegment("stale.flac")]

    app = GenerateEditor(make_state(project))
    app.load_content()

    # Stale/out-of-range segment entries create no rows; every phrase appears once.
    assert app.phrase_indices == [0, 1, 2, 3, 4, 5]


def test_best_segment_word_error_count_replaces_queue_status() -> None:
    app, project = make_app(3)
    project.sound_segments.sound_segments_map[0] = [
        StubSoundSegment("first.flac", num_errors=3),
        StubSoundSegment("best.flac", num_errors=1),
        StubSoundSegment("unknown.flac"),
    ]
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("zero-errors.flac", num_errors=0),
    ]

    assert str(app.format_line(0)) == "00001 [generated] [word errors: 1] Line 1"
    formatted_line = app.format_line(0)
    word_errors_start = str(formatted_line).index("[word errors: 1]")
    word_errors_end = word_errors_start + len("[word errors: 1]")
    assert any(
        span.start < word_errors_end
        and span.end > word_errors_start
        and isinstance(span.style, Style)
        and span.style.color is not None
        and span.style.color.get_truecolor() == (136, 136, 136)
        for span in formatted_line.spans
    )
    assert str(app.format_line(1)) == "00002 [generated] [word errors: 0] Line 2"
    assert str(app.format_line(2)) == "00003 [generated] Line 3"


def test_failed_word_error_count_has_error_colored_asterisk() -> None:
    app, project = make_app(1)
    failed_segment = StubSoundSegment("failed.flac", num_errors=2)
    project.sound_segments.sound_segments_map[0] = [failed_segment]
    project.sound_segments.failed_segment_files.add(failed_segment.file_name)

    formatted_line = app.format_line(0)

    assert str(formatted_line) == "00001 [generated] [word errors: 2 *] Line 1"
    asterisk_span = next(
        span
        for span in formatted_line.spans
        if str(formatted_line)[span.start : span.end] == "*"
    )
    assert isinstance(asterisk_span.style, Style)
    assert asterisk_span.style.color
    assert asterisk_span.style.color.get_truecolor() == (255, 0, 0)


def test_queue_toggle_applies_to_every_selected_phrase() -> None:
    app, _ = make_app(6, set())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("shift+down", "shift+down", "space")
            assert app.staged_queued_indices == {0, 1, 2}
            first_line = app.format_line(0)
            assert str(first_line).startswith("00001 [Queued   ]")
            queued_span = next(
                span
                for span in first_line.spans
                if "Queued" in str(first_line)[span.start : span.end]
            )
            assert queued_span.style
            assert str(app.query_one("#status-right", Static).render()) == (
                "3 lines queued for generation"
            )
            assert app.selected_indices == {2}
            assert app.staged_queued_indices != app.original_queued_indices

            await pilot.press("shift+up", "space")
            assert app.staged_queued_indices == {0}
            assert str(app.format_line(1)).startswith("00002 [         ]")
            assert app.selected_indices == {1}

            await pilot.press("home", "space")
            assert app.staged_queued_indices == set()
            assert str(app.query_one("#status-right", Static).render()) == (
                "0 lines queued for generation"
            )
            assert app.staged_queued_indices == app.original_queued_indices

    run(exercise())


def test_queue_toggle_refreshes_only_selected_rows_without_reflow() -> None:
    app, project = make_app(1_000, set())

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            project.sound_segments.best_item_call_count = 0
            with patch.object(app, "refresh_lines", wraps=app.refresh_lines) as refresh:
                await pilot.press("space")

            refresh.assert_called_once()
            assert refresh.call_args.args[0] == [0]
            assert refresh.call_args.kwargs == {"reflow": False}
            assert project.sound_segments.best_item_call_count == 1
            assert app.queued_ungenerated_count == 1

    run(exercise())


def test_pinned_status_update_does_not_rescan_segment_state() -> None:
    app, project = make_app(1_000, set())
    project.sound_segments.best_item_call_count = 0
    app.filter_type = FilterType.UNGENERATED
    app.queued_ungenerated_count = 400

    app.update_pinned_status()

    assert project.sound_segments.best_item_call_count == 0
    assert app.pinned_status.left == "Showing ungenerated lines (1000)"
    assert app.pinned_status.right == "400 lines queued for generation"


def test_queue_toggle_does_not_queue_lines_with_sound_segments() -> None:
    app, _ = make_app(3, {1})

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down", "space")
            assert app.staged_queued_indices == set()

            await pilot.press("home", "shift+down", "shift+down", "space")
            assert app.staged_queued_indices == {0, 2}
            assert str(app.query_one("#status-right", Static).render()) == (
                "2 lines queued for generation (all)"
            )

    run(exercise())


def test_queue_toggle_ignores_generated_lines_and_clears_their_flags() -> None:
    app, _ = make_app(3, {1})
    app.staged_queued_indices = {1}

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("ctrl+a", "space")
            assert app.staged_queued_indices == {0, 2}

            await pilot.press("ctrl+a", "space")
            assert app.staged_queued_indices == set()

            app.staged_queued_indices.add(1)
            await pilot.press("down", "space")
            assert app.staged_queued_indices == set()

    run(exercise())


def test_f_opens_filter_dialog_with_current_filter_and_all_options() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f")
            assert isinstance(app.screen, FilterDialog)
            # Per-option counts are the classification output; labels come from
            # FilterType.menu_label rather than being restated here.
            assert [
                str(app.screen.query_one(f"#filter-option-{number}", Static).render())
                for number in range(1, 6)
            ] == [
                f"[{number}] {filter_type.menu_label} ({count})"
                + (" (selected)" if filter_type is app.filter_type else "")
                for number, (filter_type, count) in enumerate(
                    zip(FilterType, (2, 0, 2, 0, 0)), start=1
                )
            ]

    run(exercise())


def test_m_opens_focused_manual_selection_dialog_and_escape_cancels() -> None:
    app, _ = make_app(8)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            assert isinstance(app.screen, ManualSelectionDialog)
            assert app.screen.query_one("#manual-selection-input", Input).has_focus

            await pilot.press("escape")
            assert not isinstance(app.screen, ManualSelectionDialog)
            assert app.selected_indices == {0}

    run(exercise())


def test_manual_selection_dialog_shows_syntax_errors_and_stays_open() -> None:
    app, _ = make_app(8)

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("m")
            app.screen.query_one("#manual-selection-input", Input).value = "0, bad, 20"
            await pilot.press("enter")

            # Out-of-range values (0, 20) are silently discarded; only the
            # syntax error "bad" is reported
            assert isinstance(app.screen, ManualSelectionDialog)
            assert str(
                app.screen.query_one("#manual-selection-error", Static).render()
            ) == "Bad value: bad"

    run(exercise())


def test_manual_selection_uses_project_lines_and_ignores_filtered_out_lines() -> None:
    app, _ = make_app(8, {1, 3, 5, 7})

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f", "2")
            assert app.phrase_indices == [0, 2, 4, 6]

            await pilot.press("m")
            app.screen.query_one("#manual-selection-input", Input).value = "2-7"
            await pilot.press("enter")

            assert app.selected_indices == {1, 2, 3}
            assert app.selected_index == 3
            assert app.selection_anchor_index == 3
            assert app.query_one("#line-list", NonWrappingOptionList).highlighted == 3

    run(exercise())


def test_filter_dialog_escape_cancels_without_changing_display() -> None:
    app, _ = make_app(3, {1})

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("f", "escape")
            assert not isinstance(app.screen, FilterDialog)
            assert app.filter_type == FilterType.ALL
            assert app.phrase_indices == [0, 1, 2]

    run(exercise())


def test_number_selection_applies_each_filter_and_updates_pinned_status() -> None:
    app, project = make_app(6, {1, 2, 3, 4})
    project.phrase_groups = [
        make_phrase_group(f"Line {index + 1}") for index in range(6)
    ]
    project.sound_segments.sound_segments_map[1][0].num_errors = 2
    project.sound_segments.sound_segments_map[2][0].num_errors = 0
    project.sound_segments.sound_segments_map[3][0].num_errors = -1
    project.sound_segments.sound_segments_map[4][0].num_errors = 1
    project.sound_segments.failed_segment_files.add("segment-4.flac")

    expected_filters = [
        (
            "2",
            FilterType.UNGENERATED,
            [0, 5],
            "Showing ungenerated lines (2)",
        ),
        (
            "3",
            FilterType.GENERATED,
            [1, 2, 3, 4],
            "Showing generated lines (4)",
        ),
        (
            "4",
            FilterType.GENERATED_WITH_ERRORS,
            [1, 4],
            "Showing lines with word errors (2)",
        ),
        ("5", FilterType.FAILED, [4], "Showing failed lines (1)"),
        ("1", FilterType.ALL, [0, 1, 2, 3, 4, 5], ""),
    ]

    async def exercise() -> None:
        with patch.object(generate_editor_module.L, "d"):
            async with app.run_test() as pilot:
                filter_header = app.query_one("#header-line-3", Static)
                status_left = app.query_one("#status-left", Static)
                status_right = app.query_one("#status-right", Static)
                assert str(filter_header.render()).endswith("[F] Filter lines")
                assert str(status_left.render()) == ""

                for key, filter_type, phrase_indices, expected_status in expected_filters:
                    await pilot.press("f", key)
                    assert not isinstance(app.screen, FilterDialog)
                    assert app.filter_type == filter_type
                    assert app.phrase_indices == phrase_indices
                    assert str(filter_header.render()).endswith("[F] Filter lines")
                    assert str(status_left.render()) == expected_status
                    assert str(status_right.render()) == "0 lines queued for generation"

    run(exercise())


def test_filter_reuses_retained_options_without_reformatting(monkeypatch) -> None:
    app, _ = make_app(6, {1, 2, 3, 4})

    async def exercise() -> None:
        async with app.run_test() as pilot:
            option_list = app.query_one("#line-list", NonWrappingOptionList)
            retained_options = {
                phrase_index: option_list.get_option(
                    f"generate-phrase-{phrase_index}"
                )
                for phrase_index in (1, 2, 3, 4)
            }
            formatted_indices: list[int] = []
            original_format_line = app.format_line

            def record_format_line(index: int):
                formatted_indices.append(index)
                return original_format_line(index)

            monkeypatch.setattr(app, "format_line", record_format_line)
            await pilot.press("f", "3")
            await pilot.pause()

            assert formatted_indices == []
            assert app.filter_type == FilterType.GENERATED
            for phrase_index, retained_option in retained_options.items():
                assert (
                    option_list.get_option(f"generate-phrase-{phrase_index}")
                    is retained_option
                )

    run(exercise())


def test_p_plays_highlighted_best_sound_segment() -> None:
    app, project = make_app(3)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("worse.flac", num_errors=2),
        StubSoundSegment("best.flac", num_errors=0),
    ]

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil, "current_sound_id", return_value="second-sound-id"
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("second-sound-id", ""),
            ) as play_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "p")
                play_sound.assert_called_once_with("/project/segments/best.flac")
                assert app.playing_sound_id == "second-sound-id"
                assert app.playing_sound_path == "/project/segments/best.flac"
                assert app.playing_phrase_index == 1
                status_line = app.query_one("#status-right", Static)
                assert str(status_line.render()) == "Playing line 2"

    run(exercise())


def test_p_on_current_sound_stops_it_instead_of_restarting() -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        with (
            patch.object(
                PlaySoundUtil,
                "current_sound_id",
                return_value="sound-id",
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ) as play_sound,
            patch.object(PlaySoundUtil, "stop_sound_async") as stop_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("p")
                await pilot.press("p")
                play_sound.assert_called_once_with("/project/segments/segment-0.flac")
                stop_sound.assert_called_once_with()
                assert app.playing_sound_id == ""
                assert app.playing_sound_path == ""
                assert app.playing_phrase_index is None
                status_line = app.query_one("#status-right", Static)
                assert str(status_line.render()) == "0 lines queued for generation"

    run(exercise())


def test_p_on_different_sound_starts_new_playback() -> None:
    app, _ = make_app(2)
    current_sound_id = "first-sound-id"

    async def exercise() -> None:
        def get_current_sound_id() -> str:
            return current_sound_id

        def play_sound(path: str) -> tuple[str, str]:
            nonlocal current_sound_id
            current_sound_id = (
                "second-sound-id" if path.endswith("segment-1.flac") else "first-sound-id"
            )
            return current_sound_id, ""

        with (
            patch.object(
                PlaySoundUtil,
                "current_sound_id",
                side_effect=get_current_sound_id,
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                side_effect=play_sound,
            ) as play_sound,
            patch.object(PlaySoundUtil, "stop_sound_async"),
        ):
            async with app.run_test() as pilot:
                await pilot.press("p")
                await pilot.press("down", "p")
                assert play_sound.call_args_list == [
                    (("/project/segments/segment-0.flac",), {}),
                    (("/project/segments/segment-1.flac",), {}),
                ]
                assert app.playing_sound_id == "second-sound-id"
                assert app.playing_sound_path == "/project/segments/segment-1.flac"
                status_line = app.query_one("#status-right", Static)
                assert str(status_line.render()) == "Playing line 2"

    run(exercise())


def test_i_shows_ansi_formatted_segment_info_for_selected_phrase(
    tmp_path: Path,
) -> None:
    app, project = make_app(2)
    project.sound_segments_path = str(tmp_path)
    project.sound_segments.sound_segments_map[1] = [
        StubSoundSegment("worse.flac", num_errors=2),
        StubSoundSegment("best.flac", num_errors=0),
    ]
    filename = make_terminal_hyperlink(
        str(tmp_path / "best.flac"), "best.flac", is_file=True
    )
    info_lines = [
        "\033[31mLine: 2, word errors detected: 0\033[0m",
        f"Filename: {filename}",
        "\033[2;3mSource text: The selected line\033[0m",
    ]

    async def exercise() -> None:
        with (
            patch.object(
                SegmentTranscriptUtil,
                "make_info_text_lines",
                return_value=info_lines,
            ) as make_info_text_lines,
            patch.object(
                AudioMetaUtil, "get_audio_duration", return_value=10.84
            ) as get_audio_duration,
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "i")
                assert isinstance(app.screen, SegmentInfoDialog)
                make_info_text_lines.assert_called_once_with(
                    1, project, is_for_dialog=True
                )
                get_audio_duration.assert_called_once_with(str(tmp_path / "best.flac"))

                rendered_info = app.screen.query_one(
                    "#segment-info-content", Static
                ).render()
                assert isinstance(rendered_info, Content)
                assert str(rendered_info) == (
                    "Line: 2, word errors detected: 0\n"
                    "Duration: 10.8s\n"
                    "\n"
                    "Filename: best.flac\n"
                    "Source text: The selected line"
                )
                styles = [span.style for span in app.screen.info_text.spans]
                assert any(getattr(style, "color", None) is not None for style in styles)
                assert any(getattr(style, "italic", None) for style in styles)
                assert any(
                    getattr(style, "link", None) == f"file://{tmp_path / 'best.flac'}"
                    for style in styles
                )

    run(exercise())


def test_p_toggles_displayed_segment_playback_without_closing_info_dialog() -> None:
    app, _ = make_app(2)

    async def exercise() -> None:
        with (
            patch.object(
                SegmentTranscriptUtil,
                "make_info_text_lines",
                return_value=["Segment info"],
            ),
            patch.object(
                PlaySoundUtil,
                "current_sound_id",
                return_value="sound-id",
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ) as play_sound,
            patch.object(PlaySoundUtil, "stop_sound_async") as stop_sound,
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "i", "p")
                assert isinstance(app.screen, SegmentInfoDialog)
                play_sound.assert_called_once_with("/project/segments/segment-1.flac")
                assert app.playing_phrase_index == 1

                await pilot.press("p")
                assert isinstance(app.screen, SegmentInfoDialog)
                stop_sound.assert_called_once_with()
                assert app.playing_phrase_index is None

    run(exercise())


def test_x_in_info_dialog_closes_it_and_confirms_only_displayed_segment() -> None:
    app, project = make_app(2)

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil,
            "make_info_text_lines",
            return_value=["Segment info"],
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "i", "x")
                assert isinstance(app.screen, SaveChangesDialog)
                assert app.screen.copy_lines == ["Delete 1 generated sound segment?"]

                await pilot.press("y")
                await pilot.pause()
                assert project.sound_segments.deleted_index_batches == [{1}]
                assert set(project.sound_segments.sound_segments_map) == {0}

    run(exercise())


def test_q_in_info_dialog_closes_it_and_quick_generates_displayed_segment() -> None:
    app, project = make_app(2)
    submitted, patches = patch_quick_job(project, status=GenerationTerminalStatus.FAILED)

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil,
            "make_info_text_lines",
            return_value=["Segment info"],
        ):
            async with app.run_test() as pilot:
                await pilot.press("down", "i", "q")
                await pilot.pause(0.3)
                assert isinstance(app.screen, QuickGenModal)
                assert app.is_running is True

    run_with_patches(patches, exercise())
    assert submitted.indices == [1]


def test_info_dialog_omits_duration_when_audio_load_fails() -> None:
    app, project = make_app(1)
    info_lines = ["\033[31mCould not load segment STT info: invalid JSON\033[0m"]

    async def exercise() -> None:
        with (
            patch.object(
                SegmentTranscriptUtil,
                "make_info_text_lines",
                return_value=info_lines,
            ),
            patch.object(
                AudioMetaUtil,
                "get_audio_duration",
                side_effect=OSError("couldn't load audio"),
            ),
        ):
            async with app.run_test() as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                assert str(
                    app.screen.query_one("#segment-info-content", Static).render()
                ) == "Could not load segment STT info: invalid JSON"

    run(exercise())


def test_info_dialog_is_height_limited_scrollable_and_escape_dismisses(
) -> None:
    app, _ = make_app(1)
    info_lines = [f"metadata {index}" for index in range(100)]

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil, "make_info_text_lines", return_value=info_lines
        ):
            async with app.run_test(size=(80, 18)) as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                scroll = app.screen.query_one("#segment-info-scroll", VerticalScroll)
                assert scroll.virtual_size.height > scroll.region.height

                await pilot.press("escape")
                assert not isinstance(app.screen, SegmentInfoDialog)

    run(exercise())


def test_info_dialog_inside_click_stays_open_and_outside_click_dismisses(
) -> None:
    app, _ = make_app(1)

    async def exercise() -> None:
        with patch.object(
            SegmentTranscriptUtil,
            "make_info_text_lines",
            return_value=["Segment info"],
        ):
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("i")
                assert isinstance(app.screen, SegmentInfoDialog)
                dialog = app.screen.query_one("#segment-info-dialog", Vertical)

                await pilot.click(dialog)
                assert isinstance(app.screen, SegmentInfoDialog)

                await pilot.click(offset=(0, 0))
                assert not isinstance(app.screen, SegmentInfoDialog)

    run(exercise())


def test_playback_status_overrides_selection_status_and_restores_it_when_cleared(
) -> None:
    app, _ = make_app(4)

    async def exercise() -> None:
        with (
            patch.object(PlaySoundUtil, "current_sound_id", return_value="sound-id"),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ),
            patch.object(PlaySoundUtil, "stop_sound_async"),
        ):
            async with app.run_test() as pilot:
                await pilot.press("p", "shift+down", "shift+down")
                status_bar = app.query_one("#status-bar", Horizontal)
                status_line = app.query_one("#status-right", Static)
                assert str(status_line.render()) == "Playing line 1"
                assert status_bar.display is True

                app.clear_playback_status()
                assert str(status_line.render()) == "3 lines selected"

    run(exercise())


def test_playback_status_clears_dynamically_when_sound_finishes() -> None:
    app, _ = make_app(1)
    current_sound_id = "sound-id"

    async def exercise() -> None:
        def get_current_sound_id() -> str:
            return current_sound_id

        with (
            patch.object(
                PlaySoundUtil, "current_sound_id", side_effect=get_current_sound_id
            ),
            patch.object(
                PlaySoundUtil,
                "play_sound_file_async",
                return_value=("sound-id", ""),
            ),
        ):
            async with app.run_test() as pilot:
                nonlocal current_sound_id
                await pilot.press("p")
                assert str(app.query_one("#status-right", Static).render()) == (
                    "Playing line 1"
                )

                current_sound_id = ""
                await pilot.pause(0.2)
                assert str(app.query_one("#status-right", Static).render()) == (
                    "0 lines queued for generation"
                )
                assert app.playing_phrase_index is None

    run(exercise())


@pytest.mark.parametrize(
    ("line_count", "generated_indices", "range_string", "presses", "expected_range"),
    [
        pytest.param(2, set(range(2)), "none", ("escape",), "none", id="none-range"),
        pytest.param(3, set(), "2", ("escape",), "2", id="loaded-range"),
        pytest.param(
            3,
            set(),
            "none",
            ("ctrl+a", "space", "escape"),
            "all",
            id="queue-all-lines-then-exit",
        ),
        pytest.param(
            4,
            {0, 1},
            "1-2",
            ("escape",),
            "1-2",
            id="range-over-generated-lines",
        ),
    ],
)
def test_exit_persists_staged_generation_range_and_closes(
    line_count: int,
    generated_indices: set[int],
    range_string: str,
    presses: tuple[str, ...],
    expected_range: str,
) -> None:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(line_count)],
        StubSoundSegments(
            {
                index: [StubSoundSegment(f"segment-{index}.flac")]
                for index in generated_indices
            }
        ),
        generate_range_string=range_string,
    )
    app = GenerateEditor(make_state(project))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.phrase_indices == list(range(line_count))
            await pilot.press(*presses)
            await pilot.pause()
            assert app.is_running is False

    run(exercise())
    assert project.generate_range_string == expected_range
    assert project.save_calls == [expected_range]
    assert app.return_value == EditorClosed()


@pytest.mark.parametrize(
    ("line_count", "generated_indices", "presses", "expected_range"),
    [
        pytest.param(
            7,
            {1, 4, 6},
            ("shift+down", "space", "escape"),
            "1",
            id="queue-single-line",
        ),
        pytest.param(
            4,
            set(),
            ("space", "down", "down", "space", "escape"),
            "1, 3",
            id="queue-disjoint-lines",
        ),
    ],
)
def test_exit_persists_toggled_queue_without_deleting_sound_segments(
    line_count: int,
    generated_indices: set[int],
    presses: tuple[str, ...],
    expected_range: str,
) -> None:
    project = StubProject(
        [StubPhraseGroup(f"Line {index + 1}") for index in range(line_count)],
        StubSoundSegments(
            {
                index: [StubSoundSegment(f"segment-{index}.flac")]
                for index in generated_indices
            }
        ),
    )
    original_map = {
        index: list(items)
        for index, items in project.sound_segments.sound_segments_map.items()
    }
    app = GenerateEditor(make_state(project))

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press(*presses)
            await pilot.pause()
            assert app.is_running is False

    run(exercise())
    assert project.generate_range_string == expected_range
    assert project.save_calls == [expected_range]
    assert project.sound_segments.sound_segments_map == original_map
    assert app.return_value == EditorClosed()


def test_clean_exit_returns_save_failure() -> None:
    app, project = make_app(2, set())
    app.staged_queued_indices.add(0)

    project.save_error = "disk full"
    assert app.persist_staged_queue() == "Save failed: disk full"

    project.save_error = ""
    assert app.persist_staged_queue() == ""
    assert project.save_calls == ["1", "1"]

    project.save_error = "disk full"

    async def exercise() -> None:
        async with app.run_test() as pilot:
            await pilot.press("escape")
            await pilot.pause()

        assert app.return_value == EditorSaveFailed("Save failed: disk full")

    run(exercise())
    assert project.save_calls == ["1", "1", "1"]
