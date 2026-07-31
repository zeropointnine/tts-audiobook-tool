from typing import ClassVar

from textual.binding import Binding, BindingType
from textual.css.errors import StylesheetError

from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.textual.content_textual_app import ContentTextualApp
from tts_audiobook_tool.textual.textual_shared import (
    HangingIndentText,
    STYLE_DIM,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import make_error_string, print_feedback


class VoiceLineEditorTextualApp(ContentTextualApp):
    BINDINGS: ClassVar[list[BindingType]] = [
        *ContentTextualApp.BINDINGS,
        *(
            Binding(str(number), f"assign_voice({number - 1})", show=False)
            for number in range(1, 10)
        ),
    ]

    def __init__(
        self,
        project: Project,
        voice_sample_count: int | None = None,
    ) -> None:
        self.original_voice_indices = [
            phrase_group.voice_index for phrase_group in project.phrase_groups
        ]
        self.staged_voice_indices = list(self.original_voice_indices)
        self.did_save_changes = False
        self.save_error = ""
        if voice_sample_count is None:
            voice_sample_count = len(
                ProjectVoiceUtil.get_voice_values(project, Tts.get_type())
            )
        highest_voice_key = min(max(voice_sample_count, 1), 9)
        header_lines = [
            f"{COL_ACCENT}Edit voice selections",
            f"{COL_DIM}- Navigation keys: [UP], [DOWN], [PAGE UP/DOWN], [HOME/END]",
            f"{COL_DIM}- Select multiple lines by holding [SHIFT] + navigation keys",
            f"{COL_DIM}- Use number keys [{COL_ACCENT}1{COL_DIM}] to [{COL_ACCENT}{highest_voice_key}{COL_DIM}] to set voice sample for selected text line/s",
            f"{COL_DIM}- Press [ESC] to finish   - Press [CTRL-F] to find text",
        ]
        super().__init__(project, header_lines)

    @property
    def has_changes(self) -> bool:
        return self.staged_voice_indices != self.original_voice_indices

    def format_line(self, index: int) -> HangingIndentText:
        """Format one row, styling selected rows except for the active row."""
        phrase_index = self.phrase_indices[index]
        phrase_group = self.project.phrase_groups[phrase_index]
        voice_index = self.staged_voice_indices[phrase_index]
        voice_number = max(voice_index + 1, 1)
        voice_values = ProjectVoiceUtil.get_voice_values(self.project, Tts.get_type())
        # Keep showing the stored number, but flag stale selections after voices change.
        voice_status = " *OUT OF RANGE*" if voice_index >= len(voice_values) else ""
        prefix = (
            f"{phrase_index + 1:05d} [Voice sample {voice_number}{voice_status}] "
        )
        is_find_match = index == self.find_match_index
        style = f"{STYLE_DIM} reverse" if is_find_match else ""
        return HangingIndentText.from_ansi(
            prefix + phrase_group.presentable_text,
            content_start=len(prefix),
            max_lines=3,
            style=style,
        )

    def action_assign_voice(self, voice_index: int) -> None:
        """Assign an available voice sample to all selected phrase groups."""
        voice_values = ProjectVoiceUtil.get_voice_values(self.project, Tts.get_type())
        if voice_index >= len(voice_values):
            return

        def assign_voice(_visible_index: int, phrase_index: int) -> bool:
            if self.staged_voice_indices[phrase_index] == voice_index:
                return False
            self.staged_voice_indices[phrase_index] = voice_index
            return True

        self.mutate_selected_items(assign_voice)

    def commit_changes_and_exit(self) -> None:
        """Apply staged values and persist them, rolling memory back on failure."""
        phrase_groups = self.project.phrase_groups
        if len(phrase_groups) != len(self.staged_voice_indices):
            self.save_error = "Save failed: project text changed while editing"
            self.exit()
            return

        for phrase_group, voice_index in zip(
            phrase_groups, self.staged_voice_indices, strict=True
        ):
            phrase_group.voice_index = voice_index

        try:
            error = ProjectTextIOUtil.save_phrase_groups(self.project)
        except Exception as exception:
            error = make_error_string(exception)
        if error:
            for phrase_group, voice_index in zip(
                phrase_groups, self.original_voice_indices, strict=True
            ):
                phrase_group.voice_index = voice_index
            self.save_error = f"Save failed: {error}"
        else:
            self.did_save_changes = True
        self.exit()

    def save_changes_and_exit(self) -> None:
        """Backward-compatible name for committing the staged voice values."""
        self.commit_changes_and_exit()

    @classmethod
    def start(cls, project: Project) -> None:
        """Run an editor for a project and report its save result."""
        if not cls.check_terminal_support():
            return
        app = cls(project)
        app.run(inline=False)
        if isinstance(app._exception, StylesheetError):
            print_feedback("Couldn't load textual css", is_error=True)
        elif app.save_error:
            print_feedback(app.save_error, is_error=True)
        elif app.did_save_changes:
            print_feedback("Saved changes", long_pause=True)
