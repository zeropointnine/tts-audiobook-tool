"""CosyVoice3 (audio.cpp): Mode-dependent request and menu behavior.

audio.cpp's CosyVoice3 session selects a prompt template from
``options.template_name``. Only ``zero_shot`` reads ``reference_text``, and
only ``instruct`` reads ``options.instruction``; values the selected template
ignores are dropped so the printed request shows only what the model reads.
The Instructions control likewise appears only in Instruct mode, where it is
required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tts_audiobook_tool.app_types import ReadinessIssue
from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppMenuControl

ZERO_SHOT = "zero_shot"
INSTRUCT = "instruct"


class CosyVoice3Behavior(AudioCppModelBehavior):

    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {
        "template_name": "choice",
        "instruction": "text",
    }

    @staticmethod
    def _is_instruct(values: ParameterValues) -> bool:
        return values["template_name"] == INSTRUCT

    @staticmethod
    def _has_instruction(values: ParameterValues) -> bool:
        return bool(str(values["instruction"]).strip())

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        return values["template_name"] == ZERO_SHOT

    def get_ignored_parameters(self, values: ParameterValues) -> frozenset[str]:
        return frozenset() if self._is_instruct(values) else frozenset({"instruction"})

    def get_blocking_issues(self, project: Project, values: ParameterValues) -> list[ReadinessIssue]:
        # audio.cpp would accept the request but fall back to a generic prompt,
        # silently producing cross-lingual output instead of what was asked for.
        if self._is_instruct(values) and not self._has_instruction(values):
            return [ReadinessIssue("instructions", "Instructions are required when Mode is Instruct")]
        return []

    def is_control_visible(self, control: AudioCppMenuControl, values: ParameterValues) -> bool:
        if control.parameter == "instruction":
            return self._is_instruct(values)
        return True

    def get_required_label(self, name: str, values: ParameterValues) -> str | None:
        if name == "instruction" and self._is_instruct(values):
            return "required for Instruct"
        return None
