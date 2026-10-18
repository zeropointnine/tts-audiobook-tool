"""CosyVoice3 template-dependent transcript and per-voice instruction use."""
from __future__ import annotations

from typing import ClassVar

from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

ZERO_SHOT = "zero_shot"
INSTRUCT = "instruct"


class CosyVoice3Behavior(AudioCppModelBehavior):

    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {"template_name": "choice"}

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        return values["template_name"] == ZERO_SHOT

    def uses_voice_instructions(self, values: ParameterValues) -> bool:
        return self.definition.voice_instructions and values["template_name"] == INSTRUCT

    def expects_voice_instructions(self, values: ParameterValues) -> bool:
        # Instruct with a blank instruction conditions on an empty style prompt.
        return self.uses_voice_instructions(values)
