"""CosyVoice3 (SGL-Omni) Mode-dependent transcript and per-voice instruction use.

The server picks its prompt mode from which fields are present (`instructions`
-> instruct2, `ref_text` -> zero-shot, neither -> cross-lingual) and rejects a
request carrying both. The app-side Mode makes that choice explicit, matching
the audio.cpp entry: only Zero-shot sends the transcript, only Instruct sends
the instructions, so the two are never combined.
"""
from __future__ import annotations

from typing import ClassVar

from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import INSTRUCT, ZERO_SHOT
from tts_audiobook_tool.tts_models.sgl_omni_behavior import ParameterValues, SglOmniModelBehavior


class CosyVoice3SglOmniBehavior(SglOmniModelBehavior):

    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {"template_name": "choice"}

    def uses_reference_transcript(self, values: ParameterValues) -> bool:
        return values["template_name"] == ZERO_SHOT

    def uses_voice_instructions(self, values: ParameterValues) -> bool:
        return self.definition.voice_instructions and values["template_name"] == INSTRUCT

    def expects_voice_instructions(self, values: ParameterValues) -> bool:
        # Blank instructions make the server fall back to cross-lingual.
        return self.uses_voice_instructions(values)
