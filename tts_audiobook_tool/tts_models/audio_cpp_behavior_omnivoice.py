"""OmniVoice behavior: translates the project code into OmniVoice's language ids.

The audio.cpp OmniVoice session accepts only an exact OmniVoice id or English
language name and rejects anything else (eg "en-US") with a generation error,
so the catalog hands over the raw project code (`language_policy =
"project_code"`) and this behavior resolves it with the same function the
local model uses (see `omnivoice_language`). A code with no OmniVoice id sends
no language (automatic selection) instead of failing.
"""
from __future__ import annotations

from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior
from tts_audiobook_tool.tts_models.language_resolution import LanguageResolution
from tts_audiobook_tool.tts_models.omnivoice_language import resolve_omnivoice_language


class OmniVoiceBehavior(AudioCppModelBehavior):

    def resolve_language(self, language: str) -> LanguageResolution:
        return resolve_omnivoice_language(language)
