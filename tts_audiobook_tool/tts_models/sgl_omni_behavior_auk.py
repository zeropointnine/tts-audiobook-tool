"""AuK / AuK-Flash (SGL-Omni): the voice's sample decides the speech route.

The server builds AuK's instruction from which request fields are present:
with reference audio it is a same-voice clone and any `instructions` are
ignored outright (verified: byte-identical output with and without them);
without reference audio, `instructions` is the voice description (blank: the
server's default "A clear, natural voice."). There is no app-side Mode: each
voice picks its route by whether it has a sample, so one project can mix
cloned and described voices. Instructions are therefore sent only for
sample-less voices. Sample-less requests also need an explicit duration,
which the adapter's `auk_seconds` prompt policy supplies (see auk_duration).
"""
from __future__ import annotations

from tts_audiobook_tool.tts_models.sgl_omni_behavior import SglOmniModelBehavior


class AukSglOmniBehavior(SglOmniModelBehavior):

    def uses_instructions_with_sample(self) -> bool:
        return False
