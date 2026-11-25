"""ZONOS3-Mini: per-voice speaking rate, and speaker guidance only with a sample.

The server takes the three condition rows (loudness, fmax, speaking_rate) as one
`condition_buckets` string of exactly three bucket IDs. An ID equal to its
table's bucket count selects that table's "unknown" entry, which is also what
the server uses when the field is omitted, so a speaking-rate override leaves
the other two rows exactly as they would be without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tts_audiobook_tool.app_types.voice import SPEAKING_RATE_BUCKET_MAX
from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior, ParameterValues

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types.voice import Voice

# The checkpoint's loudness and fmax bucket counts, ie their "unknown" IDs
# (backbone-ptq.gguf `zonos3.condition.bucket_counts` = [10, 10, 30]). The
# server does not report them.
LOUDNESS_UNKNOWN_BUCKET = 10
FMAX_UNKNOWN_BUCKET = 10


class Zonos3Behavior(AudioCppModelBehavior):
    REQUIRED_PARAMETERS: ClassVar[dict[str, str]] = {
        "speaker_mode": "choice",
        "guidance_scale": "number",
    }
    # Matches the checkpoint's 30 speaking-rate buckets.
    speaking_rate_bucket_count = SPEAKING_RATE_BUCKET_MAX

    def get_voice_request_options(self, voice: Voice | None) -> dict[str, int | float | str]:
        bucket = voice.speaking_rate_buckets.get(self.definition.spec.id) if voice is not None else None
        if bucket is None:
            # Omission selects the server's unknown embedding, not a numeric rate.
            return {}
        # The UI/storage are 1-based; the server's known buckets are 0..29.
        return {"condition_buckets": f"{LOUDNESS_UNKNOWN_BUCKET},{FMAX_UNKNOWN_BUCKET},{bucket - 1}"}

    def adjust_payload(self, payload: dict, values: ParameterValues) -> None:
        # Speaker CFG mixes in a twin pass without the voice; without a voice
        # there is nothing to leave out, and the server rejects any value but 1.
        if "voice_ref" not in payload:
            parameter = self.definition.parameters["guidance_scale"]
            target = payload.get("options", {}) if parameter.target == "options" else payload
            target.pop(parameter.request_key, None)
