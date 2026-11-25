"""ZONOS3-Mini's optional per-voice speaking-rate conditioning."""
from __future__ import annotations

from tts_audiobook_tool.app_types.voice import SPEAKING_RATE_BUCKET_MAX, Voice
from tts_audiobook_tool.tts_models.sgl_omni_behavior import SglOmniModelBehavior


class Zonos3SglOmniBehavior(SglOmniModelBehavior):
    speaking_rate_bucket_count = SPEAKING_RATE_BUCKET_MAX

    def get_voice_condition_buckets(self, voice: Voice | None) -> dict[str, int]:
        bucket = voice.speaking_rate_buckets.get(self.definition.spec.id) if voice is not None else None
        if bucket is None:
            # Omission selects the server's unknown embedding, not a numeric rate.
            return {}
        # The UI/storage are 1-based; SGL-Omni's known buckets are 0..29.
        return {"speaking_rate": bucket - 1}
