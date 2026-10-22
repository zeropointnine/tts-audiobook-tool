from dataclasses import dataclass

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.local_definition import require_local_definition


@dataclass(frozen=True)
class BreezeSampling:
    """Resolved sampling values. Temperature/top-k/top-p apply to both the backbone and the depth decoder (as upstream)."""
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    # Used only when the voice has instructions (no CFG negative prompt exists for pure cloning / plain TTS)
    cfg: float


class BreezeBaseModel(TtsBaseModel):
    """
    Breeze TTS 2, local inference (packaging fork `breeze_tts`).
    Must not import `breeze_tts` (imported by the main process).
    """

    INFO = TtsModelType.require_by_id("breeze_tts_2_local").value

    # Reference codes are tiny CPU tensors; keep one per voice sample
    RETAINS_MULTIPLE_VOICE_CLONES = True

    HF_REPO_ID = "BreezeBlue/Breeze-TTS-2"
    HF_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"

    # Sampling defaults and bounds (`[models.local.parameters]`); see `resolve_sampling()`
    LOCAL = require_local_definition("breeze_tts_2_local")

    # Frames at 12.5/s (~120s); the generation-config default is 750
    MAX_NEW_TOKENS = 1500
    # Trained context (`max_position_embeddings`): reference transcript + reference frames + text + generated frames.
    # Not enforced by eager `generate()` (only `max_new_tokens` is), so exceeding it only warrants a warning.
    MAX_POSITIONS = 2048
    FRAME_RATE = 12.5
    SPEAKER = "S0"

    @staticmethod
    def resolve_sampling(project: Project) -> BreezeSampling:
        local = BreezeBaseModel.LOCAL
        return BreezeSampling(
            temperature=local.resolve_float(project, "temperature"),
            top_k=local.resolve_int(project, "top_k"),
            top_p=local.resolve_float(project, "top_p"),
            repetition_penalty=local.resolve_float(project, "repetition_penalty"),
            cfg=local.resolve_float(project, "cfg"),
        )
