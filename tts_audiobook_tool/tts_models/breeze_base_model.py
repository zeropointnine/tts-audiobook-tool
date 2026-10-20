from dataclasses import dataclass

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


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

    # Upstream defaults (`update_generation_config_for_breeze`)
    TEMPERATURE_DEFAULT = 0.9
    TEMPERATURE_MIN = 0.01
    TEMPERATURE_MAX = 2.0
    TOP_K_DEFAULT = 50
    TOP_P_DEFAULT = 1.0
    # As upstream `api.py` / `infer.py`
    REPETITION_PENALTY_DEFAULT = 1.1
    # Upstream recommends 4 with instructions (voice design / voice direction)
    CFG_DEFAULT = 4.0
    CFG_MIN = 1.0
    CFG_MAX = 10.0

    # Frames at 12.5/s (~120s); the generation-config default is 750
    MAX_NEW_TOKENS = 1500
    # Trained context (`max_position_embeddings`): reference transcript + reference frames + text + generated frames.
    # Not enforced by eager `generate()` (only `max_new_tokens` is), so exceeding it only warrants a warning.
    MAX_POSITIONS = 2048
    FRAME_RATE = 12.5
    SPEAKER = "S0"

    @staticmethod
    def resolve_sampling(project: Project) -> BreezeSampling:
        def value(name: str, default):
            setting = project.get_model_setting("breeze_tts_2_local", name)
            return default if setting == -1 else setting

        return BreezeSampling(
            temperature=float(value("temperature", BreezeBaseModel.TEMPERATURE_DEFAULT)),
            top_k=int(value("top_k", BreezeBaseModel.TOP_K_DEFAULT)),
            top_p=float(value("top_p", BreezeBaseModel.TOP_P_DEFAULT)),
            repetition_penalty=float(value("repetition_penalty", BreezeBaseModel.REPETITION_PENALTY_DEFAULT)),
            cfg=float(value("cfg", BreezeBaseModel.CFG_DEFAULT)),
        )
