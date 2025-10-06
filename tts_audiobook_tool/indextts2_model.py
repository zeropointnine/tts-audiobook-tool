import os
import huggingface_hub
from indextts.infer_v2 import IndexTTS2 # type: ignore
from numpy import ndarray
import numpy

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_model import IndexTts2ModelProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *


class IndexTts2Model(IndexTts2ModelProtocol):
    """
    """

    def __init__(self, use_fp16: bool):

        super().__init__(info=TtsModelInfos.INDEXTTS2.value)

        # Download model to default cache location
        model_dir = huggingface_hub.snapshot_download(
            repo_id=REPO_ID,
            cache_dir=huggingface_hub.constants.HF_HUB_CACHE,
            local_files_only=False
        )
        cfg_path = os.path.join(model_dir, "config.yaml")

        # FYI, "use_cuda_kernel=True" made zero speed difference for me on two different systems,
        # so not bothering to make it a configurable option

        # FYI, "use_deepspeed=True" also had no positive impact
        # (native Windows, 3080Ti, deepspeed wheel, though using fp16 fwiw)

        self.model: IndexTTS2 | None = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            device=None, # let model auto-detect device
            use_cuda_kernel=False,
            use_deepspeed=False
        )


    def kill(self) -> None:
        self.model = None


    def generate(
            self,
            text: str,
            voice_path: str,
            temperature: float,
            emo_voice_path: str,
            emo_voice_alpha: float
    ) -> Sound | str:
        """
        Returns generated audio or error string
        """

        if not self.model:
            return "Model is not initialized"

        if temperature == -1:
            temperature = IndexTts2Model.DEFAULT_TEMPERATURE
        if emo_voice_alpha == -1:
            emo_voice_alpha = IndexTts2Model.DEFAULT_EMO_VOICE_ALPHA

        try:
            # FYI, infer() caches loaded voice sample/s internally
            result = self.model.infer(
                spk_audio_prompt=voice_path,
                text=text,
                temperature=temperature,
                emo_audio_prompt=emo_voice_path or None,
                emo_alpha=emo_voice_alpha,
                max_text_tokens_per_segment=MAX_TOKENS_PER_SEGMENT,
                output_path=None,
                verbose=False
            )
        except Exception as e:
            return make_error_string(e)

        if not result:
            return "Model did not return any data"
        if len(result) != 2:
            return f"Model returned unexpected data format: {type(result)}"

        sample_rate = result[0]
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            return f"Model returned unexpected samplerate value: {sample_rate}"

        data: ndarray = result[1] # looks like: array([[-1], [-2], [-1], etc], dtype=int16)
        data = data.flatten()
        if len(data) == 0:
            return f"Model returned empty array"
        # Convert from int16's to floats
        data = data.astype(numpy.float32) / 32768.0

        return Sound(data, sample_rate)


REPO_ID = "IndexTeam/IndexTTS-2"

# Upper bound of recommended range as per project gradio demo
# (App ofc always segments the text well within this range but yea)
MAX_TOKENS_PER_SEGMENT = 200
