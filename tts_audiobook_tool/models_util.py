import torch

from tts_audiobook_tool.app_types import ModelWarmUpResult
from tts_audiobook_tool.memory_util import MemoryUtil
from tts_audiobook_tool.music_detector import MusicDetector
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sidon_util import SidonUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class ModelsUtil:
    """
    Multiple-model management util functions
    Including YamnetDetector, and Sidonmodel (holds statics instance)
    """

    sidon_upsampler: SidonUtil | None = None


    @staticmethod
    def warm_up_models(state: State, skip_yamnet: bool=False) -> ModelWarmUpResult:
        """
        Instantiates required models for main inference flow, prints updates.

        Params:
            skip_yamnet - if True, does not init YAMNet model even if state would dictate otherwise

        Returns:
            ModelWarmUpResult indicating success, interruption, or initialization failure.
        """

        should_tts = not Tts.instance_exists()        
        should_stt = not Stt.should_skip(state) and not Stt.has_instance()
        should_yamnet = Tts.get_type().value.hallucinates_music and not MusicDetector.has_instance() and not skip_yamnet
        
        shoulds = [should_tts, should_stt, should_yamnet]
        num_shoulds = sum(1 for item in shoulds if item)
        if num_shoulds == 0:
            return ModelWarmUpResult()

        if not should_stt:
            # "Lazy unload", relevant for user flows like: 
            # Do inference, choose unsupported validation language, do inference
            Stt.clear_stt_model()

        if num_shoulds >= 2:
            print_init("Warming up models...")

        SigIntHandler().set("model init")

        # Init TTS
        if should_tts:
            try:
                _ = Tts.get_instance()
            except Exception as e:
                err_msg = str(e)
                SigIntHandler().clear()
                return ModelWarmUpResult(error=err_msg)

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return ModelWarmUpResult(did_interrupt=True)

        # Init STT
        if should_stt:
            try:
                _ = Stt.get_whisper()
            except Exception as e:
                err_msg = str(e)
                SigIntHandler().clear()
                return ModelWarmUpResult(error=err_msg)

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return ModelWarmUpResult(did_interrupt=True)
        
        # Init YAMNet
        if should_yamnet:
            try:
                _ = MusicDetector.get_model()
            except Exception as e:
                err_msg = str(e)
                SigIntHandler().clear()
                return ModelWarmUpResult(error=err_msg)

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return ModelWarmUpResult(did_interrupt=True)

        SigIntHandler().clear()
        return ModelWarmUpResult()

    @staticmethod
    def clear_all_models(except_sidon: bool = False) -> None:

        Stt.clear_stt_model()
        Tts.clear_tts_model()
        MusicDetector.clear_model()
        if not except_sidon:
            ModelsUtil.clear_sidon_upsampler()

        # For good measure
        MemoryUtil.gc_ram_vram()

    @staticmethod
    def is_any_model_loaded() -> bool:
        return Stt.has_instance() or \
            Tts.instance_exists() or \
            MusicDetector.has_instance() or \
            ModelsUtil.sidon_upsampler is not None

    @staticmethod
    def get_sidon_upsampler() -> SidonUtil | None:
        if not torch.cuda.is_available():
            return None
        if not SidonUtil.has_sidon():
            return None
        if ModelsUtil.sidon_upsampler is None:
            print_init("Initializing Sidon upsampler (CUDA)...")
            ModelsUtil.sidon_upsampler = SidonUtil()
        return ModelsUtil.sidon_upsampler

    @staticmethod
    def clear_sidon_upsampler() -> None:
        if ModelsUtil.sidon_upsampler is not None:
            ModelsUtil.sidon_upsampler.kill()
            ModelsUtil.sidon_upsampler = None
            MemoryUtil.gc_ram_vram()
