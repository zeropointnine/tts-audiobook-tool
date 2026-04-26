import torch

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

    sidon_upscaler: SidonUtil | None = None


    @staticmethod
    def warm_up_models(state: State, never_yamnet: bool=False) -> bool:
        """
        Instantiates required models for main inference flow, prints updates.

        Params:
            should_not_stt - if True, does not init STT model even if state would dictate otherwise
            should_yamnet - if True, initializes YamnetDetector

        Returns:
            True if control-c was pressed
        """

        should_tts = not Tts.instance_exists()        
        should_stt = not Stt.should_skip(state) and not Stt.has_instance()
        should_yamnet = Tts.get_type().value.hallucinates_music and not MusicDetector.has_instance() and not never_yamnet
        
        shoulds = [should_tts, should_stt, should_yamnet]
        num_shoulds = sum(1 for item in shoulds if item)
        if num_shoulds == 0:
            return False

        if not should_stt:
            # "Lazy unload", relevant for user flows like: 
            # Do inference, choose unsupported validation language, do inference
            Stt.clear_stt_model()

        if num_shoulds >= 2:
            print_init("Warming up models...")

        SigIntHandler().set("model init")

        # Init TTS
        if should_tts:
            _ = Tts.get_instance()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True

        # Init STT
        if should_stt:
            _ = Stt.get_whisper()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True
        
        # Init YAMNet
        if should_yamnet:
            _ = MusicDetector.get_model()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True

        SigIntHandler().clear()
        return False

    @staticmethod
    def clear_all_models(except_sidon: bool = False) -> None:

        Stt.clear_stt_model()
        Tts.clear_tts_model()
        MusicDetector.clear_model()
        if not except_sidon:
            ModelsUtil.clear_sidon_upscaler()

        # For good measure
        MemoryUtil.gc_ram_vram()

    @staticmethod
    def is_any_model_loaded() -> bool:
        return Stt.has_instance() or \
            Tts.instance_exists() or \
            MusicDetector.has_instance() or \
            ModelsUtil.sidon_upscaler is not None

    @staticmethod
    def get_sidon_upscaler() -> SidonUtil | None:
        if not torch.cuda.is_available():
            return None
        if not SidonUtil.has_sidon():
            return None
        if ModelsUtil.sidon_upscaler is None:
            print_init("Initializing Sidon upscaler (CUDA)...")
            ModelsUtil.sidon_upscaler = SidonUtil()
        return ModelsUtil.sidon_upscaler

    @staticmethod
    def clear_sidon_upscaler() -> None:
        if ModelsUtil.sidon_upscaler is not None:
            ModelsUtil.sidon_upscaler.kill()
            ModelsUtil.sidon_upscaler = None
            MemoryUtil.gc_ram_vram()
