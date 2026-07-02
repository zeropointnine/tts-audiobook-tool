import torch

from tts_audiobook_tool.app_types import ModelWarmUpResult
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.sidon_util import SidonUtil
from tts_audiobook_tool.sound.yamnet_detector import YamnetDetector
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *


class ModelManager:
    """
    Multiple-model management utils.

    Including YamnetDetector and SidonModel (which it holds as static instance)
    """

    sidon_upsampler: SidonUtil | None = None
    yamnet_detector: YamnetDetector | None = None


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
        shoulds = [should_tts, should_stt]
        num_shoulds = sum(1 for item in shoulds if item)
        
        if skip_yamnet:
            # "Lazy unload", relevant for user flows like: 
            # Do inference with MossLocal, select MossDelay model, do inference
            ModelManager.clear_yamnet_detector()
        
        if num_shoulds == 0 and (ModelManager.has_yamnet_detector() or skip_yamnet):
            return ModelWarmUpResult()

        if not should_stt:
            # "Lazy unload", relevant for user flows like: 
            # Do inference, choose unsupported validation language, do inference
            Stt.clear_stt_model()

        if num_shoulds >= 2:
            print_init("Warming up models...")

        Interrupts().set("model init")

        # Init TTS
        if should_tts:
            try:
                tts_instance = Tts.get_instance()
            except Exception as e:
                Interrupts().clear()
                app_memory.gc_ram_vram()
                err_msg = str(e)
                return ModelWarmUpResult(error=err_msg)
        else:
            tts_instance = Tts.get_instance_if_exists()
            if tts_instance is None:
                try:
                    tts_instance = Tts.get_instance()
                except Exception as e:
                    Interrupts().clear()
                    app_memory.gc_ram_vram()
                    err_msg = str(e)
                    return ModelWarmUpResult(error=err_msg)

        # Check for interrupt
        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)

        # Init STT
        if should_stt:
            try:
                Stt.eager_warm_up_for_inference()
            except Exception as e:
                Interrupts().clear()
                app_memory.gc_ram_vram()
                err_msg = str(e)
                return ModelWarmUpResult(error=err_msg)

        # Check for interrupt
        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)
        
        # Init YAMNet
        should_yamnet = (
            Tts.get_class().can_hallucinate_music(state.project, tts_instance)
            and not ModelManager.has_yamnet_detector()
            and not skip_yamnet
        )
        if should_yamnet:
            try:
                _ = ModelManager.get_yamnet_detector()
            except Exception as e:
                Interrupts().clear()
                app_memory.gc_ram_vram()
                err_msg = str(e)
                return ModelWarmUpResult(error=err_msg)
        else:
            # "Lazy unload", relevant for user flows like: 
            # Do inference with MossLocal, select MossDelay model, do inference
            ModelManager.clear_yamnet_detector()

        # Check for interrupt
        if Interrupts().did_interrupt:
            Interrupts().clear()
            return ModelWarmUpResult(did_interrupt=True)

        Interrupts().clear()
        return ModelWarmUpResult()

    @staticmethod
    def clear_all_models(except_sidon: bool = False) -> None:

        Stt.clear_stt_model()
        Tts.clear_tts_model()
        ModelManager.clear_yamnet_detector()
        if not except_sidon:
            ModelManager.clear_sidon_upsampler()

        # For good measure
        app_memory.gc_ram_vram()

    @staticmethod
    def is_any_model_loaded() -> bool:
        return Stt.has_instance() or \
            Tts.instance_exists() or \
            ModelManager.has_yamnet_detector() or \
            ModelManager.sidon_upsampler is not None

    @staticmethod
    def get_yamnet_detector() -> YamnetDetector:
        if ModelManager.yamnet_detector is None:
            print_init("Initializing YAMNet...")
            ModelManager.yamnet_detector = YamnetDetector()
        return ModelManager.yamnet_detector

    @staticmethod
    def has_yamnet_detector() -> bool:
        return ModelManager.yamnet_detector is not None

    @staticmethod
    def clear_yamnet_detector() -> None:
        if ModelManager.yamnet_detector is not None:
            ModelManager.yamnet_detector.kill()
            ModelManager.yamnet_detector = None
            app_memory.gc_ram_vram()

    @staticmethod
    def get_sidon_upsampler() -> SidonUtil | None:
        if not torch.cuda.is_available():
            return None
        if not SidonUtil.has_sidon():
            return None
        if ModelManager.sidon_upsampler is None:
            print_init("Initializing Sidon upsampler (CUDA)...")
            ModelManager.sidon_upsampler = SidonUtil()
        return ModelManager.sidon_upsampler

    @staticmethod
    def clear_sidon_upsampler() -> None:
        if ModelManager.sidon_upsampler is not None:
            ModelManager.sidon_upsampler.kill()
            ModelManager.sidon_upsampler = None
            app_memory.gc_ram_vram()
