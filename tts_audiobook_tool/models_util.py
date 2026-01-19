from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.music_detector import MusicDetector
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import *
from yamnet_detector import YamnetDetector


class ModelsUtil:
    """
    Multiple-model management util functions
    Plus YamnetDetector
    """

    @staticmethod
    def warm_up_models(state: State) -> bool:
        """
        Instantiates required models for main inference flow, prints updates.

        Params:
            should_not_stt - if True, does not init STT model even if state would dictate otherwise
            should_yamnet - if True, initializes YamnetDetector

        Returns:
            True if control-c was pressed
        """

        should_tts = not Tts.instance_exists()        
        should_stt = Stt.should_stt(state) and not Stt.has_instance()
        should_yamnet = Tts.get_type().value.hallucinates_music and not MusicDetector.has_instance()
        
        shoulds = [should_tts, should_stt, should_yamnet]
        num_should = sum(1 for item in shoulds if item is not None)
        if num_should == 0:
            return False

        if not should_stt:
            # "Lazy unload", useful for user flows like: 
            # Do inference, choose unsupported validation language, do inference
            Stt.clear_stt_model()

        if num_should >= 2:
            print_init("Warming up models...")

        SigIntHandler().set("model init")

        # TTS
        if should_tts:
            _ = Tts.get_instance()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True

        if should_stt or should_yamnet:
            printt() # yes rly

        # STT
        if should_stt:
            _ = Stt.get_whisper()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True
        
        # YAMNet
        if should_yamnet:
            _ = MusicDetector.get_model()

        if SigIntHandler().did_interrupt:
            SigIntHandler().clear()
            return True

        SigIntHandler().clear()
        return False

    @staticmethod
    def clear_all_models() -> None:
        
        Stt.clear_stt_model()
        
        Tts.clear_tts_model()

        MusicDetector.clear_model()
        
        # For good measure
        from tts_audiobook_tool.memory_util import MemoryUtil
        MemoryUtil.gc_ram_vram() 


