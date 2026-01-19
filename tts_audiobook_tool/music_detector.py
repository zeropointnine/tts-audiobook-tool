from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.util import *
from yamnet_detector import YamnetDetector


class MusicDetector:
    """
    Wraps a YamnetDetector. Used to detect if music exists in a Sound.
    Follows same static pattern used by `Tts` and `Stt`
    """

    _yamnet: YamnetDetector | None = None

    @staticmethod
    def get_model() -> YamnetDetector:
        if not MusicDetector._yamnet:
            print_init("Initializing YAMNet...")
            MusicDetector._yamnet = YamnetDetector()
        return MusicDetector._yamnet
            
    @staticmethod
    def has_instance() -> bool:
        return MusicDetector._yamnet is not None
    
    @staticmethod
    def clear_model() -> None:
        if MusicDetector._yamnet:
            MusicDetector._yamnet.kill()
            MusicDetector._yamnet = None
            from tts_audiobook_tool.memory_util import MemoryUtil
            MemoryUtil.gc_ram_vram()

