import gc
import torch
import whisper
from whisper.model import Whisper

from tts_audiobook_tool.util import *


class Shared:

    _whisper: Whisper | None = None

    @staticmethod
    def get_whisper() -> Whisper:
        if not Shared._whisper:
            printt("Initializing whisper model...")
            printt()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Shared._whisper = whisper.load_model("turbo", device=device)
        return Shared._whisper

    @staticmethod
    def clear_whisper() -> None:
        if not Shared._whisper:
            return
        printt("Unloading whisper...")
        printt()
        del Shared._whisper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ---

    mode = ""
    stop_flag = False