from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from tts_audiobook_tool.app_types import FailResult, PassResult, Sound, ValidationResult, Word
from tts_audiobook_tool.constants_config import DEV_SAVE_INTERMEDIATE_FILES
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.whisper_util import WhisperUtil


@dataclass 
class GenInfo:
    """
    Holds source/transcription text comparison info
    Primarily used for development, debugging
    """
    
    # JSON fields will get presented in this order:
    language_code: str
    index: int = -1
    source: str = ""
    prompt: str = ""
    transcript: str = ""
    normalized_source: str = ""
    normalized_transc: str = ""

    validation_result_class: str = ""
    validation_result_description: str = ""
    validation_num_word_fails: int = -1
    validation_word_fail_threshold: int = -1

    model_error: str = ""

    debug_base_file_path: str = "" # Should have same base filename as sound segment filenames # TODO: s/b just directory


    def __init__(self, base_dir: str, index: int, phrase_group: PhraseGroup | None, language_code: str=""):
        
        self.base_dir = base_dir
        self.index = index        
        self.language_code = language_code
        if not phrase_group:
            return
        
        self.source = phrase_group.as_flattened_phrase().text
        
        # Same logic as used in validation, good enough for now
        self.normalized_source = TextNormalizer.normalize_source(self.source, language_code=language_code) 

        # Same logic as used in GenUtil, good enough for now # TODO: "unhardcode" logic
        s = TextNormalizer.normalize_prompt_common(self.source, language_code=language_code) # TODO: unhardcode 
        s = Tts.get_instance().massage_for_inference(s)
        self.prompt = s
        
    def set_transcript(self, words: list[Word]) -> None:
        self.transcript = WhisperUtil.get_flat_text_from_words(words)
        # Same logic as used in validation, good enough for now
        self.normalized_transc = TextNormalizer.normalize_transcript(
            transcript=self.transcript, normalized_source=self.normalized_source, language_code=self.language_code
        ) 

    def set_validation_result(self, validation_result: ValidationResult) -> None:
        self.validation_result_class = type(validation_result).__name__
        self.validation_result_description = validation_result.get_ui_message()
        if isinstance(validation_result, (PassResult, FailResult)):
            self.validation_num_word_fails = validation_result.num_word_fails
            self.validation_word_fail_threshold = validation_result.word_fail_threshold

    def set_model_error(self, error: str) -> None:
        self.model_error = error

    def save(self, is_realtime: bool, is_retry: bool=False) -> None:

        if not DEV_SAVE_INTERMEDIATE_FILES:
            return

        base_dir = Path(self.debug_base_file_path).parent
        os.makedirs(base_dir, exist_ok=True)

        if self.validation_result_class:
            label = f" [{self.validation_result_class}]"
        elif self.model_error:
            label = " [error]"
        else:
            label = ""

        if is_retry:
            label += " [retry]"

        index_string = str(self.index + 1).zfill(5)
        timestamp = SoundSegmentUtil.make_timestamp_string()
        # Use same start of filename as sound segment files
        if is_realtime:
            file_name = f"[{timestamp}] [{index_string}] [geninfo] {label}.json"
        else:
            file_name = f"[{index_string}] [{timestamp}] [geninfo] {label}.json"

        path = os.path.join(self.base_dir, file_name)
        
        # Remove some unused fields (reduce noise)
        dic = asdict(self)
        if not self.model_error:
            del dic["model_error"]
        if self.validation_num_word_fails == -1:
            del dic["validation_num_word_fails"]
        if self.validation_word_fail_threshold == -1:
            del dic["validation_word_fail_threshold"]

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write( json.dumps(dic, indent=4 ) )
        except Exception as e:
            ...

    @staticmethod
    def from_dict(d: dict) -> GenInfo:
        instance = GenInfo("", -1, None)
        for key, value in d.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    @staticmethod
    def save_sound(dir_path: str, index: int, label: str, sound: Sound, is_realtime: bool): 
        
        if not DEV_SAVE_INTERMEDIATE_FILES:
            return
        
        os.makedirs(dir_path, exist_ok=True)        
        index_string = str(index + 1).zfill(5)
        timestamp = SoundSegmentUtil.make_timestamp_string()
        
        # Use same start of filename as sound segment files
        if is_realtime:
            file_name = f"[{timestamp}] [{index_string}] [{label}].flac"
        else:
            file_name = f"[{index_string}] [{timestamp}] [{label}].flac"
        path = os.path.join(dir_path, file_name)
        _ = SoundFileUtil.save_flac(sound, path)
