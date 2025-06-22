from __future__ import annotations

from enum import Enum, auto
import time
from typing import NamedTuple
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.util import *

class ValidateUtil:

    @staticmethod
    def validate_items(
        items: list[ValidateItem],
    ) -> None:
        """
        Validation-only flow

        For each file, tags validated file name with "pass", else prints fail info.
        Skips files already tagged with "pass".
        Prints counts at end.

        Inits whisper at start, and unloads it at end.
        """

        #x

        start_time = time.time()
        num_analysed = 0
        num_detected = 0 # including ones that get corrected
        num_deleted = 0
        num_corrected = 0
        num_validated = 0

        Shared.mode = "validating"

        whisper_model = Shared.get_whisper()

        for item in items:

            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                break

            whisper_data = SoundUtil.transcribe_file(whisper_model, item.path)
            if isinstance(whisper_data, str):
                printt(f"{COL_ERROR}Unexpected whisper result, skipping item: {whisper_data}")
                continue

            sound =SoundFileUtil.load(item.path, WHISPER_SAMPLERATE)
            if isinstance(sound, str):
                printt(f"{COL_ERROR}Error loading file, skipping item: {sound}")
                continue

            validate_result = ValidateUtil.validate_item(sound, item.text, whisper_data)

            num_analysed += 1

            item_info = f"{COL_ACCENT}[{COL_DEFAULT}{num_analysed}{COL_ACCENT}/{COL_DEFAULT}{len(items)}{COL_ACCENT}] Line index {COL_DEFAULT}{item.index}{COL_ACCENT}:\n"
            item_info += f"{COL_DIM}{item.path}\n{COL_DEFAULT}"

            match validate_result.result:

                case ValidateResultType.VALID:
                    # TODO rename
                    num_validated += 1
                    ...
                case ValidateResultType.TRIMMABLE:

                    ...
                case ValidateResultType.INVALID:
                    ...

        # Done - print info
        printt(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()
        printt(f"{num_analysed} file/s analysed")
        printt(f"{num_validated} file/s marked as 'verified'")
        printt(f"{num_detected} error/s detected")
        printt()

        Shared.clear_whisper()

    @staticmethod
    def validate_and_save(
        sound: Sound,
        source_text: str,
        whisper_data: dict,
        save_path: str,
        save_if_bad: bool,
        play_on_save: bool
    ) -> ValidateAction:
        """
        Takes in a generated + postprocessed sound clip
        Returns ValidateAction which describes what action was taken
        """

        def save(sound_to_save: Sound, success_action: ValidateAction):
            err = SoundFileUtil.save_flac(save_path, sound_to_save)
            if not err:
                if play_on_save:
                    SoundFileUtil.play_sound_async(sound_to_save)
                return success_action
            else:
                return ValidateAction(ValidateActionType.ACTION_FAILED, err)

        validate_result = ValidateUtil.validate_item(sound, source_text, whisper_data)

        match validate_result.result:

            case ValidateResultType.VALID:

                return save(sound, ValidateAction(ValidateActionType.SAVED, "Saved"))

            case ValidateResultType.TRIMMABLE:

                if not validate_result.trim_start and not validate_result.trim_end:
                    raise ValueError("Bad value")

                start_time = validate_result.trim_start or 0
                end_time = validate_result.trim_end or sound.duration
                new_sound = SoundUtil.trim(sound, start_time, end_time)
                action = ValidateAction(ValidateActionType.TRIMMED_AND_SAVED, "Fixed: " + validate_result.message)
                return save(new_sound, action)

            case ValidateResultType.INVALID:

                if save_if_bad:
                    action = ValidateAction(ValidateActionType.INVALID_SAVED_ANYWAY, validate_result.message)
                    # Add fail tag to filename
                    save_path = insert_bracket_tag_file_path(save_path, "fail")
                    return save(sound, action)
                else:
                    return ValidateAction(ValidateActionType.INVALID_DIDNT_SAVE, validate_result.message)


    @staticmethod
    def validate_item(sound: Sound, reference_text: str, whisper_data: dict) -> ValidateResult:

        # Order of tests matter here

        # Static audio test
        is_static = TranscribeUtil.is_audio_static(sound, whisper_data)
        if is_static:
            return ValidateResult(ValidateResultType.INVALID, "Audio is static")

        # Substring test
        timestamps = TranscribeUtil.get_substring_time_range(reference_text, whisper_data)
        if timestamps:
            message = f"Excess words detected, substring at {timestamps[0]:.2f}-{timestamps[1]:.2f}"
            result = ValidateResult(
                ValidateResultType.TRIMMABLE,
                message,
                timestamps[0],
                timestamps[1]
            )
            return result

        # Repeat phrases test
        repeats = TranscribeUtil.find_bad_repeats(reference_text, whisper_data)
        if repeats:
            return ValidateResult(ValidateResultType.INVALID, f"Repeated word/phrase: {", ".join(repeats)}")

        # Word count delta test
        fail_reason = TranscribeUtil.is_word_count_fail(reference_text, whisper_data)
        if fail_reason:
            return ValidateResult(ValidateResultType.INVALID, fail_reason)

        # Excess audio
        trim_start_time = TranscribeUtil.get_semantic_match_start_time_trim(reference_text, whisper_data, sound)
        trim_end_time = TranscribeUtil.get_semantic_match_end_time_trim(reference_text, whisper_data, sound)

        messages = []
        if trim_start_time:
            messages.append(f"Excess at start ({(trim_start_time):.2f}s)")
        if trim_end_time:
            messages.append(f"Excess at end ({(sound.duration - trim_end_time):.2f}s)")

        if trim_start_time or trim_end_time:
            result = ValidateResult(
                ValidateResultType.TRIMMABLE,
                ", ".join(messages),
                trim_start_time,
                trim_end_time
            )
            return result

        # At this point we consider the item to have "passed"
        return ValidateResult(ValidateResultType.VALID, "Passed validation tests")

# ---

class ValidateItem:
    def __init__(self, i: int, path: str, text: str):
        self.index = i
        self.path = path
        self.text = text

        duration = AudioMetaUtil.get_audio_duration(self.path)
        if not isinstance(duration, float):
            L.w(f"Couldn't get duration for {self.path}")
            duration = -1
        self.duration = duration

class ValidateResult(NamedTuple):
    result: ValidateResultType
    message: str
    trim_start: float | None = None
    trim_end: float | None = None

class ValidateResultType(Enum):
    VALID = auto()
    TRIMMABLE = auto()
    INVALID = auto()

# ---

class ValidateAction(NamedTuple):
    action: ValidateActionType
    message: str

class ValidateActionType(Enum):
    SAVED = auto()
    TRIMMED_AND_SAVED = auto()
    INVALID_SAVED_ANYWAY = auto()
    ACTION_FAILED = auto()
    INVALID_DIDNT_SAVE = auto()
