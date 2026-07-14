"""
Informal validation harness for an existing sound-segments directory.
Requires accompanying JSON files (does not do transcription dynamically).
"""

from pathlib import Path

from tts_audiobook_tool.app_types import ConcreteWord, Strictness, Word
from tts_audiobook_tool.app_types.validation_result import TranscriptResult
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.project_support.sound_segment_util import get_segment_stt_info_path
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.util import printt
from tts_audiobook_tool.validator import Validator


DIR = "/d/w/w/test_txt/segments"

LANGUAGE_CODE = "en"
STRICTNESS = Strictness.HIGH
PRINT_ONLY_ERRORS = True


def get_flac_paths(dir_path: str) -> list[Path]:
    return sorted(
        path for path in Path(dir_path).iterdir()
        if path.is_file() and path.suffix.lower() == ".flac" and path.stat().st_size > 0
    )


def make_words_from_json(word_dicts: list[dict]) -> list[Word] | str:
    words: list[Word] = []
    try:
        for word_dict in word_dicts:
            words.append(ConcreteWord(
                start=float(word_dict["start"]),
                end=float(word_dict["end"]),
                word=str(word_dict["word"]),
                probability=float(word_dict.get("probability", 1.0)),
            ))
    except Exception as e:
        return f"Could not load transcript words from sidecar JSON: {e}"
    return words


def load_sidecar_info(flac_path: Path) -> tuple[str, list[Word]] | str:
    json_path = get_segment_stt_info_path(flac_path)
    if not json_path.exists():
        return f"Missing sidecar JSON: {json_path.name}"

    result = SegmentTranscriptUtil.load(json_path)
    if isinstance(result, str):
        return f"Could not load sidecar JSON: {json_path.name}: {result}"

    words = None
    if result.transcript_words:
        words = make_words_from_json(result.transcript_words)
        if isinstance(words, str):
            return words

    if words is None:
        return f"Missing transcript words in sidecar JSON: {json_path.name}"

    return result.source, words


def validate_file(path: Path) -> TranscriptResult | str:
    sound = SoundFileUtil.load(str(path))
    if isinstance(sound, str):
        return sound

    sidecar_info = load_sidecar_info(path)
    if isinstance(sidecar_info, str):
        return sidecar_info
    else:
        source, words = sidecar_info

    validation_result = Validator.validate(sound, source, words, LANGUAGE_CODE, STRICTNESS)
    if not isinstance(validation_result, TranscriptResult):
        return f"Skipped: validation result has no transcript ({type(validation_result).__name__})"
    return validation_result


def get_result_counts(transcript_result: TranscriptResult | str) -> tuple[int, int, int]:
    """Returns failed/passed/skipped."""

    if isinstance(transcript_result, str):
        is_skip = transcript_result.startswith("Skipped:")
        return (0, 0, 1 if is_skip else 0)

    return (1 if transcript_result.is_fail else 0, 0 if transcript_result.is_fail else 1, 0)


def should_print_result(transcript_result: TranscriptResult | str) -> bool:
    if not PRINT_ONLY_ERRORS:
        return True

    if isinstance(transcript_result, str):
        return not transcript_result.startswith("Skipped:")

    return transcript_result.is_fail


def print_result(path: Path, transcript_result: TranscriptResult | str) -> None:
    """Prints result details when caller has decided the result should be shown."""

    link = make_terminal_hyperlink(str(path), is_file=True)
    printt(link)

    if isinstance(transcript_result, str):
        is_skip = transcript_result.startswith("Skipped:")
        color = COL_DIM if is_skip else COL_ERROR
        printt(f"{color}{transcript_result}{COL_DEFAULT}")
        printt()
        return

    printt(transcript_result.__class__.__name__ + ": " + transcript_result.get_ui_message_with_post_processing())
    printt()


def main() -> None:

    Tts.init_local_model_type()

    print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)
    print(f"{COL_DEFAULT}Validating FLAC files in: {COL_DIM}{DIR}{COL_DEFAULT}")
    print(f"{COL_DEFAULT}Language: {COL_DIM}{LANGUAGE_CODE}{COL_DEFAULT}")
    print(f"{COL_DEFAULT}Strictness: {COL_DIM}{STRICTNESS.label}{COL_DEFAULT}")
    print()

    paths = get_flac_paths(DIR)
    if not paths:
        print(f"{COL_ERROR}No FLAC files found.{COL_DEFAULT}")
        return

    num_fail = 0
    num_pass = 0
    num_skip = 0
    for path in paths:
        result = validate_file(path)
        fail, passed, skipped = get_result_counts(result)
        num_fail += fail
        num_pass += passed
        num_skip += skipped
        if should_print_result(result):
            print_result(path, result)

    print(f"{COL_DEFAULT}Done. {COL_OK}passed={num_pass}{COL_DEFAULT}, {COL_ERROR}failed={num_fail}{COL_DEFAULT}, {COL_DIM}skipped={num_skip}{COL_DEFAULT}")


if __name__ == "__main__":
    main()
