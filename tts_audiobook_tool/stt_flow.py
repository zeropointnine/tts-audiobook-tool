import os
from pathlib import Path
import pickle
import time
from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.stt_util import SttUtil
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.constants_config import *

from tts_audiobook_tool.util import *

class SttFlow:
    """
    Speech-to-text enchance-an-audiobook "user flow"
    """

    @staticmethod
    def ask_and_make(prefs: Prefs) -> None:

        print_heading("Enhance existing audiobook")

        AppUtil.show_hint_if_necessary(prefs, HINT_STT_ENHANCE)

        # [1] Ask text file
        if DEV and False:
            inp = r"exc.txt"
        else:
            inp = AskUtil.ask_file_path("Step 1/2 - Enter text file path: ", "Step 1/2: Select text file")
        if not inp:
            return
        if not os.path.exists(inp):
            AskUtil.ask_enter_to_continue(f"File doesn't exist.")
            return
        source_text_path = Path(inp)

        try:
            with open(source_text_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
        except Exception as e:
            AskUtil.ask_error(f"Error: {e}")
            return

        if not raw_text:
            AskUtil.ask_enter_to_continue("File has no content.")
            return

        time.sleep(1)

        # [2] Ask audio file
        if DEV and False:
            inp = r"exc.flac"
        else:
            inp = AskUtil.ask_file_path("Step 2/2 - Enter audiobook file path: ", "Step 2/2: Select audiobook file")

        if not inp:
            return
        if not os.path.exists(inp):
            printt(f"File doesn't exist.")
            return
        source_audio_path = inp

        # Optional transcode step
        if Path(source_audio_path).suffix == ".mp3":
            AppUtil.show_hint_if_necessary(prefs, HINT_MULTIPLE_MP3S)
            b = AskUtil.ask_confirm("MP3 file must first be transcoded to AAC. Do this now? ")
            if not b:
                return
            path, err = SoundFileUtil.transcode_to_aac(source_audio_path)
            if err:
                AskUtil.ask_error(err)
                return
            source_audio_path = path

        # Make normalized, 'idempotent' path
        source_audio_path = str(Path(source_audio_path).resolve().as_posix())

        types = [".flac", ".mp4", ".m4a", ".m4b"]
        if not Path(source_audio_path).suffix in types:
            printt("File suffix must be one of the following: {types}")
            return

        # Check if already has meta
        meta = AppMetadata.load_from_file(source_audio_path)
        if meta is not None:
            if DEV and False:
                b = True
            else:
                b = AskUtil.ask_confirm("Audio file already has tts-audiobook-tool metadata. Continue anyway? ")
            if not b:
                return

        # [3] Calc hash
        source_audio_hash, err = AppUtil.calc_hash_file(
            source_audio_path,
            print_progress_text="Calculating audio file hash:"
        )
        if err:
            AskUtil.ask_error(err)
            return

        # [4] Check if already has transcription pickle file
        transcription_pickle_path = make_transcription_pickle_file_path(source_audio_hash)
        if not os.path.exists(transcription_pickle_path):
            transcription_pickle_path = ""
        else:
            if DEV and False:
                b = True
            else:
                b = AskUtil.ask_confirm("You've previously transcribed this audio file. Use saved transcription data? ")
            if not b:
                transcription_pickle_path = ""

        # TODO: new [5] Ask for language code hint

        # [5] Start
        SttFlow.make(
            prefs,
            raw_text,
            source_audio_path=source_audio_path,
            source_audio_hash=source_audio_hash,
            source_pickle_path=transcription_pickle_path
        )

    @staticmethod
    def make(
            prefs: Prefs,
            raw_text: str,
            source_audio_path: str,
            source_audio_hash: str,
            source_pickle_path: str=""
    ) -> bool:
        """
        Optional source_pickle_path is the already-transcribed data from source_audio_path,

        Returns True for success
        """

        # [1] Make segmented source text

        printt("Segmenting source text...")
        printt()
        groups = PhraseGrouper.text_to_groups(raw_text, max_words=MAX_WORDS_PER_SEGMENT_STT)
        phrases = PhraseGroup.flatten_groups(groups)

        # [2] Transcribe audio file (or load pickle file)

        if source_pickle_path:

            try:
                with open(source_pickle_path, "rb") as file:
                    words = pickle.load(file)
            except Exception as e:
                AskUtil.ask_error(make_error_string(e))
                return False

        else:

            print_heading(f"Transcribing audio... {COL_DIM}(This may take some time)", dont_clear=True, non_menu=True)
            printt()

            # Always use best whisper model
            Stt.set_variant(SttVariant.LARGE_V3)

            # Warm up
            _ = Stt.get_whisper()

            words = SttUtil.transcribe_to_words(str(source_audio_path))
            if words is None: # interrupted
                printt("")
                print_feedback("Interrupted")
                return False

            printt("\a")

            # Restore variant / clean up model if necessary
            Stt.set_variant(prefs.stt_variant)

            # Save transcription data to pickle
            pickle_path = make_transcription_pickle_file_path(source_audio_hash)
            try:
                with open(pickle_path, "wb") as file:
                    pickle.dump(words, file)
            except:
                pass # eat

        # [3] "Merge" source text and transcribed text data

        print_heading("Merging data...", dont_clear=True, non_menu=True)

        timed_text_segments, did_interrupt = SttUtil.make_timed_phrases(phrases, words)

        if did_interrupt:
            print_feedback("Interrupted")
            return False

        # [4] Save "abr" audio file

        dest_name = Path(source_audio_path).stem + ".abr" + Path(source_audio_path).suffix # eg, "teh_hobbit.abr.m4a"
        dest_path = str( Path(source_audio_path).with_name(dest_name) )
        dest_path = make_unique_file_path(dest_path)
        printt(f"\nSaving audio file with added custom metadata")
        printt()

        meta = AppMetadata(timed_text_segments, raw_text, has_section_break_audio=False)
        if dest_path.lower().endswith(".flac"):
            save_error = AppMetadata.save_to_flac(meta, str(source_audio_path), str(dest_path))
        else:
            save_error = AppMetadata.save_to_mp4(meta, str(source_audio_path), str(dest_path))
        if save_error:
            printt(f"{COL_ERROR}Error: {save_error}")
            printt()
        else:
            printt(f"{COL_ACCENT}Saved {dest_path}")
            printt()

        AppUtil.show_hint_if_necessary(prefs, HINT_STT_ENHANCE_CACHED)

        # [4b] Review "discontinuity info"
        b = AskUtil.ask_confirm("View discontinuity info summary? ")
        if b:
            print_discontinuity_info(timed_text_segments)
            AskUtil.ask_enter_to_continue()

        if not save_error:
            AppUtil.show_player_hint_if_necessary(prefs)


        return bool(save_error)

# ---

def make_transcription_pickle_file_path(hash: str) -> str:
    file_name = f"transcription {hash}.pkl"
    return os.path.join(AppUtil.get_app_user_dir(), file_name)

def print_discontinuity_info(timed_text_segments: list[TimedPhrase]):

    print_heading("Unmatched text segments:", dont_clear=True, non_menu=True)
    printt()

    discon_ranges = TimedPhrase.get_discontinuities(timed_text_segments)
    if not discon_ranges:
        printt("No items found")
        printt()
        return

    for start, end in discon_ranges:

        num_consecutive = end - start + 1

        if num_consecutive == 1:
            printt(f"Line {start}")
            text = timed_text_segments[start].text.strip()
            printt(f"    {COL_DIM}Line:{COL_DEFAULT} {ellipsize(text, 50)}")
            printt()
        else:
            printt(f"Lines {start} to {end} ({COL_ERROR}{num_consecutive} lines{COL_DEFAULT})")
            first_text = timed_text_segments[start].text.strip()
            last_text = timed_text_segments[end].text.strip()
            printt(f"    {COL_DIM}First line:{COL_DEFAULT} {ellipsize(first_text, 50)}")
            printt(f"    {COL_DIM}Last line:{COL_DEFAULT} {ellipsize(last_text, 50)}")
            printt()
