import os
from pathlib import Path
import pickle
import time
from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.app_types import SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.stt_util import SttUtil
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.constants_config import *

from tts_audiobook_tool.util import *

class SttFlow:

    # TODO show reminder message once. use same message for both places.
    # TODO delete temp files

    @staticmethod
    def ask_and_make(prefs: Prefs) -> None:

        # TODO add more ui description here

        # [1] Ask text file
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

        # [2] Ask audio file
        time.sleep(1)
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

        types = [".flac", ".mp4", ".m4a", ".m4b"]
        if not Path(source_audio_path).suffix in types:
            printt("File suffix must be one of the following: {types}")
            return

        # Normalize path / make idempotent
        source_audio_path = str(Path(source_audio_path).resolve().as_posix())

        # Check if already has meta
        meta = AppMetadata.load_from_file(source_audio_path)
        if meta is not None:
            b = AskUtil.ask_confirm("Audio file already has tts-audiobook-tool metadata. Continue anyway? ")
            if not b:
                return

        # [3] Calc hash
        source_audio_hash, err = AppUtil.calc_hash_file(source_audio_path)
        if err:
            AskUtil.ask_error(err)
            return

        # [4] Check if already has transcription pickle file
        transcription_pickle_path = make_transcription_pickle_file_path(source_audio_hash)
        if not os.path.exists(transcription_pickle_path):
            transcription_pickle_path = ""
        else:
            b = AskUtil.ask_confirm("You've previously transcribed this file. Use saved transcription data? ")
            if not b:
                transcription_pickle_path = ""

        # [5] Start
        ok = SttFlow.make(
            prefs,
            raw_text,
            source_audio_path=source_audio_path,
            source_audio_hash=source_audio_hash,
            source_pickle_path=transcription_pickle_path
        )
        if ok:
            AppUtil.show_player_hint_if_necessary(prefs)

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
        text_segments = TextSegmenter.segment_text(raw_text, max_words=MAX_WORDS_PER_SEGMENT_STT)

        # [2] Transcribe audio file (or load pickle file)

        if source_pickle_path:

            try:
                with open(source_pickle_path, "rb") as file:
                    words = pickle.load(file)
            except Exception as e:
                AskUtil.ask_error(make_error_string(e))
                return False

        else:

            # TODO: control-c skips to next step wc is wrong

            printt("Transcribing audio...")
            printt()

            # Always use best whisper model
            Stt.set_variant(SttVariant.LARGE_V3)

            # Warm up
            _ = Stt.get_whisper()

            words = SttUtil.transcribe_to_words(str(source_audio_path))
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

        # TODO: rly needs to be cancellable
        printt("Merging data...")
        printt()
        timed_text_segments = SttUtil.make_timed_text_segments(text_segments, words)

        # with open("temp_timed_segments.pickle", "wb") as file: # TODO: meh?
        #     pickle.dump(timed_text_segments, file)

        # [4] Save "abr" audio file

        dest_name = Path(source_audio_path).stem + ".abr" + Path(source_audio_path).suffix # eg, "teh_hobbit.abr.m4a"
        dest_path = str( Path(source_audio_path).with_name(dest_name) )
        dest_path = get_unique_file_path(dest_path)
        printt(f"\nSaving audio file with metadata")
        printt(f"{dest_path}")
        printt()

        if dest_path.lower().endswith(".flac"):
            AskUtil.ask("TODO: ") # TODO
            return False
        else:
            meta = AppMetadata(raw_text, timed_text_segments)
            err = AppMetadata.save_to_mp4(meta, str(source_audio_path), str(dest_path))
            if err:
                printt(f"Error creating audio file: {err}")
                return False

            printt(f"Saved")
            printt()

        # [4b] Review "discontinuity info"

        discon_ranges = TimedTextSegment.get_discontinuities(timed_text_segments)
        if not discon_ranges:
            AskUtil.ask_confirm()
            return True

        b = AskUtil.ask_confirm("View discontinuity info summary? ")
        if b:
            print_discontinuity_info(timed_text_segments)
            AskUtil.ask_enter_to_continue()
        return True

# ---

def make_transcription_pickle_file_path(hash: str) -> str:
    file_name = f"transcription {hash}.pkl"
    return os.path.join(AppUtil.get_app_user_dir(), file_name)

def print_discontinuity_info(timed_text_segments: list[TimedTextSegment]):

    discon_ranges = TimedTextSegment.get_discontinuities(timed_text_segments)
    if not discon_ranges:
        printt("No discontinuity issues found")
        printt()
        return

    printt("Unmatched, contiguous text segments:")
    printt()
    for start, end in discon_ranges:
       # TODO length in time
       start_time = timed_text_segments[start - 1].time_end if start > 0 else 0
       end_time = timed_text_segments[end + 1].time_start if end + 1 < len(timed_text_segments) else start_time
       duration = end_time - start_time # TODO unconfirmed
       printt(f"Lines {start}-{end} ({end - start + 1}) duration: {duration_string(duration)}")

       first_text = timed_text_segments[start].text.strip()
       last_text = timed_text_segments[end].text.strip()
       printt(f"    {ellipsize(first_text, 50)}")
       printt(f"    {ellipsize(last_text, 50)}")
       printt()
