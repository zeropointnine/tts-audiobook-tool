import os
from pathlib import Path
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.stt_util import SttUtil
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.text_segmenter import TextSegmenter
import pickle

from tts_audiobook_tool.util import *

class SttFlow:

    # TODO show reminder message once. use same message for both places.
    # TODO delete temp files

    @staticmethod
    def ask_make(prefs: Prefs):

        # TODO add more ui description here

        # Ask text file
        inp = ask_path("Step 1/2 - Enter text file path: ")
        if not inp:
            return
        if not os.path.exists(inp):
            ask_continue(f"File doesn't exist.")
            return
        source_text_path = Path(inp)

        try:
            with open(source_text_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
        except Exception as e:
            ask_continue("Error: {e}\n")
            return

        if not raw_text:
            ask_continue("File has no content.")
            return

        # Ask audio file
        inp = ask_path("Step 2/2 - Enter audiobook file path: ")
        if not inp:
            return
        if not os.path.exists(inp):
            printt(f"File doesn't exist.")
            return
        source_audio_path = inp

        if Path(source_audio_path).suffix == ".mp3":
            # Transcode
            b = ask_confirm("MP3 file must first be transcoded to AAC/MP4. Continue? ")
            if not b:
                return
            path, err = SoundFileUtil.transcode_to_aac(source_audio_path)
            if err:
                printt(err, "error")
                return
            source_audio_path = path

        types = [".flac", ".mp4", ".m4a", ".m4b"]
        if not Path(source_audio_path).suffix in types:
            printt("File suffix must be one of the following: {types}")
            return

        # Normalize path / make path idempotent
        source_audio_path = str(Path(source_audio_path).resolve().as_posix())

        # Check if already has meta
        meta = AppMetaUtil.get_app_metadata(source_audio_path)
        if meta is not None:
            b = ask_confirm("File already has tts-audiobook-tool metadata. Continue anyway? ")
            if not b:
                return

        # Check if already transcribed
        fn = f"transcription {HashFileUtil.calc_hash(str(source_audio_path))}.pkl"
        source_pickle_path = os.path.join(AppUtil.get_app_temp_dir(), fn)

        if not os.path.exists(source_pickle_path):
            source_pickle_path = ""
        else:
            b = ask_confirm("You've previously transcribed this file. Use saved transcription data? ")
            if not b:
                source_pickle_path = ""

        ok = SttFlow.make(
            raw_text,
            source_audio_path=source_audio_path,
            source_pickle_path=source_pickle_path
        )
        if ok and not prefs.has_shown_player_reminder:
            AppUtil.show_player_reminder(prefs)

    @staticmethod
    def make(
            raw_text: str,
            source_audio_path: str,
            source_pickle_path: str=""
    ) -> bool:
        """
        Optional source_pickle_path is the already-transcribed data derived from source_audio_path.
        When using pickle, still need source_audio_path from which we will derive dest file path.

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
                printt(str(e), "error")
                return False

        else:

            # TODO: control-c skips to next step wc is wrong

            printt("Transcribing audio...")
            printt()
            words = SttUtil.transcribe_to_words(str(source_audio_path))
            printt("\a")

            # Save to pickle
            fn = f"transcription {HashFileUtil.calc_hash(str(source_audio_path))}.pkl"
            pickle_path = os.path.join(AppUtil.get_app_temp_dir(), fn)
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

        with open("temp_timed_segments.pickle", "wb") as file: # TODO: meh?
            pickle.dump(timed_text_segments, file)

        # with open("temp_timed_segments.pickle", "rb") as file:  # "rb" = read binary
        #     timed_text_segments = pickle.load(file)

        # [4] Save "abr" audio file

        dest_name = Path(source_audio_path).stem + ".abr" + Path(source_audio_path).suffix # eg, "teh_hobbit.abr.mp4"
        dest_path = str( Path(source_audio_path).with_name(dest_name) )
        print("a", dest_path)
        dest_path = get_unique_file_path(dest_path)
        print("a", dest_path)
        printt(f"\nSaving audio file with metadata")
        printt(f"{dest_path}")
        printt()

        if dest_path.lower().endswith(".flac"):
            ask("TODO: ") # TODO
            return False
        else:
            err = AppMetaUtil.set_mp4_app_metadata(
                str(source_audio_path), raw_text, timed_text_segments, str(dest_path)
            )
            if err:
                printt(f"Error creating audio file: {err}")
                return False

            printt(f"Saved")
            printt()

        # [4b] Review "discontinuity info"

        discon_ranges = TimedTextSegment.get_discontinuities(timed_text_segments)
        if not discon_ranges:
            ask_confirm()
            return True

        b = ask_confirm("View discontinuity info? ")
        if b:
            print_discontinuity_info(timed_text_segments)
            ask_continue()
        return True

# ---

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
       printt(f"{start}-{end} ({end - start + 1}) duration: {time_string(duration)}")

       first_text = timed_text_segments[start].text.strip()
       last_text = timed_text_segments[end].text.strip()
       printt(f"    {ellipsize(first_text, 50)}")
       printt(f"    {ellipsize(last_text, 50)}")
       printt()
