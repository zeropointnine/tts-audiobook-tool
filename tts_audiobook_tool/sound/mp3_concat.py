import os

import natsort
from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.audio_convert import convert_audio_files, describe_error
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class SoundConcatTranscodeUtil:

    @staticmethod
    def ask_and_concat_audio_files():

        # TODO: need pseudo-menu

        src_dir = ask.ask_dir_path("Enter directory of audio files:", "Select directory with audio files")

        if not src_dir:
            return

        if not os.path.exists(src_dir):
            ask.ask_enter_to_continue("No such directory")
            return

        src_dir = os.path.abspath(src_dir)

        SUFFIXES = (".mp3", ".m4a", ".m4b", ".flac")
        audio_files = [f for f in os.listdir(src_dir) if f.lower().endswith(SUFFIXES)]
        audio_files = natsort.natsorted(audio_files, alg=natsort.ns.IGNORECASE)

        if not audio_files:
            print_feedback("No audio files found in directory")
            return

        # Print list and confirm
        if len(audio_files) > 1:
            s = "Will concatenate files in the following order:"
            printt(s)
            printt()
            for i, item in enumerate(audio_files):
                printt(f"[{i+1}] {item}")
            printt()

        while True:
            s = f"Press {make_hotkey_string('F')} to export as FLAC or {make_hotkey_string('M')} for M4A/AAC: "
            hotkey = ask.ask_hotkey(s)
            if hotkey == "f":
                is_flac = True
                break
            elif hotkey == "m":
                is_flac = False
                break
            elif not hotkey:
                return

        # TODO: if only 1 file and same suffix, prompt and exit

        if len(audio_files) > 1:
            s = "Transcoding and concatenating audio files..."
        else:
            s = "Transcoding audio file..."
        printt(s)
        printt()

        # Output file is saved in same dir
        file_name = "concatenated" + (".flac" if is_flac else ".m4a")

        err = SoundConcatTranscodeUtil.concatenate_audio_files(
            audio_files, src_dir, file_name, is_flac
        )
        if err:
            ask.ask_error(err)
            return

        printt("Finished")
        printt()
        s = f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open directory: "
        hotkey = ask.ask_hotkey(s)
        if hotkey == "o":
            err = open_directory_in_gui(src_dir)
            if err:
                ask.ask_error(err)

    @staticmethod
    def concatenate_audio_files(
            audio_files: list[str], dest_dir: str, dest_filename: str, is_flac: bool
    ) -> str:
        """
        Decodes the files in order (relative paths are relative to `dest_dir`) and
        writes them back to back to `dest_dir/dest_filename`, as FLAC or AAC.
        Inputs may mix formats, sample rates and channel counts: the output uses
        the highest input sample rate, and stereo if any input isn't mono.
        Control-C cancels (partial output is removed).

        Returns error string on fail, else empty string
        """

        if not audio_files:
            return "No files specified"

        os.makedirs(dest_dir, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, dest_filename)
        suffix = Path(dest_file_path).suffix.lower()
        if (suffix != ".flac") if is_flac else (suffix not in AAC_SUFFIXES):
            return f"Unsupported output file name for {'FLAC' if is_flac else 'AAC'}: {dest_filename}"

        source_paths = [os.path.join(dest_dir, path) for path in audio_files]

        def on_source(index: int, path: str) -> None:
            printt(f"[{index + 1}/{len(source_paths)}] {Path(path).name}")

        Interrupts().set("concatenating")
        try:
            convert_audio_files(
                source_paths,
                dest_file_path,
                cancel_check=lambda: Interrupts().did_interrupt,
                on_source=on_source,
            )
        except Exception as e:
            return describe_error(e)
        finally:
            Interrupts().clear()
            printt()
        return ""
