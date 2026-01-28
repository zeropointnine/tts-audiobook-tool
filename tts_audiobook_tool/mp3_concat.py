import os

import natsort
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class SoundConcatTranscodeUtil:

    @staticmethod
    def ask_and_concat_audio_files():

        # TODO: need pseudo-menu

        src_dir = AskUtil.ask_dir_path("Enter directory of audio files:", "Select directory with audio files")

        if not src_dir:
            return

        if not os.path.exists(src_dir):
            AskUtil.ask_enter_to_continue("No such directory")
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
            hotkey = AskUtil.ask_hotkey(s)
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
        file_name = "concatenated" + ".flac" if is_flac else ".m4a"
        
        err = SoundConcatTranscodeUtil.concatenate_audio_files(
            audio_files, src_dir, file_name, is_flac
        )
        if err:
            AskUtil.ask_error(err)
            return

        printt("Finished")
        printt()
        s = f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open directory: "
        hotkey = AskUtil.ask_hotkey(s)
        if hotkey == "o":
            err = open_directory_in_gui(src_dir)
            if err:
                AskUtil.ask_error(err)

    @staticmethod
    def concatenate_audio_files(
            audio_files: list[str], dest_dir: str, dest_filename: str, is_flac: bool
    ) -> str:
        """
        Returns error string on fail, else empty string
        """

        if not audio_files:
            return "No files specified"

        os.makedirs(dest_dir, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, dest_filename)

        # Create a temporary text file with file list for ffmpeg.
        # This is the safest method, handling special characters in paths.
        temp_text_file_path = os.path.join(dest_dir, "temp_filelist.txt")
        try:
            with open(temp_text_file_path, 'w', encoding='utf-8') as f:
                for filepath in audio_files:
                    # The 'file' directive requires proper quoting for special chars
                    s = filepath.replace("'", "'\\''")
                    f.write(f"file '{s}'\n")
        except Exception as e:
            delete_silently(temp_text_file_path)
            return make_error_string(e)

        # Make ffmpeg command
        partial_command = FFMPEG_TYPICAL_OPTIONS[:]

        partial_command.extend([
            '-f', 'concat',             # Use the concat demuxer
            '-safe', '0',               # Allow unsafe filenames (needed for this method)
            '-i', temp_text_file_path,  # Input file list
        ])

        rest = FFMPEG_ARGUMENTS_OUTPUT_FLAC[:] if is_flac else FFMPEG_ARGUMENTS_OUTPUT_AAC[:]
        partial_command.extend(rest)

        err = FfmpegUtil.make_file(partial_command, dest_file_path, False)

        delete_silently(temp_text_file_path)

        return err