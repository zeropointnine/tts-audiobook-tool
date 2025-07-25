import os

import natsort
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

class Mp3ConcatTranscodeUtil:

    @staticmethod
    def ask_mp3_dir():

        dir = ask_dir_path("Enter directory of MP3 files:", "Select directory with MP3 files")

        if not os.path.exists(dir):
            ask_continue("No such directory")
            return

        dir = os.path.abspath(dir)

        mp3_files = [f for f in os.listdir(dir) if f.lower().endswith('.mp3')]
        mp3_files = natsort.natsorted(mp3_files, alg=natsort.ns.IGNORECASE)

        if not mp3_files:
            ask_continue("No mp3 files found in directory")

        printt("Will concatenate and transcode the mp3 files in the following order:")
        printt()
        for i, item in enumerate(mp3_files):
            printt(f"[{i+1}] {item}")
        printt()
        b = ask_confirm()
        if not b:
            return

        printt(f"Concatenating and transcoding mp3 files...")
        printt()

        # m4a file is saved in same dir as mp3 files
        err = Mp3ConcatTranscodeUtil.concatenate_mp3s(mp3_files, dir, "transcoded.m4a")
        if err:
            ask_error(err)
            return

        printt("Finished")
        printt()
        s = f"Press {make_hotkey_string("Enter")}, or press {make_hotkey_string("O")} to open directory: "
        hotkey = ask_hotkey(s)
        if hotkey == "o":
            err = open_directory_in_gui(dir)
            if err:
                ask_error(err)

    @staticmethod
    def concatenate_mp3s(mp3_files: list[str], destination_dir: str, destination_filename: str) -> str:
        """
        Returns error string on fail, else empty string
        """

        if not mp3_files:
            return "No mp3 files specified"

        os.makedirs(destination_dir, exist_ok=True)
        output_file_path = os.path.join(destination_dir, destination_filename)

        # Create a temporary text file listing the mp3s for ffmpeg
        # This is the safest method, handling special characters in paths.
        temp_text_file_path = os.path.join(destination_dir, "temp_filelist.txt")
        try:
            with open(temp_text_file_path, 'w', encoding='utf-8') as f:
                for filepath in mp3_files:
                    # The 'file' directive requires proper quoting for special chars
                    f.write(f"file '{filepath.replace("'", "'\\''")}'\n")
        except Exception as e:
            delete_silently(temp_text_file_path)
            return str(e)

        # Make ffmpeg command
        partial_command = FFMPEG_TYPICAL_OPTIONS[:]

        partial_command.extend([
            '-f', 'concat',             # Use the concat demuxer
            '-safe', '0',               # Allow unsafe filenames (needed for this method)
            '-i', temp_text_file_path,  # Input file list
        ])

        partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_AAC[:])

        err = FfmpegUtil.make_file(partial_command, output_file_path, False)

        delete_silently(temp_text_file_path)

        return err