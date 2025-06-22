from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class TranscodeUtil:
    """
    Transcodes final FLAC file to AAC/MP4, and transferring over app metadata
    """

    @staticmethod
    def ask_transcode(state: State) -> None:

        path = ask("Enter FLAC file path or directory of FLAC files: ")
        if not path:
            return
        if not os.path.exists(path):
            ask("No such file or directory. Press enter: ")
            return

        if os.path.isfile(path):
            if os.path.isfile(path):
                if not path.lower().endswith(".flac"):
                    ask("Must have \".flac\" file suffix. Press enter: ")
                    return
                m4a_path = Path(path).with_stem(".m4a")
                if m4a_path.exists():
                    ask("M4A file already exists with that file stem. Press enter: ")
                    return
                dir_flac_paths = [path]
        else:
            dir_flac_paths = []
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if file_path.lower().endswith(".flac"):
                    dir_flac_paths.append(file_path)
            if not dir_flac_paths:
                ask("No FLAC files in directory. Press enter: ")
                return

            warnings = []
            flac_paths = []
            for dir_flac_path in dir_flac_paths:
                m4a_path = Path(dir_flac_path).with_suffix(".m4a")
                if m4a_path.exists():
                    warnings.append(f"M4A file already exists for {Path(dir_flac_path).name}")
                else:
                    meta_string = AudioMetaUtil.get_flac_metadata_field(dir_flac_path, APP_META_FLAC_FIELD)
                    if not meta_string:
                        warnings.append(f"Not a tts-audiobook-tool FLAC file: {Path(dir_flac_path).name}")
                    else:
                        flac_paths.append(dir_flac_path)

            if warnings:
                printt("\n".join(warnings))
                printt()

            if not flac_paths:
                ask("No files to transcode. Press enter: ")
                return

            printt("Will transcode the following files:")
            printt("\n".join(flac_paths))
            printt()
            if not ask_confirm():
                return

            for flac_path in flac_paths:
                TranscodeUtil.do_transcode_aac(state, flac_path)

        ask("Press enter to continue: ")


    @staticmethod
    def do_transcode_aac(state: State, flac_path: str) -> None:

        printt(f"Transcoding to AAC/M4A: {flac_path}")
        printt()

        new_path, err = TranscodeUtil.transcode_abr_flac_to_aac(flac_path)
        printt()

        if err:
            s = f"{COL_ERROR}{err}"
        else:
            s = f"Saved: {COL_ACCENT}{new_path}"
        printt(s)
        printt()


    @staticmethod
    def transcode_abr_flac_to_aac(src_path: str, kbps=96) -> tuple[str, str]:
        """
        1) Reads the app metadata from flac file
        2) Converts flac to to MP4 using ffmpeg
        3) Adds the app metadata using mutagen

        Returns tuple of output file path and error message, mutually exclusive
        """

        m4a_path = Path(src_path).with_suffix(".m4a")
        if m4a_path.exists():
            return "", "M4A file already exists with same file stem"
        m4a_path = str(m4a_path)

        meta_string = AudioMetaUtil.get_flac_metadata_field(src_path, APP_META_FLAC_FIELD)
        if not meta_string:
            return "", "FLAC file has no tts-audiobook-tool metadata"

        partial_command = [
            "-hide_banner", "-loglevel", "warning", "-stats",
            "-i", src_path,
            "-c:a", "aac",
            "-b:a", f"{kbps}k"
        ]
        err = FfmpegUtil.make_file(partial_command, dest_file_path=m4a_path, use_temp_file=True)
        if err:
            return "", err

        err = AudioMetaUtil.set_mp4_metadata_tag(m4a_path, APP_META_MP4_MEAN, APP_META_MP4_TAG, meta_string)
        if err:
            try:
                os.unlink(m4a_path) # even though encoding itself is success
            except:
                pass # meh
            return "", err

        return m4a_path, "" # success
