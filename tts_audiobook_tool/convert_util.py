from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class ConvertUtil:

    @staticmethod
    def ask_convert(state: State) -> None:

        flac_path = ask("Enter FLAC file path: ") # TODO: "or directory of FLAC files"
        if not flac_path:
            return
        if not flac_path.lower().endswith(".flac"):
            printt("Must have flac file suffix", "error")
            return
        if not os.path.exists(flac_path):
            printt("No such file", "error")
            return
        mp4_path = Path(flac_path).with_stem(".mp4")
        if mp4_path.exists():
            printt("MP4 file already exists with that file stem", "error")
            return

        ConvertUtil.do_convert_aac(state, flac_path)

        ask("Press enter to continue: ")


    @staticmethod
    def do_convert_aac(state: State, flac_path: str) -> None:

        printt(f"Converting {flac_path}")
        printt()

        new_path, err = ConvertUtil.convert_flac_to_aac(flac_path)
        printt()

        if err:
            s = f"{COL_ERROR}{err}"
        else:
            s = f"Saved: {COL_ACCENT}{new_path}"
        printt(s)
        printt()


    @staticmethod
    def convert_flac_to_aac(flac_path: str, kbps=96) -> tuple[str, str]:
        """
        1) Reads the app metadata from flac file
        2) Converts flac to to MP4 using ffmpeg
        3) Adds the app metadata using mutagen

        Returns tuple of output file name and error message, mutually exclusive
        """

        mp4_path = Path(flac_path).with_suffix(".mp4")
        if mp4_path.exists():
            return "", "MP4 file already exists with same file stem"
        mp4_path = str(mp4_path)

        meta_string = AppMetaUtil.get_flac_metadata_field(flac_path, APP_META_FLAC_FIELD)
        if not meta_string:
            return "", "FLAC file has no tts-audiobook-tool metadata"

        partial_command = [
            "-hide_banner", "-loglevel", "warning", "-stats",
            "-i", flac_path,
            "-c:a", "aac",
            "-b:a", f"{kbps}k",
        ]
        err = FfmpegUtil.make_file(partial_command, dest_file_path=mp4_path, use_temp_file=True)
        if err:
            return "", err

        err = AppMetaUtil.set_mp4_metadata_tag(mp4_path, APP_META_MP4_MEAN, APP_META_MP4_TAG, meta_string)
        if err:
            try:
                os.unlink(mp4_path) # even though encoding itself is success
            except:
                pass # meh
            return "", err

        return mp4_path, "" # success
