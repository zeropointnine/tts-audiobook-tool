from pathlib import Path
import subprocess
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class FfmpegUtil:

    @staticmethod
    def is_ffmpeg_available() -> bool:
        """Check if 'ffmpeg' is installed and accessible in the system PATH."""
        try:
            subprocess.run(
                [FFMPEG_COMMAND, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def make_file(
            partial_command: list[str],
            dest_file_path: str,
            use_temp_file: bool,
    ) -> str:
        """
        `partial_command` is expected to be an ffmpeg command _sans_ "ffmpeg" and sans dest path
        (ie, the full command string list w/o the first item and last item).

        When `use_temp_file`, outputs a temp file in same directory as `dest_file_path`,
        and on success, renames temp file to `dest_file_path`.

        Returns error message on fail
        """

        if use_temp_file:
            dest_file_suffix = Path(dest_file_path).suffix
            temp_file_name = make_random_hex_string() + dest_file_suffix
            working_dest_file_path = str( Path(dest_file_path).parent / temp_file_name )
        else:
            working_dest_file_path = dest_file_path

        # TODO: check for redundant first and last element (especially last element)
        full_command = partial_command[:]
        full_command.insert(0, FFMPEG_COMMAND)
        full_command.append(working_dest_file_path)

        try:
            completed_process = subprocess.run(
                full_command,
                check=True,  # Raise CalledProcessError if ffmpeg returns non-zero exit code
                text=True,
                encoding='utf-8'
            )
            if completed_process.returncode != 0:
                if use_temp_file:
                    delete_silently(working_dest_file_path)
                return f"ffmpeg fail, returncode - {completed_process.returncode}"

        except subprocess.CalledProcessError as e:
            if use_temp_file:
                delete_silently(working_dest_file_path)
            return f"ffmpeg fail, returncode - {e.returncode} - {e.stderr}"

        except Exception as e:
            if use_temp_file:
                delete_silently(working_dest_file_path)
            return f"subprocess fail, ffmpeg - {e}"

        # Ffmpeg finished successfully

        if use_temp_file:
            # Rename file
            try:
                if os.path.exists(dest_file_path):
                    # Note, deleting pre-existing file
                    # Note also that this allows the dest path to be the same as the (implicit) src file,
                    # so long as use_temp_file is True.
                    L.w(f"file already exists, will replace: {dest_file_path}")
                    os.unlink(dest_file_path)
                os.rename(working_dest_file_path, dest_file_path)
            except Exception as e:
                # Don't delete temp file in this case
                return f"Couldn't rename temp file: {e}"
            delete_silently(working_dest_file_path)
            return ""

        return ""
