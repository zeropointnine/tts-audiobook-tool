from pathlib import Path
import ctypes
import ctypes.util
import os
import subprocess
import sys
from tts_audiobook_tool import text_util
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *

class FfmpegUtil:

    @staticmethod
    def is_ffmpeg_available() -> bool:
        """ Checks if 'ffmpeg' executable is installed and accessible in the system PATH """
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
    def are_ffmpeg_libraries_available() -> bool:
        """
        Checks if FFmpeg shared libraries are loadable by the current Python process.

        This is separate from `is_ffmpeg_available()` because Python packages such as
        TorchCodec need FFmpeg's shared libraries (`.dll`, `.so`, `.dylib`) at import/runtime,
        not the `ffmpeg` command-line executable.
        """
        library_names = [
            "avcodec",
            "avdevice",
            "avfilter",
            "avformat",
            "avutil",
            "swresample",
            "swscale",
        ]

        return all(FfmpegUtil._can_load_ffmpeg_library(library_name) for library_name in library_names)

    @staticmethod
    def _can_load_ffmpeg_library(library_name: str) -> bool:
        """Checks whether a single FFmpeg shared library can be loaded."""

        if sys.platform == "win32":
            # Python 3.8+ does not reliably use PATH for DLL dependencies of extension modules.
            # Bare-name loading is the closest cheap test for what TorchCodec will be able to do.
            if FfmpegUtil._can_load_shared_library(f"{library_name}.dll"):
                return True

            for library_path in FfmpegUtil._find_files_on_path(f"{library_name}-"):
                # Intentionally load by bare name, not full path. If this fails, packages such
                # as TorchCodec will likely also fail until `os.add_dll_directory()` is called.
                if FfmpegUtil._can_load_shared_library(library_path.name):
                    return True

            return False

        found_library = ctypes.util.find_library(library_name)
        if found_library and FfmpegUtil._can_load_shared_library(found_library):
            return True

        if sys.platform == "darwin":
            library_dirs = os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
            library_dirs += ["/usr/local/lib", "/opt/homebrew/lib", "/opt/local/lib"]
            names = [f"lib{library_name}.dylib"]
            prefixes = [f"lib{library_name}."]
        else:
            library_dirs = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            library_dirs += ["/lib", "/usr/lib", "/usr/local/lib", "/lib64", "/usr/lib64"]
            names = [f"lib{library_name}.so"]
            prefixes = [f"lib{library_name}.so."]

        for library_dir in library_dirs:
            path = Path(library_dir) if library_dir else None
            if not path or not path.is_dir():
                continue

            for name in names:
                if FfmpegUtil._can_load_shared_library(str(path / name)):
                    return True

            for child_path in FfmpegUtil._iter_file_paths(path):
                if any(child_path.name.startswith(prefix) for prefix in prefixes):
                    if FfmpegUtil._can_load_shared_library(str(child_path)):
                        return True

        return False



    @staticmethod
    def _can_load_shared_library(library_path: str) -> bool:
        """Checks if the current Python process can load a shared library."""
        try:
            ctypes.CDLL(library_path)
            return True
        except OSError:
            return False

    @staticmethod
    def attempt_add_ffmpeg_dll_windows() -> bool:
        """
        On Windows, attempts to register an FFmpeg DLL directory with the current Python process.

        Newer Python versions do not reliably use PATH for DLL dependencies loaded by packages
        such as TorchCodec. This looks for an FFmpeg bin directory already present in PATH, calls
        `os.add_dll_directory()` for it, then verifies that the FFmpeg libraries are loadable.

        Returns True if FFmpeg libraries are loadable after the attempt, otherwise False.
        """
        if sys.platform != "win32":
            return False

        if FfmpegUtil.are_ffmpeg_libraries_available():
            return True

        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(path_dir) if path_dir else None
            if not path or not path.is_dir():
                continue

            if not any(child_path.name.startswith("avcodec-") for child_path in FfmpegUtil._iter_file_paths(path)):
                continue

            try:
                os.add_dll_directory(str(path))
            except OSError:
                continue

            # Some libraries still consult PATH directly. Keep it in sync with the DLL directory list.
            if str(path) not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")

            if FfmpegUtil.are_ffmpeg_libraries_available():
                return True

        return False

    @staticmethod
    def _find_files_on_path(file_name_prefix: str) -> list[Path]:
        file_paths = []
        for path_dir in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(path_dir) if path_dir else None
            if not path or not path.is_dir():
                continue
            file_paths += [child_path for child_path in FfmpegUtil._iter_file_paths(path) if child_path.name.startswith(file_name_prefix)]
        return file_paths

    @staticmethod
    def _iter_file_paths(path: Path) -> list[Path]:
        try:
            return [child_path for child_path in path.iterdir() if child_path.is_file()]
        except OSError:
            return []

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

        Returns error string on fail, else empty string
        """

        if use_temp_file:
            dest_file_suffix = Path(dest_file_path).suffix
            temp_file_name = text_util.make_random_hex_string() + dest_file_suffix
            working_dest_file_path = str( Path(dest_file_path).parent / temp_file_name )
        else:
            working_dest_file_path = dest_file_path

        # TODO: check for redundant first and last element (especially last element)
        full_command = partial_command[:]
        full_command.insert(0, FFMPEG_COMMAND)
        full_command.append(working_dest_file_path)

        if False:
            printt(" ".join(full_command))
            printt()

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
                return f"Ffmpeg error, returncode - {completed_process.returncode}"

        except subprocess.CalledProcessError as e:
            if use_temp_file:
                delete_silently(working_dest_file_path)
            return f"Ffmpeg error, returncode - {e.returncode} - {e.stderr}"

        except Exception as e:
            if use_temp_file:
                delete_silently(working_dest_file_path)
            return f"Subprocess error: {e}"

        # Success at this point

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

    SHARED_LIBS_MISSING_MESSAGE = \
"""FFmpeg shared libraries were not found

Please install a full FFmpeg build that includes shared libraries, then make sure those libraries are
discoverable by your system:
- Windows: add the FFmpeg bin directory containing the DLL files to PATH
- Linux: install FFmpeg libraries through your package manager, or set LD_LIBRARY_PATH
- macOS: install FFmpeg with Homebrew or MacPorts, or set DYLD_LIBRARY_PATH

Download/install info:
https://ffmpeg.org/download.html
"""
