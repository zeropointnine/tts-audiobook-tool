"""
System FFmpeg shared-library discovery, for local models that declare
`requires_ffmpeg_libs` (see `Start.exit_on_missing_ffmpeg_libs`).

The app itself needs no system FFmpeg: its audio I/O uses PyAV's and
soundfile's bundled libraries (see sound/audio_writers.py, audio_decoder.py).

No model declares the exception at present. Windows DLL-directory registration
(`os.add_dll_directory()` for an FFmpeg bin folder on PATH, which Python 3.8+
extension modules such as TorchCodec need) was removed with the last
declaring model; a future Windows exception has to add it back.
"""

from __future__ import annotations

from pathlib import Path
import ctypes
import ctypes.util
import os
import re
import sys
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants import APP_URL

if TYPE_CHECKING:
    from tts_audiobook_tool.tts_models.model_spec import TtsModelSpec

class FfmpegUtil:

    @staticmethod
    def are_ffmpeg_libraries_available() -> bool:
        """
        Checks if FFmpeg shared libraries are loadable by the current Python process.

        Only for local models that declare `requires_ffmpeg_libs`: packages such as
        TorchCodec need system FFmpeg shared libraries (`.dll`, `.so`, `.dylib`) at
        import/runtime. The app's own audio I/O never uses them (it uses PyAV's
        bundled FFmpeg).
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
    def make_missing_libs_message(spec: TtsModelSpec, platform: str | None = None) -> str:
        """
        Startup message for a model declaring `requires_ffmpeg_libs` whose
        libraries were not found: names the model, says why it needs them
        (catalog `ui.ffmpeg_libs_reason`), gives the step for this OS and
        points to the model's README install section
        (`ui.readme_install_section`), which has the steps for every OS.
        """
        platform = platform or sys.platform
        name = spec.ui.get("proper_name") or spec.id
        reason = spec.ui.get("ffmpeg_libs_reason", "")
        section = spec.ui.get("readme_install_section", "")
        if platform == "win32":
            step = (
                "Download a \"full-shared\" FFmpeg build and add its bin folder "
                "(the one containing avcodec-*.dll) to PATH."
            )
        elif platform == "darwin":
            step = "Run `brew install ffmpeg`, or point DYLD_LIBRARY_PATH at a shared build's lib folder."
        else:
            step = (
                "Install FFmpeg through your package manager (e.g. `sudo apt install ffmpeg`), "
                "or point LD_LIBRARY_PATH at a shared build's lib folder."
            )
        lines = [f"{name} needs FFmpeg shared libraries, which were not found.", ""]
        if reason:
            lines.append(f"Reason: {reason}.")
        lines.append(f"The rest of the app does not need FFmpeg; only {name} does.")
        lines += ["", step]
        if section:
            lines += [
                "",
                f"See the README section \"{section}\":",
                f"{APP_URL}#{github_heading_anchor(section)}",
            ]
        return "\n".join(lines)


def github_heading_anchor(heading: str) -> str:
    """GitHub's anchor for a Markdown heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")
