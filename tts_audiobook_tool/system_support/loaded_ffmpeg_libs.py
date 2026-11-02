"""
Reports which FFmpeg shared libraries a process has loaded, and from where.

Used to check that FFmpeg comes only from PyAV's bundled-library directory
(pip-managed) and that a process does not mix FFmpeg builds — e.g. PyAV's
bundled copy plus system `libav*` pulled in by torchaudio or TorchCodec.

Built on `psutil.Process().memory_maps()`, which is available on Linux and
Windows but not macOS.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field

import psutil

FFMPEG_LIB_FAMILIES = (
    "avcodec", "avformat", "avutil", "avfilter", "avdevice", "avresample",
    "swresample", "swscale", "postproc",
)

_FAMILIES = "|".join(FFMPEG_LIB_FAMILIES)

# Linux: system `libavcodec.so.61.19.101`; auditwheel `libavcodec-5e6b81c2.so.62.28.102`.
# macOS (for completeness; memory_maps is unavailable there): `libavcodec.62.dylib`.
_UNIX_LIB = re.compile(rf"^lib({_FAMILIES})(?:-[0-9a-f]+)?\.(?:so(?:\.|$)|(?:\d+\.)*dylib$)")
# Windows: `avcodec-62.dll`; delvewheel-mangled `avcodec-62-<hash>.dll`.
_WINDOWS_LIB = re.compile(rf"^({_FAMILIES})-\d+(?:-[0-9a-f]+)?\.dll$", re.IGNORECASE)


@dataclass(frozen=True)
class LoadedFfmpegLib:
    family: str
    path: str

    @property
    def directory(self) -> str:
        return _norm(os.path.dirname(self.path))


@dataclass
class FfmpegLibReport:
    libs: list[LoadedFfmpegLib]
    bundled_dirs: list[str] = field(default_factory=list)

    @property
    def directories(self) -> list[str]:
        """Distinct directories FFmpeg libraries were loaded from."""
        return sorted({lib.directory for lib in self.libs})

    @property
    def outside_bundled(self) -> list[LoadedFfmpegLib]:
        """Loaded FFmpeg libraries that are not inside PyAV's bundled directory."""
        bundled = {_norm(directory) for directory in self.bundled_dirs}
        return [lib for lib in self.libs if lib.directory not in bundled]

    @property
    def single_build(self) -> bool:
        """At most one FFmpeg build: one directory, one file per library family."""
        if len(self.directories) > 1:
            return False
        families = [lib.family for lib in self.libs]
        return len(families) == len(set(families))

    @property
    def only_bundled(self) -> bool:
        """Every loaded FFmpeg library resolves inside PyAV's bundled directory."""
        return not self.outside_bundled

    def describe(self) -> str:
        lines = [f"PyAV bundled dirs: {self.bundled_dirs or '(none found)'}"]
        if not self.libs:
            lines.append("No FFmpeg libraries loaded")
        for lib in self.libs:
            lines.append(f"  {lib.family:<11} {lib.path}")
        lines.append(f"single build: {self.single_build}; only bundled: {self.only_bundled}")
        return "\n".join(lines)


def ffmpeg_lib_family(path: str) -> str | None:
    """The FFmpeg library family ("avcodec", ...) for a shared-library path, else None."""
    name = re.split(r"[\\/]", path)[-1]
    match = _WINDOWS_LIB.match(name) or _UNIX_LIB.match(name)
    return match.group(1).lower() if match else None


def pyav_bundled_lib_dirs() -> list[str]:
    """Directories where an installed PyAV wheel keeps its bundled FFmpeg.

    Located without importing `av`, so the check itself loads no FFmpeg.
    auditwheel (Linux) and delvewheel (Windows) use `site-packages/av.libs`;
    delocate (macOS) uses `av/.dylibs`.
    """
    spec = importlib.util.find_spec("av")
    if spec is None or not spec.origin:
        return []
    package_dir = os.path.dirname(spec.origin)
    candidates = [
        os.path.join(os.path.dirname(package_dir), "av.libs"),
        os.path.join(package_dir, ".dylibs"),
    ]
    return [_norm(candidate) for candidate in candidates if os.path.isdir(candidate)]


def libs_from_paths(paths) -> list[LoadedFfmpegLib]:
    """FFmpeg libraries among mapped-file paths, deduplicated and sorted."""
    found = {
        (family, path)
        for path in paths
        if path and (family := ffmpeg_lib_family(path)) is not None
    }
    return [LoadedFfmpegLib(family, path) for family, path in sorted(found)]


def loaded_ffmpeg_libs(pid: int | None = None) -> list[LoadedFfmpegLib]:
    """FFmpeg libraries mapped into process `pid` (default: this process)."""
    process = psutil.Process(pid)
    if not hasattr(process, "memory_maps"):
        raise NotImplementedError(f"psutil memory_maps() is unavailable on {sys.platform}")
    return libs_from_paths(mapping.path for mapping in process.memory_maps(grouped=True))


def ffmpeg_lib_report(pid: int | None = None) -> FfmpegLibReport:
    return FfmpegLibReport(loaded_ffmpeg_libs(pid), pyav_bundled_lib_dirs())


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))
