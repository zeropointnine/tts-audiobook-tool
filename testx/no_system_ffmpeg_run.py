"""
Run a model venv's Python with system FFmpeg hidden (Linux only).

Proves that something works on a machine *without* system FFmpeg, on a machine
that has it, while reusing the existing venvs, GPU and Hugging Face cache.

How: an unprivileged user + mount namespace (`unshare --user --map-root-user
--mount`) bind-mounts /dev/null over every system FFmpeg executable
(ffmpeg/ffprobe/ffplay) and shared library (libavcodec, libavformat,
libavutil, libavfilter, libavdevice, libswresample, libswscale, libpostproc,
libavresample) found on PATH, the ldconfig cache, LD_LIBRARY_PATH and the
standard library directories. The mounts exist only inside the namespace.
A nested user namespace then maps the user's own uid/gid back, so the command
does not run as (fake) root. Nothing on the host changes.

So inside the runner, `shutil.which("ffmpeg")` is None and loading a system
FFmpeg library fails ("file too short"). FFmpeg bundled in pip wheels (PyAV's
`av.libs`, or eg opencv-python's) is untouched: that is pip-managed.

Before running the command the runner verifies itself in a separate process
of the same venv: no FFmpeg executable on PATH, every hidden library fails to
load, the app's `FfmpegUtil.are_ffmpeg_libraries_available()` is False, and
after `import torchaudio` (and `av`) no FFmpeg library is mapped from outside
the venv. It prints torchaudio's view of the environment (version, audio
backends). The command gets `NO_SYSTEM_FFMPEG_RUNNER=1` in its environment.

Usage (from the repo root, with any Python 3.11+):

    # Verify only
    python3 -m testx.no_system_ffmpeg_run --venv venv-g

    # Run a module / script / the app with the venv's Python
    python3 -m testx.no_system_ffmpeg_run --venv venv-cb -- -m testx.no_ffmpeg_model_clone
    python3 -m testx.no_system_ffmpeg_run --venv venv-q -- -m tts_audiobook_tool

    # Run an arbitrary command (eg a shell to look around)
    python3 -m testx.no_system_ffmpeg_run --venv venv-base --raw -- bash

    # List what would be hidden
    python3 -m testx.no_system_ffmpeg_run --list

Options: --no-verify skips the self-check (not recommended); --verify-only
exits after it.

Requires util-linux `unshare` with `--map-user`/`--map-group` (2.38+) and
unprivileged user namespaces. Stock Ubuntu 24.04+ restricts those through
AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`); the runner then
fails with a permission error. The alternative is a container with the NVIDIA
runtime that mounts the repo, venvs and HF cache.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MARKER_ENV = "NO_SYSTEM_FFMPEG_RUNNER"
# JSON list of the hidden paths, for the inner stage (once hidden, they are no longer regular files)
HIDDEN_ENV = "NO_SYSTEM_FFMPEG_HIDDEN"

EXECUTABLES = ("ffmpeg", "ffprobe", "ffplay")
LIB_FAMILIES = (
    "avcodec", "avformat", "avutil", "avfilter", "avdevice", "avresample",
    "swresample", "swscale", "postproc",
)
_LIB_NAME = re.compile(rf"^lib({'|'.join(LIB_FAMILIES)})\.so(\.\d+)*$")
STANDARD_LIB_DIRS = (
    "/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/local/lib64",
    "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
)


# --- host side: what to hide

def find_system_ffmpeg_files() -> list[str]:
    """Real paths of system FFmpeg executables and shared libraries."""
    found: set[str] = set()

    exe_dirs = os.environ.get("PATH", "").split(os.pathsep) + ["/usr/bin", "/usr/local/bin", "/bin", "/snap/bin"]
    for directory in exe_dirs:
        for name in EXECUTABLES:
            path = os.path.join(directory, name)
            if directory and os.path.isfile(path):
                found.add(os.path.realpath(path))

    lib_dirs = list(STANDARD_LIB_DIRS)
    lib_dirs += [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    lib_dirs += _ldconfig_lib_dirs()
    for directory in dict.fromkeys(lib_dirs):
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            path = os.path.join(directory, name)
            if _LIB_NAME.match(name) and os.path.isfile(path):
                found.add(os.path.realpath(path))

    # Pip wheels' own copies (PyAV's av.libs etc.) are pip-managed, not system
    return sorted(path for path in found if not _is_under(path, REPO_ROOT) and "site-packages" not in path)


def _ldconfig_lib_dirs() -> list[str]:
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    try:
        output = subprocess.run([ldconfig, "-p"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    dirs = []
    for line in output.splitlines():
        if "=>" in line:
            path = line.split("=>", 1)[1].strip()
            if _LIB_NAME.match(os.path.basename(path)):
                dirs.append(os.path.dirname(path))
    return dirs


def _is_under(path: str, root: Path) -> bool:
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def venv_python(venv: str) -> str:
    path = Path(venv)
    if not path.is_absolute():
        path = REPO_ROOT / path
    python = path / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"No Python at {python}")
    # Not resolved: the venv's own bin/python selects the venv
    return str(python)


def run_in_namespace(python: str, hidden: list[str], inner_args: list[str]) -> int:
    """Hide `hidden` in a new mount namespace, then run this module's inner stage."""
    if sys.platform != "linux":
        raise SystemExit("The no-system-FFmpeg runner is Linux only")
    unshare = shutil.which("unshare")
    if not unshare:
        raise SystemExit("util-linux `unshare` not found")

    mounts = "\n".join(f"mount --bind /dev/null {shlex.quote(path)}" for path in hidden)
    inner = [
        unshare, "--user", f"--map-user={os.getuid()}", f"--map-group={os.getgid()}", "--",
        python, "-m", "testx.no_system_ffmpeg_run", "--inner", *inner_args,
    ]
    script = f"set -e\n{mounts}\nexec {shlex.join(inner)}\n"
    import json
    env = dict(os.environ, **{MARKER_ENV: "1", HIDDEN_ENV: json.dumps(hidden)})
    command = [unshare, "--user", "--map-root-user", "--mount", "--", "sh", "-c", script]
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


# --- inside the namespace

VERIFY_SNIPPET = r"""
import ctypes, importlib.util, json, os, shutil, sys
hidden = json.loads(sys.argv[1])
problems = []
for name in ("ffmpeg", "ffprobe", "ffplay"):
    if shutil.which(name):
        problems.append(f"{name} executable still found: {shutil.which(name)}")
for path in hidden:
    if "/lib" in path:
        try:
            ctypes.CDLL(path)
            problems.append(f"hidden library still loads: {path}")
        except OSError:
            pass
from tts_audiobook_tool.sound.ffmpeg_util import FfmpegUtil
if FfmpegUtil.are_ffmpeg_libraries_available():
    problems.append("FfmpegUtil.are_ffmpeg_libraries_available() is True")
print(f"python: {sys.executable} ({sys.version.split()[0]})")
if importlib.util.find_spec("av"):
    import av
    print(f"av {av.__version__} (FFmpeg {av.ffmpeg_version_info})")
if importlib.util.find_spec("torchaudio"):
    import torchaudio
    backends = None
    try:
        backends = torchaudio.list_audio_backends()
    except Exception as e:
        backends = f"unavailable ({type(e).__name__})"
    print(f"torchaudio {torchaudio.__version__}; audio backends: {backends}")
else:
    print("torchaudio: not installed")
from tts_audiobook_tool.system_support.loaded_ffmpeg_libs import ffmpeg_lib_report
report = ffmpeg_lib_report()
prefix = os.path.realpath(sys.prefix)
outside = [lib.path for lib in report.libs if not os.path.realpath(lib.path).startswith(prefix + os.sep)]
if outside:
    problems.append(f"FFmpeg libraries mapped from outside the venv: {outside}")
print(report.describe())
for problem in problems:
    print(f"FAIL: {problem}")
print("runner self-check: " + ("FAIL" if problems else "PASS"))
sys.exit(1 if problems else 0)
"""


def inner_main(args: argparse.Namespace, command: list[str], hidden: list[str]) -> int:
    import json

    if not args.no_verify:
        print("--- no-system-FFmpeg runner self-check", flush=True)
        result = subprocess.run([sys.executable, "-c", VERIFY_SNIPPET, json.dumps(hidden)], cwd=REPO_ROOT)
        print("---", flush=True)
        if result.returncode != 0:
            return result.returncode
    if args.verify_only:
        return 0
    if not command:
        print("No command given (see --help); self-check only")
        return 0
    argv = command if args.raw else [sys.executable, *command]
    os.execvp(argv[0], argv)


def main() -> int:
    argv = sys.argv[1:]
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", default="venv-base", help="venv dir (relative to the repo root or absolute)")
    parser.add_argument("--raw", action="store_true", help="the command is a full command, not venv Python args")
    parser.add_argument("--no-verify", action="store_true", help="skip the runner self-check")
    parser.add_argument("--verify-only", action="store_true", help="run the self-check, then exit")
    parser.add_argument("--list", action="store_true", help="list the files that would be hidden, then exit")
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.inner:
        import json
        return inner_main(args, command, json.loads(os.environ.get(HIDDEN_ENV, "[]")))

    hidden = find_system_ffmpeg_files()

    if args.list:
        print("\n".join(hidden) if hidden else "(no system FFmpeg found)")
        return 0
    if not hidden:
        print("Note: no system FFmpeg found; nothing to hide")
    python = venv_python(args.venv)
    print(f"Hiding {len(hidden)} system FFmpeg files; venv: {args.venv}", flush=True)
    inner_args = [flag for flag, on in (("--raw", args.raw), ("--no-verify", args.no_verify),
                                        ("--verify-only", args.verify_only or not command)) if on]
    return run_in_namespace(python, hidden, inner_args + (["--", *command] if command else []))


if __name__ == "__main__":
    sys.exit(main())
