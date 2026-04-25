#!/usr/bin/env python3
"""
Launcher script for tts-audiobook-tool.

Scans the project root directory for virtual environments, detects which
TTS model libraries each venv has installed, and lets the user pick one
to launch the app with.

Duplicates some app code and logic to avoid importing cascade of app dependencies.
"""

import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# ANSI helpers lifted from tts_audiobook_tool/ansi.py (truecolor + xterm-256
# fallback) and color constants from tts_audiobook_tool/constants.py.
# ---------------------------------------------------------------------------
def rgb_to_xterm256(r: int, g: int, b: int) -> int:
    """Convert RGB to nearest xterm 256 color index."""
    if abs(r - g) < 8 and abs(g - b) < 8:
        gray = round((r + g + b) / 3)
        if gray < 8:
            return 16
        if gray > 248:
            return 231
        gray_index = round(((gray - 8) / 240) * 23)
        return 232 + gray_index
    r_idx = round((r / 255) * 5)
    g_idx = round((g / 255) * 5)
    b_idx = round((b / 255) * 5)
    return 16 + (r_idx * 36) + (g_idx * 6) + b_idx


class Ansi:
    RESET = "\033[0m"
    CLEAR_SCREEN_AND_SCROLLBACK = "\033[2J\033[3J\033[H"

    @staticmethod
    def hex(hex_color: str) -> str:
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
        hex_color = hex_color.ljust(6, "0")[:6]
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except ValueError:
            r = g = b = 255

        colorterm = os.environ.get("COLORTERM", "").lower()
        if colorterm in ("truecolor", "24bit"):
            return f"\033[38;2;{r};{g};{b}m"
        else:
            idx = rgb_to_xterm256(r, g, b)
            return f"\033[38;5;{idx}m"


# Color constants matching tts_audiobook_tool/constants.py
COL_ACCENT = Ansi.hex("ffaa44")
COL_DIM = Ansi.hex("888888")
COL_DEFAULT = Ansi.RESET
COL_ERROR = Ansi.hex("ff0000")
COL_OK = Ansi.hex("00ff00")
COL_INPUT = Ansi.hex("aaaaaa")


# ---------------------------------------------------------------------------
# Import TtsModelInfos directly — its deps are all stdlib (enum, functools,
# typing), so this is safe without installing the app package.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tts_audiobook_tool.tts_model.tts_model_info import TtsModelInfos

# Build the list of (module_test, proper_name) from the enum, skipping NONE
QUALIFIED_MODELS: list[tuple[str, str]] = []
for member in TtsModelInfos:
    if member.name == "NONE":
        continue
    QUALIFIED_MODELS.append((member.value.module_test, member.value.ui["proper_name"]))


def find_venvs(base_dir: str) -> list[str]:
    """Return paths to subdirectories that contain a pyvenv.cfg file."""
    venvs: list[str] = []
    if not os.path.isdir(base_dir):
        return venvs
    for entry in os.listdir(base_dir):
        child = os.path.join(base_dir, entry)
        if os.path.isdir(child) and os.path.isfile(os.path.join(child, "pyvenv.cfg")):
            venvs.append(child)
    return sorted(venvs)


def get_venv_python(venv_path: str) -> str | None:
    """Return the path to the Python executable inside the venv."""
    if sys.platform == "win32":
        candidate = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        candidate = os.path.join(venv_path, "bin", "python")
    return candidate if os.path.isfile(candidate) else None


def probe_venv(venv_path: str) -> tuple[list[str], int]:
    """
    Inside *venv_path*, run a subprocess that checks each module_test.

    Returns (list_of_proper_names, match_count) where match_count is the
    number of matched models *after* Fish S2 dedup (matching tts.py's
    init_model_type logic).  If match_count > 1 the venv is ambiguous
    and should not be used.
    """
    python_exe = get_venv_python(venv_path)
    if python_exe is None:
        return [], 0

    tests_json = json.dumps(QUALIFIED_MODELS)

    probe_code = (
        "import importlib.util\n"
        "import json\n"
        "tests = " + tests_json + "\n"
        "def check(mod):\n"
        "    try:\n"
        "        return importlib.util.find_spec(mod) is not None\n"
        "    except ModuleNotFoundError:\n"
        "        return False\n"
        "matched = [i for i, (mod, _) in enumerate(tests) if check(mod)]\n"
        "infos = [tests[i] for i in matched]\n"
        "mod_list = [m for m, _ in infos]\n"
        "if 'fish_speech' in mod_list and 'fish_speech.callbacks' in mod_list:\n"
        "    infos = [(m, n) for m, n in infos if m != 'fish_speech']\n"
        "names = [n for _, n in infos]\n"
        "match_count = len(infos)\n"
        "print(json.dumps({'names': names, 'match_count': match_count}))\n"
    )

    try:
        result = subprocess.run(
            [python_exe, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parsed = json.loads(result.stdout.strip())
            names = parsed.get("names", [])
            count = parsed.get("match_count", len(names))
            return names, count
        return [], 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError):
        return [], 0


def build_venv_list(base_dir: str) -> list[tuple[str, str, list[str]]]:
    """
    Returns list of (venv_path, display_name, detected_models).
    Only includes venvs that have exactly one detected model (after Fish S2
    dedup), mirroring tts.py's init_model_type assertion that exactly 0 or 1
    model should match.  Ambiguous venvs (>1 match) are printed as warnings
    and skipped.
    """
    results: list[tuple[str, str, list[str]]] = []
    for vp in find_venvs(base_dir):
        name = os.path.basename(vp)
        models, match_count = probe_venv(vp)
        if match_count == 0:
            continue
        if match_count > 1:
            print(
                f"  {COL_DIM}Warning: {name} matched {match_count} models"
                f" ({' / '.join(models)}){COL_DEFAULT}"
                f"\u2014skipping (ambiguous environment)",
                file=sys.stderr,
            )
            continue
        # match_count == 1
        results.append((vp, name, models))
    return results


def make_bracket(num: int, width: int) -> str:
    """Return e.g. '[ 9]' or '[10]' with proper alignment."""
    return f"[{COL_ACCENT}{num:>{width}}{Ansi.RESET}]"


def show_menu(venvs: list[tuple[str, str, list[str]]]) -> int:
    """Print a colored numbered menu and return the chosen index (0-based)."""
    # Determine padding width for alignment when >= 10 items
    if len(venvs) >= 10:
        width = len(str(len(venvs)))
    else:
        width = 1

    print()
    heading = "tts-audiobook-tool - Virtual environment convenience launcher"
    print(f"{COL_DIM}{'-' * len(heading)}{Ansi.RESET}")
    print(f"{COL_ACCENT}{heading}{Ansi.RESET}")
    print(f"{COL_DIM}{'-' * len(heading)}{Ansi.RESET}")
    print()

    for i, (_, name, models) in enumerate(venvs, start=1):
        if len(models) == 1:
            models_str = models[0]
        else:
            models_str = ", ".join(models)
        print(f"  {make_bracket(i, width)} {name}  {COL_DIM}({models_str}){Ansi.RESET}")

    print(f"  {make_quit_bracket(width)} {COL_DIM}Quit{Ansi.RESET}")
    print()

    while True:
        print(f"{COL_INPUT}Selection", end="")
        print(f": {Ansi.RESET}", end="")
        choice = input().strip().lower()
        if choice == "q":
            sys.exit(0)
        try:
            idx = int(choice)
            if 1 <= idx <= len(venvs):
                return idx - 1
        except ValueError:
            pass
        print(f"{COL_ERROR}Invalid choice.{Ansi.RESET} Enter a number between 1 and {len(venvs)}, or 'q'.")
        print()


def make_quit_bracket(width: int) -> str:
    """Return a quit bracket that matches width of the number brackets."""
    return f"[{COL_ACCENT}{'q':>{width}}{Ansi.RESET}]"


def main() -> None:
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    elif env_dir := os.environ.get("TTS_LAUNCH_BASE_DIR"):
        base_dir = env_dir
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    venvs = build_venv_list(base_dir)

    if not venvs:
        print()
        heading = "  tts-audiobook-tool \u2014 No qualified virtual environments found"
        print(f"{COL_DIM}{'-' * (len(heading) + 2)}{Ansi.RESET}")
        print(f"{COL_DEFAULT}{heading}{Ansi.RESET}")
        print(f"{COL_DIM}{'-' * (len(heading) + 2)}{Ansi.RESET}")
        print()
        print(f"  {COL_DIM}Searched in: {base_dir}{Ansi.RESET}")
        print()
        print(f"  {COL_DIM}A qualified venv is a subdirectory containing pyvenv.cfg and")
        print(f"  at least one supported TTS model library installed.{Ansi.RESET}")
        print()
        sys.exit(1)

    choice_idx = show_menu(venvs)
    venv_path = venvs[choice_idx][0]
    python_exe = get_venv_python(venv_path)
    if python_exe is None:
        print(f"\n  {COL_ERROR}Error:{Ansi.RESET} Could not find Python executable in {venv_path}")
        sys.exit(1)

    # Change to the project root so `python -m tts_audiobook_tool` resolves the
    # package regardless of the caller's current working directory.
    os.chdir(SCRIPT_DIR)

    args = [python_exe, "-m", "tts_audiobook_tool"]
    if sys.platform == "win32":
        subprocess.call(args)
        sys.exit()
    os.execv(python_exe, args)


if __name__ == "__main__":
    main()