"""
Standalone CLI: extract embedded JSON metadata ("AppMetadata") from an
"abr" audio file (.abr.flac / .abr.m4a / .abr.m4b) to a .json file of
the same stem.

Usage:
    ./venv-base/bin/python -m testx.abr_meta_extract path/to/book.abr.m4b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_audiobook_tool.project_support.project_transfer_util import (
    ProjectTransferUtil,
)

ABR_SUFFIXES = {".flac", ".m4a", ".m4b"}


def make_output_path(abr_path: Path) -> Path:
    """
    "foo.abr.flac" -> "foo.json"; plain "foo.flac" -> "foo.json".
    """
    if abr_path.name.lower().endswith(".abr" + abr_path.suffix.lower()):
        abr_path = abr_path.with_name(abr_path.name[: -len(abr_path.suffix) - len(".abr")])
    return abr_path.with_suffix(".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract embedded JSON metadata from an 'abr' audio file to a .json file of the same stem."
    )
    parser.add_argument("abr_path", help="Path to an .abr.flac / .abr.m4a / .abr.m4b file")
    args = parser.parse_args()

    abr_path = Path(args.abr_path)
    if not abr_path.is_file():
        print(f"Error: file not found: {abr_path}", file=sys.stderr)
        return 1

    suffix = abr_path.suffix.lower()
    if suffix not in ABR_SUFFIXES:
        print(f"Error: unsupported suffix '{suffix}', expected one of {sorted(ABR_SUFFIXES)}", file=sys.stderr)
        return 1

    string = ProjectTransferUtil.load_raw_abr_metadata_string(str(abr_path))
    if not string:
        print(f"Error: no embedded metadata found in {abr_path}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(string)
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    except json.JSONDecodeError as e:
        print(f"Warning: embedded metadata is not valid JSON ({e}); writing raw string.", file=sys.stderr)
        content = string if string.endswith("\n") else string + "\n"

    out_path = make_output_path(abr_path)
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
