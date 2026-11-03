"""
One-time process bootstrap that must run as early as possible,
before any other app imports.

- Sets HF_HUB_DISABLE_SYMLINKS_WARNING to suppress noisy warnings
- Sets FOR_DISABLE_CONSOLE_CTRL_HANDLER (Windows): the Intel Fortran runtime
  that some numerical libraries load installs its own console control-c
  handler, which aborts the process ("forrtl: error (200): program aborting
  due to control-C event") regardless of Python's SIGINT handling. In the
  model worker that abort could stall halfway, leaving a worker that never
  answers (blank menu, hang on quit). The runtime reads this variable when
  its DLL loads, so it is set before any other import; spawned workers
  inherit it.
- Imports `huggingface_hub.constants` as early as possible, or else
  HF_HUB_CACHE can end up returning a relative path due to an unknown
  import side-effect (possibly from a specific model library, not sure).
  This is asserted below so that a regression in import ordering
  surfaces immediately at startup instead of at model-download time.

Keep this module free of all app imports so it can be imported first
from any entrypoint without pulling in the dependency cascade.
"""

import os
from pathlib import PurePath

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

from huggingface_hub import constants # type: ignore

# Guard against the known HF_HUB_CACHE relative-path import-ordering issue
assert PurePath(constants.HF_HUB_CACHE).is_absolute(), \
    f"Unexpected relative HF_HUB_CACHE: {constants.HF_HUB_CACHE}"