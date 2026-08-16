"""
One-time process bootstrap that must run as early as possible,
before any other app imports.

- Sets HF_HUB_DISABLE_SYMLINKS_WARNING to suppress noisy warnings
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

from huggingface_hub import constants # type: ignore

# Guard against the known HF_HUB_CACHE relative-path import-ordering issue
assert PurePath(constants.HF_HUB_CACHE).is_absolute(), \
    f"Unexpected relative HF_HUB_CACHE: {constants.HF_HUB_CACHE}"