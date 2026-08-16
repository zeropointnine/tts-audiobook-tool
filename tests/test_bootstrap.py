import os
from pathlib import PurePath

import pytest

huggingface_hub = pytest.importorskip("huggingface_hub")


def test_bootstrap_sets_env_and_cache_is_absolute():
    """
    Regression test for the import-ordering invariant documented in
    `tts_audiobook_tool/hf_bootstrap.py`: importing it as early as possible
    keeps HF_HUB_CACHE from degrading to a relative path, and it
    suppresses the symlink warning noise.
    """

    # Import here (module level) so the test is meaningful even if other
    # test modules in the same session already pulled in the app
    from tts_audiobook_tool import hf_bootstrap  # pyright: ignore[reportUnusedImport]

    assert hf_bootstrap is not None
    assert os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "true"
    assert PurePath(huggingface_hub.constants.HF_HUB_CACHE).is_absolute()