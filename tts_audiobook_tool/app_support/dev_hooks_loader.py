"""
Loads optional, developer-local hooks from `tts_audiobook_tool/dev_hooks.py`.

That file is gitignored and only loaded when DEV is enabled. Hook points in the app
call `call("<function_name>", ...)`; if the local module defines that function, it runs.
"""

from types import ModuleType
from typing import Any

from tts_audiobook_tool.constants_config import DEV

_MODULE_NAME = "tts_audiobook_tool.dev_hooks"

_module: ModuleType | None = None


def load() -> None:
    global _module
    if not DEV:
        return
    try:
        import importlib
        _module = importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as e:
        if e.name != _MODULE_NAME:
            raise # Error inside dev_hooks.py itself should surface
        return
    print("### DEV_HOOKS LOADED ###")


def call(name: str, **kwargs: Any) -> None:
    """
    Runs the named hook if defined. Exceptions are printed but swallowed so that
    a broken local hook can't take down the app.
    """
    if _module is None:
        return
    fn = getattr(_module, name, None)
    if not callable(fn):
        return
    try:
        fn(**kwargs)
    except Exception:
        import traceback
        print(f"### dev_hooks.{name}() raised: ###")
        traceback.print_exc()
