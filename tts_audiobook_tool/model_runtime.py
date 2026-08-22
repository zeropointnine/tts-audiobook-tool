from __future__ import annotations

from enum import Enum


class ModelRuntimeRole(str, Enum):
    STANDALONE = "standalone"
    INTERACTIVE_MAIN = "interactive-main"
    MODEL_WORKER = "model-worker"


_role = ModelRuntimeRole.STANDALONE


def mark_standalone() -> None:
    global _role
    _role = ModelRuntimeRole.STANDALONE


def mark_interactive_main() -> None:
    global _role
    _role = ModelRuntimeRole.INTERACTIVE_MAIN


def mark_model_worker() -> None:
    global _role
    _role = ModelRuntimeRole.MODEL_WORKER


def current_role() -> ModelRuntimeRole:
    return _role


def require_model_owner(model_name: str) -> None:
    """Reject heavyweight model construction in the interactive UI process."""
    if _role is ModelRuntimeRole.INTERACTIVE_MAIN:
        raise RuntimeError(
            f"{model_name} may only be loaded through ModelWorker in interactive app mode"
        )
