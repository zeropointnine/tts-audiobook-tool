"""Shared model-worker hard-reset requests and outcomes.

A hard reset always terminates the current model-worker process and attempts to
start a replacement.  The request records *why* that action was initiated;
the outcome records whether replacement startup reported an error.  Keeping
those facts separate avoids inferring control-flow policy from user-facing
message text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tts_audiobook_tool.generation_events import GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.gen_timeout_util import make_gen_timeout_message
from tts_audiobook_tool.model_worker import ModelWorker


class HardResetCause(str, Enum):
    USER_ESCALATION = "user_escalation"
    GENERATION_TIMEOUT = "generation_timeout"
    MODEL_UNHEALTHY = "model_unhealthy"
    INTERFACE_FAILURE = "interface_failure"

    @property
    def should_alert(self) -> bool:
        """Whether generation should play its completion/attention sound."""
        return self in {
            HardResetCause.GENERATION_TIMEOUT,
            HardResetCause.MODEL_UNHEALTHY,
        }


@dataclass(frozen=True)
class HardResetRequest:
    cause: HardResetCause
    reason: str = ""


@dataclass(frozen=True)
class HardResetOutcome:
    request: HardResetRequest
    restart_error: str = ""

    @property
    def message(self) -> str:
        reason = self.request.reason
        if self.restart_error:
            return f"{reason}\n{self.restart_error}" if reason else self.restart_error
        return reason


def hard_reset_request_from_generation_update(
    update: object,
) -> HardResetRequest | None:
    """Translate generation control updates into one reset request shape."""
    if isinstance(update, GenerationTimedOut):
        return HardResetRequest(
            HardResetCause.GENERATION_TIMEOUT,
            make_gen_timeout_message(update.timeout_seconds),
        )
    if isinstance(update, ModelUnhealthy):
        return HardResetRequest(HardResetCause.MODEL_UNHEALTHY, update.reason)
    return None


def perform_hard_reset(request: HardResetRequest) -> HardResetOutcome:
    """Terminate the current worker, attempt replacement, and retain errors."""
    return HardResetOutcome(request, ModelWorker.reset())
