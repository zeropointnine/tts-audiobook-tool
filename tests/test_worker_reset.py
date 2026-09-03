from tts_audiobook_tool.generation_events import GenerationTimedOut, ModelUnhealthy
from tts_audiobook_tool.worker_reset import (
    HardResetCause,
    HardResetRequest,
    hard_reset_request_from_generation_update,
    perform_hard_reset,
)


def test_generation_reset_updates_map_to_explicit_causes() -> None:
    timeout = hard_reset_request_from_generation_update(GenerationTimedOut(180.0))
    unhealthy = hard_reset_request_from_generation_update(ModelUnhealthy("bad model"))

    assert timeout is not None
    assert timeout.cause is HardResetCause.GENERATION_TIMEOUT
    assert "GEN_TIMEOUT" in timeout.reason
    assert unhealthy == HardResetRequest(HardResetCause.MODEL_UNHEALTHY, "bad model")
    assert hard_reset_request_from_generation_update(object()) is None


def test_alert_policy_depends_on_cause_not_reason_text() -> None:
    assert HardResetCause.GENERATION_TIMEOUT.should_alert
    assert HardResetCause.MODEL_UNHEALTHY.should_alert
    assert not HardResetCause.USER_ESCALATION.should_alert
    assert not HardResetCause.INTERFACE_FAILURE.should_alert

    assert HardResetRequest(HardResetCause.MODEL_UNHEALTHY, "").cause.should_alert
    assert not HardResetRequest(
        HardResetCause.USER_ESCALATION,
        "user supplied an explanatory reason",
    ).cause.should_alert


def test_perform_hard_reset_preserves_reason_and_restart_error(monkeypatch) -> None:
    reset_calls: list[None] = []
    monkeypatch.setattr(
        "tts_audiobook_tool.worker_reset.ModelWorker.reset",
        lambda: reset_calls.append(None) or "replacement failed",
    )
    request = HardResetRequest(HardResetCause.INTERFACE_FAILURE, "interface failed")

    outcome = perform_hard_reset(request)

    assert reset_calls == [None]
    assert outcome.request is request
    assert outcome.restart_error == "replacement failed"
    assert outcome.message == "interface failed\nreplacement failed"
