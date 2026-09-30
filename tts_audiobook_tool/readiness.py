from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import ReadinessIssue, SttVariant
from tts_audiobook_tool.conversation.chat_config import is_microphone_input_mode
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo

if TYPE_CHECKING:
    from tts_audiobook_tool.state import State
    from tts_audiobook_tool.project import Project


def get_generate_blockers(state: State) -> list[ReadinessIssue]:
    """Returns blocking issues that prevent audiobook generation from starting."""
    items = []

    if not state.project.phrase_groups:
        items.append(
            ReadinessIssue("text", "Text must be imported into the project")
        )

    model_errors = get_tts_blockers(state.project)
    if model_errors:
        items.extend(model_errors)

    return items


def refresh_remote_model_state(project: Project) -> None:
    """Bind the project's model with a forced remote check at run start."""
    from tts_audiobook_tool.tts import Tts

    Tts.bind_project(project, refresh=True)


def get_tts_blockers(project: Project, instance: object | None = None) -> list[ReadinessIssue]:
    """Project-selected model/voice checks plus its cached binding failure."""
    from tts_audiobook_tool.tts import Tts

    issues = Tts.get_model_support(project).get_blocking_issues(project, instance)
    binding_issue = Tts._binding_issue if Tts._bound_project_type_id == project.tts_model_type else None
    if binding_issue is not None and all(issue.verbose != binding_issue.verbose for issue in issues):
        issues.append(binding_issue)
    return issues


def get_generate_blocker_text(state: State, verbose=False) -> str:
    """Blocker text from cached state; safe to build for menu labels."""
    items = get_generate_blockers(state)
    return format_issues(items, verbose)


def get_run_blocker_text(state: State, verbose: bool = True) -> str:
    """Blocker text for a run start, after forcing a remote re-probe.

    Run starts use this instead of ``get_generate_blocker_text`` because a
    stale discovery observation must not decide whether a long run may start.
    Menu labels and headers keep using the cached variant.
    """
    refresh_remote_model_state(state.project)
    return get_generate_blocker_text(state, verbose)


def get_tts_preview_blocker_text(state: State, verbose: bool = False) -> str:
    """Return model/voice blockers for an explicit preview prompt.

    Unlike audiobook generation, a preview supplies its own text and therefore
    does not require imported project phrase groups.
    """
    refresh_remote_model_state(state.project)
    items = get_tts_blockers(state.project)
    return format_issues(items, verbose)


def get_chat_blockers(state: State) -> list[ReadinessIssue]:
    """Returns blocking issues that prevent chat from starting."""
    errors = []

    if is_microphone_input_mode(state.prefs.chat_input_mode):
        # Microphone chat still requires an input device and enabled STT.
        sound_input_error = SoundInputDeviceInfo.get_check_error()
        if sound_input_error:
            errors.append(ReadinessIssue("microphone", sound_input_error))

        if state.prefs.stt_variant == SttVariant.DISABLED:
            errors.append(
                ReadinessIssue(
                    "Whisper model enabled",
                    "Speech-to-text must be enabled (see Options menu > Whisper model)",
                )
            )

    # LLM settings are irrelevant when input is echoed directly to TTS.
    # Rem, api key and model name may or may not be required depending on endpoint
    if (
        not getattr(state.prefs, "chat_echo_override", False)
        and not state.prefs.llm_url.strip()
    ):
        errors.append(
            ReadinessIssue(
                "LLM settings", "LLM endpoint URL not set (see Options > LLM Settings)"
            )
        )

    # TTS Model generation readiness

    model_errors = get_tts_blockers(state.project)
    if model_errors:
        errors.extend(model_errors)

    return errors


def get_chat_blocker_text(state: State, verbose=False) -> str:
    items = get_chat_blockers(state)
    return format_issues(items, verbose)


def format_issues(issues: list[ReadinessIssue], verbose=False) -> str:
    if not issues:
        return ""
    if len(issues) == 1:
        if verbose:
            return issues[0].verbose
        else:
            return f"requires {issues[0].short}"
    else:
        if verbose:
            verbose_items = [item.verbose for item in issues]
            return "\n".join(verbose_items)
        else:
            short_items = [item.short for item in issues]
            return f"requires: {', '.join(short_items)}"
