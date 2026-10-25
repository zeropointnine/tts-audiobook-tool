from __future__ import annotations

import ntpath

from tts_audiobook_tool import app_support, ask, text_util
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_support.remote_tts_discovery import RemoteTtsDiscovery
from tts_audiobook_tool.app_types import Hint
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.model_support import max_words_exceeds_recommended
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import ModelStateSnapshot
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.state import PendingBindingNotice, PendingTtsModelChange, State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class MenuStatus:
    """
    Prints "menu status block" at top of console when in "clear screen mode"
    """

    @staticmethod
    def prepare_tts(state: State) -> None:
        """Reconcile before building menu items or displaying project metadata."""
        from tts_audiobook_tool.tts import Tts, TtsSelectionChangeReason

        previous_id = state.project.tts_model_type
        previous_server_id = state.project.audio_cpp_model_id
        # Captured first: reconciling to a new audio.cpp entry overwrites it.
        last_audio_cpp = (state.project.last_audio_cpp_model_type, state.project.last_audio_cpp_model_id)
        change = Tts.reconcile_project_selection(state.project)
        changed = change.model if change is not None else None
        if changed is not None and state.project.dir_path:
            error = state.project.save()
            if error:
                ask.ask_error(error)
        MenuStatus._queue_binding_notice(state, Tts.bind_project(state.project))
        local_startup_finished = (
            not Tts.is_remote_mode() and getattr(state, "has_shown_main_menu", False)
            and not getattr(state, "pending_project_load_checks", False)
        )
        if local_startup_finished or (
            changed is not None and changed.id == "none" and not Tts.is_remote_mode()
        ):
            state.pending_tts_model_change = None
        elif change is not None and changed is not None:
            pending = getattr(state, "pending_tts_model_change", None)
            coalesce = pending is not None and pending.new_model_id == previous_id
            old_id = pending.old_model_id if pending is not None and coalesce else previous_id
            old_server_id = pending.old_audio_cpp_model_id if pending is not None and coalesce else previous_server_id
            new_server_id = state.project.audio_cpp_model_id
            remembered = change.reason in (TtsSelectionChangeReason.RESTORED_AUDIO_CPP,
                                           TtsSelectionChangeReason.LAST_AUDIO_CPP_UNAVAILABLE)
            entry_replaced = change.reason is TtsSelectionChangeReason.SAVED_ENTRY_UNAVAILABLE
            if remembered:
                # Restoring, or failing to restore, the remembered audio.cpp
                # entry is always worth a notice, even from an unselected project.
                announce = old_id != changed.id
            elif entry_replaced:
                # The saved exact entry is gone; even a same-type entry is a change.
                announce = (old_id, old_server_id) != (changed.id, new_server_id)
            else:
                # Clearing selection only announces an unavailable audio.cpp model.
                announce = old_id not in ("none", changed.id) and (
                    changed.id != "none"
                    or TtsModelType.get_by_id(old_id).value.backend_kind is TtsBackendKind.AUDIO_CPP
                )
            state.pending_tts_model_change = (
                PendingTtsModelChange(
                    old_id, changed.id, change.reason.value,
                    *(last_audio_cpp if remembered else ("", "")),
                    # Exact entries are named only when the entry itself changed.
                    old_audio_cpp_model_id=old_server_id if entry_replaced else "",
                    new_audio_cpp_model_id=new_server_id if entry_replaced else "",
                )
                if announce else None
            )

    @staticmethod
    def _queue_binding_notice(state: State, issue: object) -> None:
        """Queue a one-time FYI when a server-entry binding issue first appears.

        Shown once per (selection, server URL, issue); it re-arms when the
        issue clears or a project is loaded. The status row keeps only a terse
        qualifier while the issue persists.
        """
        from tts_audiobook_tool.tts import Tts, TtsBindingIssue

        if not isinstance(issue, TtsBindingIssue) or not Tts.is_remote_mode():
            state.binding_notice_key = None
            state.pending_binding_notice = None
            return
        project = state.project
        key = (project.tts_model_type, project.audio_cpp_model_id, RemoteTtsDiscovery.get_base_url(), issue.verbose)
        if key == getattr(state, "binding_notice_key", None):
            return
        state.binding_notice_key = key
        state.pending_binding_notice = PendingBindingNotice(
            project.tts_model_type, project.audio_cpp_model_id, issue.kind, issue.verbose, issue.server_ids)

    @staticmethod
    def show_pending_binding_hint(state: State) -> None:
        pending = getattr(state, "pending_binding_notice", None)
        if pending is None:
            return
        state.pending_binding_notice = None
        if (state.project.tts_model_type, state.project.audio_cpp_model_id) != (
                pending.model_id, pending.audio_cpp_model_id):
            return
        name = _make_model_name(pending.model_id, pending.audio_cpp_model_id)
        if pending.kind == "ambiguous_entry":
            text = (f"Several server entries match this project's TTS model, {name}:\n"
                    f"{', '.join(pending.server_ids)}.\nSelect one in Model settings > TTS model.")
        elif pending.kind in ("entry_unavailable", "no_match"):
            text = (f"This project's TTS model, {name}, is not available on this server.\n"
                    "Select a TTS model in Model settings > TTS model.")
        else:
            text = pending.verbose
        hints.print_hint(Hint("", "FYI", text))

    @staticmethod
    def show_pending_project_hints(state: State, *, is_first_main_menu: bool = False) -> None:
        """Run shared project-load checks once, after rendering a complete menu."""
        MenuStatus.show_pending_tts_model_hint(state, is_first_main_menu=is_first_main_menu)
        MenuStatus.show_pending_binding_hint(state)
        if not getattr(state, "pending_project_load_checks", False):
            return
        if not is_first_main_menu and not getattr(state, "has_shown_main_menu", False):
            return
        state.pending_project_load_checks = False

        from tts_audiobook_tool.tts import Tts

        model = state.project.get_tts_model_type()
        max_duration = model.value.ui.get("voice_sample_max_duration_s")
        if max_duration is not None:
            count = 0
            entries = state.project.voice_references
            for index, voice in enumerate(ProjectVoiceUtil.get_voice_values(state.project, model)):
                if not voice:
                    continue
                # An active crop is what generation uses, so its (shorter)
                # duration is what this hint should measure.
                entry = entries[index] if index < len(entries) else Voice(voice)
                path = ProjectVoiceUtil.effective_voice_file_path(state.project, entry)
                duration = AudioMetaUtil.get_audio_duration(path)
                if duration is not None and duration > max_duration:
                    count += 1
            if count:
                samples = "is 1 sample" if count == 1 else f"are {count} samples"
                verb = "exceeds" if count == 1 else "exceed"
                hints.print_hint(Hint("", "FYI", (
                    f"The current model's recommended duration for voice clone samples is {max_duration:g}s,\n"
                    f"but there {samples} for this project that {verb} that value."
                )))

        if model.id == "none" or not state.project.phrase_groups:
            return
        limit = Tts.get_model_support(state.project).get_max_words_range_reco(state.project)[1]
        max_words = state.project.book.segmentation_settings.max_words_per_segment
        if max_words_exceeds_recommended(max_words, limit):
            hints.print_hint(Hint("", "FYI", (
                f"This project's text was segmented with a maximum of {max_words} words per segment,\n"
                f"exceeding the current model's recommended maximum of {limit}.\n"
                "Output accuracy on longer prompts may be degraded."
            )))

    @staticmethod
    def show_pending_tts_model_hint(state: State, *, is_first_main_menu: bool = False) -> None:
        """Consume a deferred notice at startup or when another project is loaded."""
        from tts_audiobook_tool.tts import Tts

        pending = getattr(state, "pending_tts_model_change", None)
        if pending is None:
            return
        if state.project.tts_model_type != pending.new_model_id:
            state.pending_tts_model_change = None
            return
        runtime_project_load = (
            getattr(state, "pending_project_load_checks", False)
            and getattr(state, "has_shown_main_menu", False)
        )
        if not Tts.is_remote_mode() and not is_first_main_menu and not runtime_project_load:
            if getattr(state, "has_shown_main_menu", False):
                state.pending_tts_model_change = None
            return
        state.pending_tts_model_change = None

        old_name = _make_model_name(pending.old_model_id, pending.old_audio_cpp_model_id)
        if pending.reason in ("restored_audio_cpp", "last_audio_cpp_unavailable"):
            hints.print_hint(Hint("", "FYI", _make_remembered_audio_cpp_notice(pending, old_name)))
            return
        if pending.new_model_id == "none":
            hints.print_hint(Hint("", "FYI", (
                f"This project was last used with TTS model {old_name},\nwhich is currently unavailable.\n"
                "Select a TTS model in Model settings > TTS model."
            )))
            return
        current_name = _make_model_name(pending.new_model_id, pending.new_audio_cpp_model_id)
        text = f"This project was last used with TTS model {old_name}.\n"
        if Tts.get_active_type().id != pending.new_model_id:
            text += f"It has been switched to {current_name}, {_UNUSABLE_MODEL_TEXT}"
        elif Tts.is_remote_mode():
            # The server offers exactly one model (an SGL-Omni server, or audio.cpp with one entry).
            text += _make_server_model_text(current_name)
        else:
            # Local mode has exactly one installed model library.
            text += f"This environment has {current_name} installed, so it will use that."
        hints.print_hint(Hint("", "FYI", text))

    @staticmethod
    def print_block(state: State) -> None:
        from tts_audiobook_tool.tts import Tts

        MenuStatus.prepare_tts(state)
        lines = []
        worker_models, _ = ModelWorker.get_model_state_blocking()

        project_text = _make_project_text(state)
        lines.append(("Project", project_text))

        text_text = _make_text_text(state)
        lines.append(("Text", text_text))

        if Tts.is_remote_mode():
            server_tts_text = _make_server_tts_text(state)
            lines.append(("TTS model", server_tts_text))
        else:
            local_tts_text = _make_local_tts_text(state, worker_models)
            lines.append(("TTS model", local_tts_text))

        lines.append(("Voices", _make_voices_text(state)))

        stt_text = _make_stt_text(state, worker_models)
        lines.append(("STT model", stt_text))

        memory_text = _make_memory_text()
        if memory_text:
            lines.append(("Memory", memory_text))

        label_len = 0
        for label, _ in lines:
            label_len = max(label_len, len(label))

        for label, value in lines:
            label = (label + ":").ljust(label_len + 1)
            s = f"{LABEL_COLOR}{label} {VALUE_COLOR}{value}"
            printt(s)


# Display label per backend kind, matching the status block's "TTS model" line.
_BACKEND_KIND_LABELS: dict[TtsBackendKind | None, str] = {
    TtsBackendKind.LOCAL: "local",
    TtsBackendKind.SGL_OMNI: "SGL-Omni",
    TtsBackendKind.AUDIO_CPP: "audio.cpp",
}


def binding_issue_status(issue: object) -> str:
    """Terse pinned qualifier for a binding issue; full text for other issues."""
    return getattr(issue, "status", "") or getattr(issue, "verbose", "")


def _make_model_name(raw_id: str, server_id: str = "") -> str:
    """Eg: "Chatterbox TTS (local)"; the backend qualifies identically-named models.

    With `server_id`: "Fish Audio Drama 3 (audio.cpp; fish-drama-q8)".
    """
    model = TtsModelType.get_by_id(raw_id)
    if model.id == "none":
        return "None (unselected)" if raw_id == model.id else f"Unknown model: {raw_id}"
    proper_name = model.value.ui["proper_name"]
    qualifiers = [label for label in (_BACKEND_KIND_LABELS.get(model.value.backend_kind), server_id) if label]
    return f"{proper_name} ({'; '.join(qualifiers)})" if qualifiers else proper_name


def _make_remembered_audio_cpp_notice(pending: PendingTtsModelChange, old_name: str) -> str:
    """Notice text for restoring (or failing to restore) the remembered audio.cpp entry."""
    from tts_audiobook_tool.tts import Tts

    remembered = _make_model_name(pending.last_audio_cpp_model_type, pending.last_audio_cpp_model_id)
    has_old = pending.old_model_id != "none"
    is_bound = Tts.get_active_type().id == pending.new_model_id
    # Without the "last used" line there is no antecedent for "It"/"Its".
    text = f"This project was last used with TTS model {old_name}.\n" if has_old else ""
    if pending.reason == "restored_audio_cpp":
        subject = "It" if has_old else "This project"
        if is_bound:
            return text + f"{subject} will switch back to its last-used audio.cpp model, {remembered}."
        return text + (f"{subject} has been switched back to its last-used audio.cpp model, {remembered},\n"
                       + _UNUSABLE_MODEL_TEXT)
    subject = "Its" if has_old else "This project's"
    if pending.new_model_id == "none":
        return text + (f"{subject} last-used audio.cpp model, {remembered},\nis currently unavailable.\n"
                       "Select a TTS model in Model settings > TTS model.")
    current_name = _make_model_name(pending.new_model_id)
    text += f"{subject} last-used audio.cpp model, {remembered}, is not on this server.\n"
    if is_bound:
        return text + _make_server_model_text(current_name)
    return text + f"This project has been switched to {current_name}, {_UNUSABLE_MODEL_TEXT}"


def _make_server_model_text(current_name: str) -> str:
    """Explains choosing the server's sole model."""
    return f"The server is serving {current_name}, so it will use that."


# Completes a "switched to" sentence for a selection that failed to bind.
_UNUSABLE_MODEL_TEXT = "but that model can't be used yet (see Model settings)."


def _make_project_text(state: State) -> str:
    if state.project.dir_path:
        text = text_util.make_terminal_hyperlink(state.project.dir_path, is_file=True)
    else:
        text = COL_ERROR + "required"

    return text

def _make_local_tts_text(
    state: State,
    worker_models: ModelStateSnapshot | None,
) -> str:
    """Eg: Some TTS (special sauce: True) (cuda, loaded)."""

    from tts_audiobook_tool.tts import Tts

    text = Tts.get_model_support(state.project).get_menu_text(state.project, None)
    if text == NONE_MODEL_NAME:
        text = COL_ERROR + text
    text += f" {QUALIFIER_COLOR}(local)"

    extras = []
    tts_loaded = bool(
        worker_models and worker_models.tts_loaded
        and worker_models.tts_type_id == state.project.get_tts_model_type().id
    )
    if tts_loaded and worker_models and worker_models.tts_device:
        extras.append(worker_models.tts_device)
    if tts_loaded:
        extras.append("loaded")
    if not tts_loaded and state.prefs.tts_force_cpu:
        # Note, showing "force cpu" qualifier only if no instance exists
        extras.append("will force cpu")

    if extras:
        text += f" {QUALIFIER_COLOR}({', '.join(extras)})"
    return text

def _make_server_tts_text(state: State) -> str:
    from tts_audiobook_tool.tts import Tts

    snapshot = RemoteTtsDiscovery.get_snapshot()
    selected = state.project.get_tts_model_type()
    # The qualifier identifies the saved model, not the connected server.
    # Only an unselected/unknown model falls back to the discovered backend.
    backend_kind = (selected.value.backend_kind if selected.id != "none"
                    else snapshot.backend_kind)
    backend = _BACKEND_KIND_LABELS.get(backend_kind, "server")
    base_url = RemoteTtsDiscovery.get_base_url()
    server_matches_backend = snapshot.backend_kind in (None, backend_kind)
    if backend_kind is TtsBackendKind.AUDIO_CPP and base_url and server_matches_backend:
        backend = text_util.make_terminal_hyperlink(
            f"{base_url}/v1/models?include_session_options=true", backend
        )
    elif backend_kind is TtsBackendKind.SGL_OMNI and base_url and server_matches_backend:
        backend = text_util.make_terminal_hyperlink(f"{base_url}/v1/models", backend)
    model = selected.value.ui["proper_name"]
    if selected.id == "none":
        model = (COL_ERROR + NONE_MODEL_NAME if state.project.tts_model_type == selected.id else
                 f"Unknown model: {state.project.tts_model_type}")
    server_unreachable = snapshot.issue is not None and snapshot.issue.code in {"unavailable", "timeout"}
    qualifiers = [] if backend == "server" and server_unreachable else [backend]
    # Use server residency from cached metadata, not the worker's adapter state.
    if (
        snapshot.backend_kind is TtsBackendKind.AUDIO_CPP
        and backend_kind is TtsBackendKind.AUDIO_CPP
        and snapshot.issue is None
        and Tts._binding_issue is None
        and selected.id != "none"
        and Tts.get_active_type() == selected
        and Tts._selected_server_model_id
    ):
        loaded_entry = next((entry for entry in snapshot.models
                             if entry.get("id") == Tts._selected_server_model_id
                             and entry.get("loaded") is True), None)
        if loaded_entry is not None:
            qualifiers.append("loaded")
            path = loaded_entry.get("path")
            if isinstance(path, str):
                # Server path style is independent of the client's OS.
                filename = ntpath.basename(path.strip())
                if filename and filename not in (".", ".."):
                    qualifiers.append(ellipsize(filename, 30))
    text = model
    if qualifiers:
        text += f" {QUALIFIER_COLOR}({', '.join(qualifiers)})"
    if snapshot.issue is not None:
        if server_unreachable:
            text += f" {COL_ERROR}(server unreachable)"
        elif snapshot.issue.code == "no_supported_models":
            server_backend = _BACKEND_KIND_LABELS.get(snapshot.backend_kind, "server")
            text += f" {COL_ERROR}(server mode; {server_backend} has no supported models)"
        else:
            text += f" {COL_ERROR}({snapshot.issue.message})"
    elif selected.id != "none" and backend_kind is TtsBackendKind.LOCAL:
        text += f" {COL_ERROR}(unavailable in server mode)"
    elif selected.id != "none" and Tts._binding_issue is not None:
        # Terse pinned qualifier; the full explanation is a one-time FYI.
        text += f" {COL_ERROR}({binding_issue_status(Tts._binding_issue)})"
    return text

def _make_voices_text(state: State) -> str:
    """The first voice that has a value (preset, sample, or instructions; see
    `ProjectVoiceUtil.resolve_voice_source`), then "+N more" counting every
    other voice. With no such voice, error-colored "required" when the model
    requires a value, else (non-blocking, so not error-colored) what the voices
    lack, eg "no sample set"; see `ProjectVoiceUtil.make_no_voice_value_text`.
    For VibeVoice with a LoRA set, ", +LoRA" is appended last, or with no
    voice, just "LoRA"."""
    project = state.project
    has_lora = (
        project.tts_model_type == "vibevoice_local"
        and bool(project.get_model_setting('vibevoice_local', 'lora_target'))
    )
    lora_suffix = ", +LoRA" if has_lora else ""
    voices = project.voice_references
    sources = [ProjectVoiceUtil.resolve_voice_source(project, voice) for voice in voices]
    first = next((source for source in sources if source.has_value), None)
    if first is None:
        missing = any(source.kind == "missing" for source in sources)
        if has_lora and not missing:
            # The LoRA supplies a fixed speaker, so no voice is not a problem
            return "LoRA"
        if missing:
            text = COL_ERROR + "required"
            if lora_suffix:
                text += VALUE_COLOR + lora_suffix
            return text
        return ProjectVoiceUtil.make_no_voice_value_text(project)
    value = first.label
    if len(voices) > 1:
        value += f", +{len(voices) - 1} more"
    return value + lora_suffix


def _make_text_text(state: State) -> str:
    total_lines = len(state.project.phrase_groups)
    if total_lines == 0:
        return COL_ERROR + "required"
    num_generated = state.project.sound_segments.num_generated()
    text = f"{total_lines} lines"
    language_code = state.project.book.segmentation_settings.language_code.strip()
    if language_code:
        text += f", {language_code}"
    generated_text = f"{num_generated} generated"
    if state.project.sound_segments_path:
        generated_text = text_util.make_terminal_hyperlink(
            state.project.sound_segments_path, generated_text, is_file=True
        )
    text += f" {COL_DIM}({generated_text})"
    return text

def _make_stt_text(
    state: State,
    worker_models: ModelStateSnapshot | None = None,
) -> str:
    from tts_audiobook_tool.stt import Stt
    from tts_audiobook_tool.app_types import SttVariant

    text = "mlx-whisper" if Stt.should_use_mlx_whisper() else "faster-whisper"

    qualifiers = []
    if not Stt.should_use_mlx_whisper():
        qualifiers.append(Stt.get_variant().id) # eg, "large-v3"
        if Stt.get_variant() != SttVariant.DISABLED:
            qualifiers.append(state.prefs.stt_config.device) # eg, "cuda"
    if worker_models and worker_models.stt_loaded:
        qualifiers.append("loaded")
    if qualifiers:
        s2 = ", ".join(qualifiers)
        text += f" {COL_DIM}({s2})"

    return text

def _make_memory_text() -> str:
    text = text_util.strip_ansi_codes(app_support.make_memory_string())
    text = text.replace(":", "") # careful
    return text if text else ""

# ---

LABEL_COLOR = COL_DIM
VALUE_COLOR = COL_MEDIUM
QUALIFIER_COLOR = COL_DIM

# Displayed name of the "no TTS model selected" placeholder
NONE_MODEL_NAME = TtsModelType.require_by_id("none").value.ui["proper_name"]
