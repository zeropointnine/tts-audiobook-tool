from typing import Any

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ACCENT, COL_DIM, COL_ERROR, ellipsize_path_for_menu, make_currently_string, print_feedback, truncate_pretty
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

class VoiceQwen3Menu:

    @staticmethod
    def menu(state: State) -> None:
        metadata: dict[str, Any] = {}

        def get_model_type() -> str:
            return str(metadata.get("model_type", state.project.get_model_setting('qwen3tts_local', 'model_type')))

        def get_speakers() -> list[str] | None:
            # Unknown until Set speaker is selected; opening sample management
            # must not load a checkpoint just to discover its speaker inventory.
            value = metadata.get("supported_speakers")
            return [str(speaker) for speaker in value] if isinstance(value, (list, tuple)) else None

        def make_voice_label(_) -> str:
            if not state.project.get_model_setting('qwen3tts_local', 'file_name'):
                unused_checkpoint = {
                    "voice_design": "VoiceDesign",
                    "custom_voice": "CustomVoice",
                }.get(get_model_type())
                if unused_checkpoint:
                    currently = f"{COL_DIM}(not used by {unused_checkpoint})"
                else:
                    currently = make_currently_string("required", value_prefix="", color_code=COL_ERROR)
            else:
                value = ProjectVoiceUtil.get_voice_label(state.project)
                value = ellipsize_path_for_menu(value)
                currently = make_currently_string(value)
            return f"Select voice clone sample {currently}"

        def make_speaker_label(_) -> str:
            speakers = get_speakers()
            if speakers is not None and len(speakers) == 1:
                speaker_id = speakers[0]
            else:
                speaker_id = state.project.get_model_setting('qwen3tts_local', 'speaker_id')
            value = speaker_id or "None"
            suffix = make_currently_string(value)
            if not speaker_id:
                suffix = f"({COL_ERROR}required{COL_DIM})"
            elif speakers is not None and speaker_id not in speakers:
                suffix += f" ({COL_ERROR}required - current id is invalid{COL_DIM})"
            return "Set speaker " + suffix

        def on_speaker(_: State, __: MenuItem) -> None:
            nonlocal metadata
            inspection, error = ModelWorker.inspect_tts_blocking(state)
            if error or inspection is None:
                ask.ask_error(error or "Couldn't inspect Qwen3-TTS model")
                return
            metadata = inspection.metadata or {}
            if get_model_type() != "custom_voice":
                ask.ask_error("Speaker selection requires a CustomVoice checkpoint")
                return
            ask_speaker_id(state.project, get_speakers() or [])

        def make_instructions_label(_) -> str:
            instructions = state.project.get_model_setting('qwen3tts_local', 'instructions')
            if instructions:
                value = truncate_pretty(instructions, 40, content_color=COL_ACCENT)
                suffix = make_currently_string(value)
            elif get_model_type() == "voice_design":
                suffix = make_currently_string("none", color_code=COL_ERROR)
            else:
                suffix = f"{COL_DIM}(optional)"
            return f"Instructions {suffix}"

        def on_clear_instructions(_: State, __: MenuItem) -> None:
            state.project.set_model_setting('qwen3tts_local', 'instructions', "")
            state.project.save()
            print_feedback("Instructions cleared")

        def on_clear_speaker(_: State, __: MenuItem) -> None:
            state.project.set_model_setting('qwen3tts_local', 'speaker_id', "")
            state.project.save()
            print_feedback("Speaker cleared")

        def make_items(_: State) -> list[MenuItem]:
            items = VoiceMenuShared.make_voice_sample_items(
                state,
                TtsModelType.require_by_id("qwen3tts_local"),
                no_samples_label=make_voice_label,
            )
            model_type = get_model_type()
            if model_type == "custom_voice":
                items.append(MenuItem(make_speaker_label, on_speaker, blank_line_before=True))
                if state.project.get_model_setting('qwen3tts_local', 'speaker_id'):
                    items.append(MenuItem("Clear speaker", on_clear_speaker))
            if model_type in ("custom_voice", "voice_design"):
                items.append(MenuItem(
                    make_instructions_label,
                    lambda _, __: ask_instructions(state.project),
                    blank_line_before=model_type == "voice_design",
                ))
                if state.project.get_model_setting('qwen3tts_local', 'instructions'):
                    items.append(MenuItem("Clear instructions", on_clear_instructions))
            return items

        VoiceMenuShared.menu_wrapper(state, make_items)


def ask_speaker_id(project: Project, speakers: list[str]) -> None:
    if len(speakers) == 1:
        message = f"Model has only one speaker id ({speakers[0]})"
        print_feedback(message)
        return

    def validate_speaker_id(value: str) -> str:
        return "" if value in speakers else "Invalid speaker id"

    ask.ask_string_and_save(
        project,
        f"Choose a speaker:\n{speakers}\n",
        SettingRef("qwen3tts_local", "speaker_id"),
        "Set speaker id:",
        validator=validate_speaker_id,
    )


def ask_instructions(project: Project) -> None:
    ask.ask_string_and_save(
        project,
        "Enter instructions prompt:",
        SettingRef("qwen3tts_local", "instructions"),
        "Set instructions:",
    )
