from types import SimpleNamespace
from typing import cast

from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.voice import voice_qwen3_menu
from tts_audiobook_tool.menus.voice.voice_qwen3_menu import VoiceQwen3Menu
from tts_audiobook_tool.model_worker_protocol import TtsInspected
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes


def test_model_selection_refreshes_qwen_menu_model_type(monkeypatch) -> None:
    project = Project.model_validate(
        {
            "qwen3_target": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "qwen3_model_type": "custom_voice",
        }
    )
    state = cast(State, SimpleNamespace(project=project))
    initial_inspection = TtsInspected(
        operation_id="initial",
        tts_type_id="qwen3tts",
        metadata={
            "model_type": "custom_voice",
            "supported_speakers": ["Vivian"],
        },
    )
    rendered_labels: list[list[str]] = []

    def fake_model_target_submenu(state: State, apply_target) -> None:
        apply_target("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    def fake_apply_model_and_validate(
        state: State, target: str, on_applied=None
    ) -> None:
        state.project.qwen3_target = target
        state.project.qwen3_model_type = "base"
        updated_inspection = TtsInspected(
            operation_id="updated",
            tts_type_id="qwen3tts",
            metadata={"model_type": "base"},
        )
        if on_applied is not None:
            on_applied(updated_inspection)

    def fake_menu_wrapper(state: State, make_items) -> None:
        custom_items: list[MenuItem] = make_items(state)
        rendered_labels.append(
            [strip_ansi_codes(get_string_from(state, item.label)) for item in custom_items]
        )
        target_item = next(
            item
            for item in custom_items
            if strip_ansi_codes(get_string_from(state, item.label)).startswith(
                "Select Qwen3-TTS model"
            )
        )
        target_item.handler(state, target_item)

        base_items: list[MenuItem] = make_items(state)
        rendered_labels.append(
            [strip_ansi_codes(get_string_from(state, item.label)) for item in base_items]
        )

    monkeypatch.setattr(
        voice_qwen3_menu, "model_target_submenu", fake_model_target_submenu
    )
    monkeypatch.setattr(
        voice_qwen3_menu, "apply_model_and_validate", fake_apply_model_and_validate
    )
    monkeypatch.setattr(
        voice_qwen3_menu.VoiceMenuShared, "menu_wrapper", fake_menu_wrapper
    )

    VoiceQwen3Menu.menu(state, initial_inspection)

    assert any(label.startswith("Set speaker") for label in rendered_labels[0])
    assert not any(
        label.startswith("Select voice clone sample") for label in rendered_labels[0]
    )
    assert any(
        label.startswith("Select voice clone sample") for label in rendered_labels[1]
    )
    assert not any(label.startswith("Set speaker") for label in rendered_labels[1])
    assert any("model type: base" in label for label in rendered_labels[1])
