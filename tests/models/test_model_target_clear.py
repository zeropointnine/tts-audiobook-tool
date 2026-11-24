from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model import (
    model_moss_menu,
    model_omnivoice_menu,
    model_qwen3_menu,
    model_vibevoice_menu,
)
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel


@pytest.mark.parametrize("module, menu_class, model_id", [
    (model_omnivoice_menu, model_omnivoice_menu.ModelOmniVoiceMenu, "omnivoice_local"),
    (model_qwen3_menu, model_qwen3_menu.ModelQwen3Menu, "qwen3tts_local"),
    (model_moss_menu, model_moss_menu.ModelMossMenu, "moss_local"),
    (model_vibevoice_menu, model_vibevoice_menu.ModelVibeVoiceMenu, "vibevoice_local"),
])
def test_model_target_menu_routes_clear_to_existing_reset_semantics(
        monkeypatch, module, menu_class, model_id):
    # Each model has its own handler: exercise the real menu-to-prompt callback
    # chain, while shared prompt key handling is covered by its owner tests.
    project = Project(tts_model_type=model_id)
    state = cast(State, SimpleNamespace(project=project))
    target = ModelTarget.hf("example/custom-model")
    project.set_model_target(model_id, target)
    seed_name = "delay_seed" if model_id == "moss_local" else "seed"
    project.set_model_setting(model_id, seed_name, 123)
    if model_id == "qwen3tts_local":
        project.set_model_setting(model_id, "model_type", "custom_voice")
    events = []
    captured = {}
    monkeypatch.setattr(Project, "save", lambda self: events.append("save"))
    monkeypatch.setattr(ModelWorker, "clear_models_if_running_blocking",
                        lambda: events.append("unload"))
    monkeypatch.setattr(Tts, "set_model_params_using_project",
                        lambda current: events.append(("sync", current)))
    monkeypatch.setattr(module, "print_feedback", lambda *args: events.append("feedback"))
    monkeypatch.setattr(ModelMenuShared, "menu_wrapper",
                        lambda state, factory: captured.update(factory=factory))
    monkeypatch.setattr(ModelMenuShared, "target_submenu",
                        lambda **kwargs: kwargs["ask_custom_target"]())
    monkeypatch.setattr(ModelMenuShared, "ask_target",
                        lambda **kwargs: captured.update(prompt=kwargs))

    if model_id == "qwen3tts_local":
        inspection = SimpleNamespace(metadata={
            "model_type": "custom_voice", "generate_defaults": {"temperature": 0.123},
        })
        menu_class.menu(state, inspection)
    else:
        menu_class.menu(state)

    items = captured["factory"](state)
    assert not any(get_string_from(state, item.label) == "Clear custom model" for item in items)
    items[0].handler(state, items[0])
    assert captured["prompt"]["current_target"] == target
    captured["prompt"]["on_clear"]()

    assert project.get_model_target(model_id) is None
    assert project.get_model_setting(model_id, seed_name) == 123
    expected_events = ["save"]
    if model_id == "omnivoice_local":
        expected_events.append(("sync", project))
    expected_events += ["unload", "feedback"]
    assert events == expected_events

    if model_id == "qwen3tts_local":
        assert project.get_model_setting(model_id, "model_type") == ""
        defaults = []
        original = ModelMenuShared.make_temperature_item

        def capture_temperature(**kwargs):
            defaults.append(kwargs["default_value"])
            return original(**kwargs)

        monkeypatch.setattr(ModelMenuShared, "make_temperature_item", capture_temperature)
        refreshed_items = captured["factory"](state)
        assert "base" in get_string_from(state, refreshed_items[0].label)
        assert defaults == [Qwen3BaseModel.TEMPERATURE_FALLBACK_DEFAULT]


def test_qwen_target_prompt_without_menu_reuses_clear_handler(monkeypatch):
    # The standalone entry point must clear persisted type even without a menu
    # metadata-refresh closure to supply its callback.
    project = Project(tts_model_type="qwen3tts_local")
    state = cast(State, SimpleNamespace(project=project))
    project.set_model_target("qwen3tts_local", ModelTarget.hf("example/custom-model"))
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    monkeypatch.setattr(Project, "save", lambda self: None)
    monkeypatch.setattr(ModelWorker, "clear_models_if_running_blocking", lambda: None)
    monkeypatch.setattr(model_qwen3_menu, "print_feedback", lambda *args: None)
    monkeypatch.setattr(ModelMenuShared, "ask_target", lambda **kwargs: kwargs["on_clear"]())

    model_qwen3_menu.ask_target(state)

    assert project.get_model_target("qwen3tts_local") is None
    assert project.get_model_setting("qwen3tts_local", "model_type") == ""
