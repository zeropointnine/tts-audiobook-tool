import types
from types import SimpleNamespace
from typing import cast

import pytest

import tts_audiobook_tool.tts as tts_module
from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
from tts_audiobook_tool.menus.model.model_dots_menu import ModelDotsMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY, SettingRef
from project_settings_test_support import get_setting, set_setting
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.dots_base_model import (
    DotsBaseModel,
    DotsCompileMode,
)
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType


def test_dots_spec_and_tts_registry(monkeypatch):
    info = TtsModelType.require_by_id("dots_local").value

    assert info.backend_kind is TtsBackendKind.LOCAL
    assert info.local_module_test == "dots_tts"
    assert info.local_torch_devices == [DeviceType.CUDA, DeviceType.CPU]
    assert info.default_output_sample_rate == 48_000
    assert DotsBaseModel.get_output_sample_rate(Project.model_validate({})) == 48_000
    assert REGISTRY.voice_binding(info.id).name == "file_name"
    assert REGISTRY.transcript_binding(info.id).name == "transcript"
    assert not info.requires_voice
    assert info.can_stream
    assert info.requirements_file_name == "requirements-dots.txt"

    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id("dots_local"))
    assert Tts.get_class_for_type(TtsModelType.require_by_id("dots_local")) is DotsBaseModel
    assert Tts._MODEL_REGISTRY[TtsModelType.require_by_id("dots_local")][0] is DotsBaseModel
    assert Tts._MODEL_REGISTRY[TtsModelType.require_by_id("dots_local")][2] == "_dots"


def test_local_model_probe_detects_dots(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: types.ModuleType(name) if name == "dots_tts" else None,
    )

    tts_type, matches = Tts.init_local_model_type()

    assert tts_type is TtsModelType.require_by_id("dots_local")
    assert matches == 1


def test_dots_target_change_invalidates_worker_model(monkeypatch):
    clears: list[bool] = []
    monkeypatch.setattr(Tts, "_model_params", {"dots_target": ""})
    monkeypatch.setattr(tts_module, "current_role", lambda: object())
    monkeypatch.setattr(Tts, "clear_tts_model", staticmethod(lambda: clears.append(True)))

    Tts.set_model_params({"dots_target": DotsBaseModel.MF_REPO_ID})

    assert clears == [True]


def test_dots_project_fields_normalize_and_serialize():
    warnings: list[str] = []
    project = Project.model_validate(
        {
            "dots_target": DotsBaseModel.MF_REPO_ID,
            "dots_voice_file_name": "voice.flac",
            "dots_voice_transcript": ["reference words", None],
            "dots_seed": 12.0,
            "dots_speaker_scale": 2,
            "dots_num_steps_soar": 6.0,
            "dots_num_steps_mf": 3.0,
            "dots_guidance_scale": 1.4,
            "dots_compile": False,
        },
        context={"warnings": warnings},
    )

    assert get_setting(project, "dots_target") == DotsBaseModel.MF_REPO_ID
    assert get_setting(project, "dots_voice_file_name") == ["voice.flac"]
    assert get_setting(project, "dots_voice_transcript") == ["reference words"]
    assert get_setting(project, "dots_seed") == 12
    assert get_setting(project, "dots_speaker_scale") == 2.0
    assert get_setting(project, "dots_num_steps_soar") == 6
    assert get_setting(project, "dots_num_steps_mf") == 3
    assert get_setting(project, "dots_guidance_scale") == 1.4
    assert get_setting(project, "dots_compile") is False
    payload = ProjectSerializationUtil.to_project_json_dict(project)
    dots_object = payload["model_settings"]["models"]["dots_local"]
    parameters = dots_object["parameters"]
    assert parameters["target"] == DotsBaseModel.MF_REPO_ID
    # V4: one project-wide list, no scoped clone storage.
    assert payload["voice_references"] == [
        {"file_name": "voice.flac", "transcript": "reference words", "room_reverb": "off"}
    ]
    assert "voice_references" not in dots_object
    assert parameters["seed"] == 12
    assert parameters["speaker_scale"] == 2.0
    assert parameters["num_steps_soar"] == 6
    assert parameters["num_steps_mf"] == 3
    assert parameters["guidance_scale"] == 1.4
    assert parameters["compile"] is False
    assert not any("dots_" in warning for warning in warnings)


def test_dots_project_invalid_values_fall_back_with_warnings():
    warnings: list[str] = []
    project = Project.model_validate(
        {
            "dots_target": "custom/not-supported",
            "dots_seed": "bad",
            "dots_speaker_scale": 4.0,
            "dots_num_steps_soar": 0,
            "dots_num_steps_mf": 999,
            "dots_guidance_scale": 0.5,
            "dots_compile": "yes",
        },
        context={"warnings": warnings},
    )

    assert get_setting(project, "dots_target") == ""
    assert get_setting(project, "dots_seed") == DotsBaseModel.SEED_DEFAULT
    assert get_setting(project, "dots_speaker_scale") == -1
    assert get_setting(project, "dots_num_steps_soar") == -1
    assert get_setting(project, "dots_num_steps_mf") == -1
    assert get_setting(project, "dots_guidance_scale") == -1
    assert get_setting(project, "dots_compile") is DotsCompileMode.default().enabled
    assert all(
        any(field in warning for warning in warnings)
        for field in (
            "dots_target",
            "dots_seed",
            "dots_speaker_scale",
            "dots_num_steps_soar",
            "dots_num_steps_mf",
            "dots_guidance_scale",
            "dots_compile",
        )
    )


def test_voice_menu_routes_dots_to_shared_default_voice_menu(monkeypatch):
    # dots.tts adds no voice behavior, so it needs no voice menu module.
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="dots_local")))
    calls: list[object] = []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", staticmethod(lambda _state, items: calls.append(items)))

    VoiceMenuShared.menu(state)

    assert calls == [VoiceMenuShared.make_local_items]


def test_voice_menu_has_no_default_for_the_none_placeholder():
    # The shared default is for local models only; `none` has no voices to manage.
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="none")))
    with pytest.raises(NotImplementedError):
        VoiceMenuShared.menu(state)


def test_dots_menu_sampling_controls_visibility_by_target(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project()))
    captured: list[list[str]] = []

    def capture_wrapper(actual_state, item_maker, subheading=None):
        items = item_maker(actual_state)
        captured.append(
            [
                strip_ansi_codes(get_string_from(actual_state, item.label))
                for item in items
            ]
        )

    monkeypatch.setattr(ModelMenuShared, "menu_wrapper", staticmethod(capture_wrapper))

    ModelDotsMenu.menu(state)  # SOAR default
    set_setting(state.project, "dots_target", DotsBaseModel.MF_REPO_ID)
    ModelDotsMenu.menu(state)
    set_setting(state.project, "dots_target", DotsBaseModel.MF_2STEPS_REPO_ID)
    ModelDotsMenu.menu(state)

    soar, mf, mf_2steps = captured

    # SOAR: full flow-matching; both NFE and CFG are live knobs
    assert any(label.startswith("Select dots.tts model") for label in soar)
    assert any(label.startswith("Compile") for label in soar)
    assert any(label.startswith("Steps (soar)") for label in soar)
    assert any(label.startswith("Speaker scale") for label in soar)
    assert any(label.startswith("CFG") for label in soar)
    assert any(label.startswith("Seed") for label in soar)

    # Meanflow: NFE is live, but CFG is distilled into the model and the
    # solver drops the parameter, so the item must not be offered
    assert any(label.startswith("Steps (mf)") for label in mf)
    assert not any(label.startswith("CFG") for label in mf)

    # Fixed-step artifacts lock the whole sampling contract
    assert not any(label.startswith("Steps (") for label in mf_2steps)
    assert not any(label.startswith("CFG") for label in mf_2steps)


def test_dots_menu_num_steps_branches_by_variant(monkeypatch):
    # Numeric items come from the catalog parameters; the menu only picks which.
    from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu

    state = cast(State, SimpleNamespace(project=Project()))
    number_items: list[tuple[str, str]] = []

    monkeypatch.setattr(
        ModelMenuShared,
        "menu_wrapper",
        staticmethod(lambda actual_state, item_maker, subheading=None: item_maker(actual_state)),
    )

    def fake_item(_state, model_id, name, label):
        assert model_id == "dots_local"
        number_items.append((name, label))
        return MenuItem("num", lambda *_: None)

    monkeypatch.setattr(ModelCatalogMenu, "make_local_parameter_item", staticmethod(fake_item))

    ModelDotsMenu.menu(state)  # SOAR default
    # Number items are ordered: num steps (directly under Compile), speaker scale, CFG
    assert number_items == [
        ("num_steps_soar", "Steps (soar)"),
        ("speaker_scale", "Speaker scale"),
        ("guidance_scale", "CFG"),
    ]
    soar = DotsBaseModel.LOCAL.number("num_steps_soar")
    assert (soar.default, soar.min, soar.max) == (10, 1, 32)

    number_items.clear()
    set_setting(state.project, "dots_target", DotsBaseModel.MF_REPO_ID)
    ModelDotsMenu.menu(state)
    # Meanflow drops CFG, so no guidance number item is offered
    assert number_items == [("num_steps_mf", "Steps (mf)"), ("speaker_scale", "Speaker scale")]
    mf = DotsBaseModel.LOCAL.number("num_steps_mf")
    assert (mf.default, mf.min, mf.max) == (4, 1, 8)


def test_dots_target_submenu_has_only_presets_and_invalidates_worker(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project()))
    captured: dict[str, object] = {}
    saves: list[bool] = []
    clears: list[bool] = []

    monkeypatch.setattr(
        "tts_audiobook_tool.menus.model.model_dots_menu.MenuUtil.options_menu",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(Project, "save", lambda self: saves.append(True) or "")
    monkeypatch.setattr(
        "tts_audiobook_tool.menus.model.model_dots_menu.ModelWorker.clear_models_if_running_blocking",
        lambda: clears.append(True) or "",
    )

    ModelDotsMenu.target_submenu(state)

    assert captured["labels"] == DotsBaseModel.PRESET_REPO_IDS
    assert captured["values"] == DotsBaseModel.PRESET_REPO_IDS
    assert len(captured["sublabels"]) == 4
    captured["on_select"](DotsBaseModel.MF_REPO_ID)  # type: ignore[operator]
    assert get_setting(state.project, "dots_target") == DotsBaseModel.MF_REPO_ID
    assert saves == [True]
    assert clears == [True]


def test_dots_compile_submenu_options_and_invalidates_worker(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project()))
    captured: dict[str, object] = {}
    saves: list[bool] = []
    clears: list[bool] = []

    monkeypatch.setattr(
        "tts_audiobook_tool.menus.model.model_dots_menu.MenuUtil.options_menu",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(Project, "save", lambda self: saves.append(True) or "")
    monkeypatch.setattr(
        "tts_audiobook_tool.menus.model.model_dots_menu.ModelWorker.clear_models_if_running_blocking",
        lambda: clears.append(True) or "",
    )

    ModelDotsMenu.compile_submenu(state)

    modes = list(DotsCompileMode)
    assert captured["labels"] == [mode.name.title() for mode in modes]
    assert captured["values"] == modes
    assert captured["sublabels"] == [mode.description for mode in modes]
    assert captured["subheading"]
    assert captured["current_value"] is DotsCompileMode.default()

    captured["on_select"](DotsCompileMode.DISABLED)  # type: ignore[operator]
    assert get_setting(state.project, "dots_compile") is False
    assert saves == [True]
    assert clears == [True]
