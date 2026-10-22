"""Local models declared in `[models.local]` render through the shared catalog menu."""
from types import SimpleNamespace

import pytest

from catalog_toml_support import read_catalog, write_catalog
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.catalog_menu_source import LocalMenuSource, catalog_menu_source
from tts_audiobook_tool.menus.model.model_catalog_menu import ModelCatalogMenu
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts_models.local_definition import get_local_definition, load_local_definitions
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH


CATALOG_MENU_IDS = [
    "higgs_v2_local", "mira_local", "breeze_tts_2_local",
    "fish_s1_local", "fish_s2_local", "indextts2_local", "glm_local",
]


def labels(state, items):
    return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]


@pytest.mark.parametrize("model_id", CATALOG_MENU_IDS)
def test_menu_routes_catalog_declared_local_models_to_shared_renderer(monkeypatch, model_id):
    state = SimpleNamespace(project=Project(tts_model_type=model_id))
    captured = {}
    monkeypatch.setattr(ModelMenuShared, "menu_wrapper", lambda _state, items, **_: captured.update(items=items))
    ModelMenuShared.menu(state)
    assert captured["items"] is ModelMenuShared.make_local_items
    definition = get_local_definition(model_id)
    assert definition is not None and isinstance(catalog_menu_source(definition), LocalMenuSource)
    rendered = labels(state, ModelMenuShared.make_local_items(state))
    assert rendered[-1] == "Seed (currently: random)"
    assert len(rendered) == len(definition.menu)


def test_mira_labels_show_catalog_defaults():
    state = SimpleNamespace(project=Project(tts_model_type="mira_local"))
    assert labels(state, ModelMenuShared.make_local_items(state)) == [
        "Temperature (currently: 0.70 default)",
        "Top-P (currently: 0.95 default)",
        "Top-K (currently: 50 default)",
        "Repetition penalty (currently: 1.20 default)",
        "Seed (currently: random)",
    ]


@pytest.fixture
def edit_ui(monkeypatch):
    prompts, errors, saves = [], [], []
    responses = []
    monkeypatch.setattr("tts_audiobook_tool.ask.printt", prompts.append)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_error", errors.append)
    monkeypatch.setattr("tts_audiobook_tool.ask.print_feedback", lambda *_: None)
    monkeypatch.setattr("tts_audiobook_tool.ask.ask_input", lambda **kwargs: responses.pop(0))
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self) or "")
    return prompts, errors, saves, responses


def _breeze_cfg_item(state):
    definition = get_local_definition("breeze_tts_2_local")
    assert definition is not None
    return ModelCatalogMenu.make_parameter_item(
        state, definition.parameters["cfg"], "CFG", catalog_menu_source(definition))


def test_local_editor_saves_rejects_and_resets_to_sentinel(edit_ui):
    # Local settings keep the -1 "use default" sentinel: entering -1 resets.
    prompts, errors, saves, responses = edit_ui
    project = Project(tts_model_type="breeze_tts_2_local")
    state = SimpleNamespace(project=project)
    item = _breeze_cfg_item(state)

    responses.append("2.5")
    item.handler(state, item)
    assert project.get_model_setting("breeze_tts_2_local", "cfg") == 2.5
    assert strip_ansi_codes(prompts[0]) == (
        "Enter CFG\n(used only by voices with instructions)\n(valid range: 1.0-10.0; default: 4.0)")

    responses.append("11")
    item.handler(state, item)
    assert errors == ["Out of range"]
    assert project.get_model_setting("breeze_tts_2_local", "cfg") == 2.5

    responses.append("-1")
    item.handler(state, item)
    assert project.get_model_setting("breeze_tts_2_local", "cfg") == -1
    assert strip_ansi_codes(get_string_from(state, item.label)) == "CFG (currently: 4.00 default)"
    assert len(saves) == 2


def test_local_editor_prefills_effective_default_without_saving(edit_ui):
    _, _, saves, responses = edit_ui
    project = Project(tts_model_type="breeze_tts_2_local")
    state = SimpleNamespace(project=project)
    item = _breeze_cfg_item(state)
    responses.append("4.0")  # The prefilled default, accepted unchanged
    item.handler(state, item)
    assert project.get_model_setting("breeze_tts_2_local", "cfg") == -1
    assert saves == []


def test_invalid_stored_local_value_is_flagged_in_label():
    project = Project(tts_model_type="higgs_v2_local")
    project.set_model_setting("higgs_v2_local", "temperature", 5.0)
    state = SimpleNamespace(project=project)
    first = labels(state, ModelMenuShared.make_local_items(state))[0]
    assert first.startswith("Temperature (invalid:")


def test_models_without_a_local_menu_keep_hand_written_menus():
    definition = get_local_definition("dots_local")
    assert definition is not None and not definition.has_menu
    state = SimpleNamespace(project=Project(tts_model_type="dots_local"))
    assert ModelMenuShared.make_local_items(state) == []


def test_stage_two_menus_render_choice_and_rolling_continuation_labels():
    fish = SimpleNamespace(project=Project(tts_model_type="fish_s2_local"))
    assert labels(fish, ModelMenuShared.make_local_items(fish))[:2] == [
        "Torch compile (currently: True default)",
        "Rolling continuation (currently: disabled default)",
    ]
    glm = SimpleNamespace(project=Project(tts_model_type="glm_local"))
    assert labels(glm, ModelMenuShared.make_local_items(glm)) == [
        "Model samplerate (currently: 24000hz default)", "Seed (currently: random)"]


@pytest.mark.parametrize("model_id,name,key,expected", [
    ("fish_s1_local", "compile_enabled", "False", False),
    ("indextts2_local", "use_fp16", "True", True),
    ("glm_local", "sr", "32000", 32000),
])
def test_choice_saves_typed_value_and_syncs_model_params(monkeypatch, model_id, name, key, expected):
    # Choices are often construction parameters, so saving one re-syncs the
    # model params (which invalidates a loaded instance when they differ).
    from tts_audiobook_tool.tts import Tts
    synced = []
    monkeypatch.setattr(Tts, "set_model_params_using_project", synced.append)
    monkeypatch.setattr(Project, "save", lambda self: "")
    project = Project(tts_model_type=model_id)
    definition = get_local_definition(model_id)
    assert definition is not None
    source = catalog_menu_source(definition)
    error = source.set_choice(project, source.choice(name), key)
    assert error == ""
    value = project.get_model_setting(model_id, name)
    assert value == expected and type(value) is type(expected)
    assert source.get_choice(project, source.choice(name)) == key
    assert synced == [project]


def test_invalid_stored_choice_is_rejected():
    project = Project(tts_model_type="glm_local")
    project.set_model_setting("glm_local", "sr", 44100)
    definition = get_local_definition("glm_local")
    assert definition is not None
    source = catalog_menu_source(definition)
    with pytest.raises(ValueError, match="must be one of"):
        source.get_choice(project, source.choice("sr"))


def test_rolling_continuation_uses_parameter_max(monkeypatch):
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared as Shared
    calls = []
    monkeypatch.setattr(Shared, "ask_rolling_continuation", lambda **kwargs: calls.append(kwargs))
    state = SimpleNamespace(project=Project(tts_model_type="fish_s2_local"))
    item = ModelMenuShared.make_local_items(state)[1]
    item.handler(state, item)
    assert calls[0]["max_value"] == 3
    assert calls[0]["qualifier_line"] == "Fish S2 rolling continuation requires batch size 1."


# --- Catalog validation ---

def _load_with(tmp_path, model_id, mutate):
    data = read_catalog(CATALOG_PATH)
    entry = next(entry for entry in data["models"] if entry["id"] == model_id)
    mutate(entry)
    return load_local_definitions(write_catalog(tmp_path / "catalog.toml", data))


@pytest.mark.parametrize("mutate,message", [
    (lambda e: e["local"]["parameters"]["temperature"].update(default_sentinel=-2), "default_sentinel"),
    (lambda e: e["local"]["parameters"]["temperature"].update(default=3.0), "min <= default <= max"),
    (lambda e: e["local"]["parameters"]["temperature"].update(type="list"), "expected float, int, bool or str"),
    (lambda e: e["local"]["parameters"]["temperature"].update(type="str", default="x"), "requires choices"),
    (lambda e: (e["local"]["parameters"].update(missing={"type": "float", "default": 1.0, "min": 0.0, "max": 2.0}),
                e["local"]["menu"].append({"parameter": "missing", "label": "Missing"})),
     "requires a `parameters` setting"),
    (lambda e: e["local"]["parameters"]["top_k"].update(type="float", default=50.0, min=1.0, max=100.0),
     "requires a `parameters` setting"),
    (lambda e: e["local"]["menu"].append(dict(e["local"]["menu"][0])), "each parameter at most once"),
    (lambda e: e["local"]["menu"].append({"kind": "seed"}), "at most one seed"),
    (lambda e: e["local"]["menu"].append({"kind": "voice_samples"}), "invalid control"),
    # Numeric controls omit `kind`; spelling out the implicit "parameter" is rejected.
    (lambda e: e["local"]["menu"][0].update(kind="parameter"), "numeric controls omit `kind`"),
    (lambda e: e["local"]["menu"][0].update(target_menu="voice"), "unsupported field"),
    (lambda e: e["local"].update(extra=1), "unsupported field"),
])
def test_local_declaration_is_validated(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        _load_with(tmp_path, "higgs_v2_local", mutate)


@pytest.mark.parametrize("model_id,mutate,message", [
    ("fish_s1_local", lambda e: e["local"]["parameters"]["compile_enabled"].update(min=0), "not valid for a choice"),
    ("fish_s1_local", lambda e: e["local"]["parameters"]["compile_enabled"].update(default=1), "expected bool"),
    ("fish_s1_local", lambda e: e["local"]["menu"][0].pop("kind"), "requires the choice control"),
    ("fish_s1_local", lambda e: e["local"]["menu"][1].update(kind="choice"), "choice requires a bool"),
    ("fish_s1_local", lambda e: e["local"]["parameters"]["compile_enabled"].update(default=False),
     "must equal the local parameter default"),
    ("glm_local", lambda e: e["local"]["parameters"]["sr"].update(default=44100), "default must be one of"),
    ("glm_local", lambda e: e["local"]["parameters"]["sr"]["choices"].append({"value": "x", "label": "X"}),
     "expected int"),
    ("glm_local", lambda e: e["local"]["parameters"]["sr"].pop("choices") and None, "min"),
    ("fish_s2_local", lambda e: e["local"]["menu"][1].update(label="Rolling"), "fixed label"),
    ("fish_s2_local", lambda e: e["local"]["menu"][1].update(parameter="temperature"),
     "requires an int parameter without a sentinel"),
])
def test_choice_and_special_controls_are_validated(tmp_path, model_id, mutate, message):
    with pytest.raises(ValueError, match=message):
        _load_with(tmp_path, model_id, mutate)


def test_seed_control_requires_a_seed_setting(tmp_path):
    def mutate(entry):
        entry["settings"] = [s for s in entry["settings"] if s["name"] != "seed"]
    with pytest.raises(ValueError, match="seed control requires a seed setting"):
        _load_with(tmp_path, "higgs_v2_local", mutate)


def test_server_entries_cannot_declare_local_groups(tmp_path):
    def mutate(entry):
        entry["local"] = {"parameters": {}}
    with pytest.raises(ValueError, match="server models cannot contain local definitions"):
        _load_with(tmp_path, "higgs_v3_sglomni", mutate)


@pytest.mark.parametrize("name,invalid", [("temperature", 2.5), ("top_k", 0)])
def test_out_of_bounds_saved_local_value_resets_on_load(name, invalid):
    # Local catalog bounds validate saved values like server parameter bounds do.
    model_id = "mira_local"
    source = {"models": {model_id: {"parameters": {name: invalid}}}}
    project = Project.model_validate({"version": 3, "tts_model_type": model_id, "model_settings": source})
    assert project.get_model_setting(model_id, name) == -1


@pytest.mark.parametrize("stored,expected", [(44100, 24000), (32000, 32000)])
def test_saved_local_choice_outside_its_choices_resets_on_load(stored, expected):
    # Local models have no readiness check on stored settings, so a value of the
    # right type but outside the catalog choices resets on load, like bounds do.
    model_id = "glm_local"
    source = {"models": {model_id: {"parameters": {"sr": stored}}}}
    project = Project.model_validate({"version": 3, "tts_model_type": model_id, "model_settings": source})
    assert project.get_model_setting(model_id, "sr") == expected


def test_saved_local_sentinel_and_in_bounds_values_survive_load():
    model_id = "mira_local"
    source = {"models": {model_id: {"parameters": {"temperature": 1.5, "top_k": -1}}}}
    project = Project.model_validate({"version": 3, "tts_model_type": model_id, "model_settings": source})
    assert project.get_model_setting(model_id, "temperature") == 1.5
    assert project.get_model_setting(model_id, "top_k") == -1


# --- Model-side resolution ---

def test_resolve_maps_sentinel_and_invalid_values_to_catalog_defaults():
    # Inference reads effective values leniently: unset (-1) or invalid storage
    # (eg an older project) resolves to the catalog default, never an error.
    definition = get_local_definition("mira_local")
    assert definition is not None
    project = Project(tts_model_type="mira_local")
    assert definition.resolve_float(project, "temperature") == 0.7
    project.set_model_setting("mira_local", "temperature", 1.4)
    assert definition.resolve_float(project, "temperature") == 1.4
    project.set_model_setting("mira_local", "temperature", 9.0)
    assert definition.resolve_float(project, "temperature") == 0.7
    top_k = definition.resolve_int(project, "top_k")
    assert top_k == 50 and type(top_k) is int


def test_resolve_typed_choices():
    glm = get_local_definition("glm_local")
    fish = get_local_definition("fish_s1_local")
    assert glm is not None and fish is not None
    project = Project(tts_model_type="glm_local")
    assert glm.resolve(project, "sr") == 24000
    project.set_model_setting("glm_local", "sr", 32000)
    assert glm.resolve(project, "sr") == 32000
    project.set_model_setting("glm_local", "sr", 44100)
    assert glm.resolve(project, "sr") == 24000
    assert fish.resolve(Project(tts_model_type="fish_s1_local"), "compile_enabled") is True


def test_float_choice_keys_are_canonical(tmp_path):
    # A float setting may hold an int (catalog `default = 1`, or a stored `1`);
    # both must key as the matching choice ("1.0"), or the menu's label lookup fails.
    def mutate(entry):
        storage = next(s for s in entry["settings"] if s["name"] == "temperature")
        storage.pop("sentinel")
        storage["default"] = 1.0
        entry["local"]["parameters"]["temperature"] = {"type": "float", "default": 1, "choices": [
            {"value": 0.5, "label": "Low"}, {"value": 1.0, "label": "Normal"}]}
        control = next(c for c in entry["local"]["menu"] if c.get("parameter") == "temperature")
        control["kind"] = "choice"

    parameter = _load_with(tmp_path, "mira_local", mutate)["mira_local"].choice("temperature")
    keys = {choice.value for choice in parameter.choices}
    assert parameter.default == "1.0" and parameter.default in keys
    assert str(parameter.validate(1)) == "1.0"
    assert type(parameter.validate(1)) is float


@pytest.mark.parametrize("model_id", [
    "breeze_tts_2_local", "chatterbox_local", "dots_local", "fish_s1_local", "fish_s2_local", "glm_local",
    "higgs_v2_local", "indextts2_local", "mira_local", "omnivoice_local", "pocket_local", "vibevoice_local",
])
def test_every_declared_parameter_resolves_to_its_default_for_a_new_project(model_id):
    definition = get_local_definition(model_id)
    assert definition is not None and definition.parameters
    project = Project(tts_model_type=model_id)
    for name, parameter in definition.parameters.items():
        expected = getattr(parameter, "typed_default", None)
        if expected is None:
            expected = parameter.default
        assert definition.resolve(project, name) == expected
