"""Backend adapters that let one renderer draw catalog-declared settings menus.

`ModelCatalogMenu` and `VoiceCatalogMenu` know nothing about a specific
backend: they walk a definition's menu controls and ask a `CatalogMenuSource`
for parameters, stored values, persistence and the few per-backend
presentation choices. Adapters exist for SGL-Omni, audio.cpp and local models
(`[models.local]`); supporting another backend means adding an adapter and one
branch in `catalog_menu_source`, not touching the renderers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol, Sequence

from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import (
    AudioCppModelDefinition,
    AudioCppParameter,
    AudioCppTextParameter,
)
from tts_audiobook_tool.tts_models.local_definition import LocalChoiceParameter, LocalModelDefinition
from tts_audiobook_tool.tts_models.model_catalog import ParameterChoice
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredSettings
from tts_audiobook_tool.tts_models.sgl_omni_definition import (
    ChoiceParameter,
    NumericParameter,
    SglOmniModelDefinition,
)

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.state import State
    from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class NumericSetting(Protocol):
    """The bounded numeric parameter shape every backend shares."""
    @property
    def model_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def type(self) -> str: ...
    @property
    def default(self) -> int | float: ...
    @property
    def default_sentinel(self) -> int | float | None: ...
    @property
    def min(self) -> int | float: ...
    @property
    def max(self) -> int | float: ...
    @property
    def input_prompt_suffix(self) -> str: ...
    def validate(self, value: object) -> int | float: ...


class ChoiceSetting(Protocol):
    """The fixed-choice parameter shape every backend shares."""
    @property
    def model_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def default(self) -> str: ...
    @property
    def choices(self) -> Sequence[ParameterChoice]: ...


class CatalogMenuControl(Protocol):
    """One declared menu entry (`[[models.<backend>.menu]]`)."""
    @property
    def kind(self) -> str: ...
    @property
    def target_menu(self) -> str | None: ...
    @property
    def parameter(self) -> str: ...
    @property
    def label(self) -> str: ...
    @property
    def subheading(self) -> str: ...


# "standard": the stock bounded-number editor (`ask.ask_number_and_save`).
# "standard_reset": the stock editor, also accepting -1 to reset to the default.
# "sentinel_reset": SGL-Omni's editor, where entering the sentinel resets to default.
NumericEditStyle = Literal["standard", "standard_reset", "sentinel_reset"]

PresetChoices = Callable[["State"], list[str]] | None


class CatalogMenuSource(Protocol):
    """What the catalog renderers need from a backend."""
    @property
    def model_id(self) -> str: ...
    @property
    def controls(self) -> Sequence[CatalogMenuControl]: ...
    @property
    def seed_batch_warning(self) -> bool: ...

    def numeric(self, name: str) -> NumericSetting: ...
    def choice(self, name: str) -> ChoiceSetting: ...
    def get_number(self, project: Project, parameter: NumericSetting) -> int | float:
        """Effective value; raises ValueError on invalid storage.

        Numeric saves go through the stock editor's `SettingRef`; only
        SGL-Omni's sentinel-reset editor saves through its adapter."""
        ...
    def get_choice(self, project: Project, parameter: ChoiceSetting) -> str: ...
    def set_choice(self, project: Project, parameter: ChoiceSetting, value: str) -> str: ...
    def edit_style(self, parameter: NumericSetting) -> NumericEditStyle: ...
    def shows_moss_hint(self, parameter: NumericSetting) -> bool: ...
    def request_cap(self, parameter: NumericSetting) -> int | float | None:
        """A request-side upper limit the editor enforces (may be below the
        UI max), or None when only the parameter bounds apply."""
        ...
    def voice_preset_choices(self, model_type: TtsModelType) -> tuple[PresetChoices, bool]:
        """`(preset_choices, allow_custom_preset)` for the voice-sample items."""
        ...


@dataclass(frozen=True)
class SglOmniMenuSource:
    definition: SglOmniModelDefinition

    @property
    def model_id(self) -> str:
        return self.definition.spec.id

    @property
    def controls(self) -> Sequence[CatalogMenuControl]:
        return self.definition.menu

    @property
    def seed_batch_warning(self) -> bool:
        return True

    def numeric(self, name: str) -> NumericSetting:
        return self.definition.parameters[name]

    def choice(self, name: str) -> ChoiceSetting:
        return self.definition.choices[name]

    def get_number(self, project: Project, parameter: NumericSetting) -> int | float:
        assert isinstance(parameter, NumericParameter)
        return ConfiguredSettings.get(project, parameter)

    def set_number(self, project: Project, parameter: NumericSetting, value: int | float | None) -> str:
        """Save (None resets); returns an error message or "". Used by the sentinel-reset editor."""
        assert isinstance(parameter, NumericParameter)
        return ConfiguredSettings.set(project, parameter, value)

    def get_choice(self, project: Project, parameter: ChoiceSetting) -> str:
        assert isinstance(parameter, ChoiceParameter)
        return ConfiguredSettings.get_choice(project, parameter)

    def set_choice(self, project: Project, parameter: ChoiceSetting, value: str) -> str:
        assert isinstance(parameter, ChoiceParameter)
        return ConfiguredSettings.set_choice(project, parameter, value)

    def _is_moss_temperature(self, parameter: NumericSetting) -> bool:
        return (self.definition.language_policy == "moss"
                and isinstance(parameter, NumericParameter) and parameter.request_key == "temperature")

    def edit_style(self, parameter: NumericSetting) -> NumericEditStyle:
        # MOSS temperatures use the stock bounded-number editor; other controls
        # keep the server-specific reset/validation editor.
        return "standard" if self._is_moss_temperature(parameter) else "sentinel_reset"

    def shows_moss_hint(self, parameter: NumericSetting) -> bool:
        return self._is_moss_temperature(parameter)

    def request_cap(self, parameter: NumericSetting) -> int | float | None:
        return parameter.max_request_value if isinstance(parameter, NumericParameter) else None

    def voice_preset_choices(self, model_type: TtsModelType) -> tuple[PresetChoices, bool]:
        return None, False


# audio.cpp exposes no checkpoint speaker discovery. These are Qwen's official
# ids; custom GGUF checkpoints may accept additional speaker names.
QWEN_SPEAKERS = (
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
)


def get_qwen_speaker_choices(_: State) -> list[str]:
    return list(QWEN_SPEAKERS)


@dataclass(frozen=True)
class AudioCppMenuSource:
    definition: AudioCppModelDefinition

    @property
    def model_id(self) -> str:
        return self.definition.spec.id

    @property
    def controls(self) -> Sequence[CatalogMenuControl]:
        return self.definition.menu

    @property
    def seed_batch_warning(self) -> bool:
        return False

    def numeric(self, name: str) -> NumericSetting:
        parameter = self.definition.parameters[name]
        if not isinstance(parameter, AudioCppParameter):
            # String settings are expanded by a `choice` control.
            raise ValueError(f"{self.model_id}.{name} is not a numeric parameter")
        return parameter

    def choice(self, name: str) -> ChoiceSetting:
        parameter = self.definition.parameters[name]
        if not isinstance(parameter, AudioCppTextParameter) or not parameter.choices:
            raise ValueError(f"{self.model_id}.{name} is not a choice parameter")
        return parameter

    def get_number(self, project: Project, parameter: NumericSetting) -> int | float:
        assert isinstance(parameter, AudioCppParameter)
        value = AudioCppSettings.get(project, parameter)
        assert not isinstance(value, str)
        return value

    def get_choice(self, project: Project, parameter: ChoiceSetting) -> str:
        assert isinstance(parameter, AudioCppTextParameter)
        return str(AudioCppSettings.get(project, parameter))

    def set_choice(self, project: Project, parameter: ChoiceSetting, value: str) -> str:
        assert isinstance(parameter, AudioCppTextParameter)
        return AudioCppSettings.set(project, parameter, value)

    def edit_style(self, parameter: NumericSetting) -> NumericEditStyle:
        return "standard"

    def shows_moss_hint(self, parameter: NumericSetting) -> bool:
        return self.definition.language_policy == "moss" and parameter.name.endswith("temperature")

    def request_cap(self, parameter: NumericSetting) -> int | float | None:
        return None

    def voice_preset_choices(self, model_type: TtsModelType) -> tuple[PresetChoices, bool]:
        if model_type.id == "qwen3tts_audiocpp":
            return get_qwen_speaker_choices, True
        return None, False


@dataclass(frozen=True)
class LocalMenuSource:
    """Local models: catalog parameters over explicitly declared settings."""
    definition: LocalModelDefinition

    @property
    def model_id(self) -> str:
        return self.definition.spec.id

    @property
    def controls(self) -> Sequence[CatalogMenuControl]:
        return self.definition.menu

    @property
    def seed_batch_warning(self) -> bool:
        return self.definition.seed_batch_warning

    def numeric(self, name: str) -> NumericSetting:
        return self.definition.number(name)

    def choice(self, name: str) -> ChoiceSetting:
        return self.definition.choice(name)

    def get_number(self, project: Project, parameter: NumericSetting) -> int | float:
        stored = project.get_model_setting(parameter.model_id, parameter.name)
        if parameter.default_sentinel is not None and stored == parameter.default_sentinel:
            return parameter.default
        return parameter.validate(stored)

    def get_choice(self, project: Project, parameter: ChoiceSetting) -> str:
        assert isinstance(parameter, LocalChoiceParameter)
        return str(parameter.validate(project.get_model_setting(parameter.model_id, parameter.name)))

    def set_choice(self, project: Project, parameter: ChoiceSetting, value: str) -> str:
        assert isinstance(parameter, LocalChoiceParameter)
        project.set_model_setting(parameter.model_id, parameter.name, parameter.from_key(value))
        error = project.save()
        if not error:
            # Choices may be model construction parameters (eg compile, fp16,
            # sample rate); a changed value invalidates the loaded instance.
            from tts_audiobook_tool.tts import Tts
            Tts.set_model_params_using_project(project)
        return error

    def edit_style(self, parameter: NumericSetting) -> NumericEditStyle:
        return "standard" if parameter.default_sentinel is None else "standard_reset"

    def shows_moss_hint(self, parameter: NumericSetting) -> bool:
        return False

    def request_cap(self, parameter: NumericSetting) -> int | float | None:
        return None

    def voice_preset_choices(self, model_type: TtsModelType) -> tuple[PresetChoices, bool]:
        # Local voice menus are model-specific; they never render catalog voice controls.
        return None, False


CatalogDefinition = SglOmniModelDefinition | AudioCppModelDefinition | LocalModelDefinition


def catalog_menu_source(definition: object) -> CatalogMenuSource:
    """The adapter for a catalog-driven definition; the single backend switch."""
    if isinstance(definition, SglOmniModelDefinition):
        return SglOmniMenuSource(definition)
    if isinstance(definition, AudioCppModelDefinition):
        return AudioCppMenuSource(definition)
    if isinstance(definition, LocalModelDefinition):
        return LocalMenuSource(definition)
    raise TypeError(f"No catalog menu source for {type(definition).__name__}")
