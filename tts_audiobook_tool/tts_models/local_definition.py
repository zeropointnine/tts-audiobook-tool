"""Validated local-model parameter and menu declarations from the application catalog.

A local entry's optional `[models.local]` group describes settings it already
declares in `settings`: bounds, default and the -1 "use default" sentinel
(`[models.local.parameters.*]`), and optionally a catalog-driven Model settings
menu (`[[models.local.menu]]`). Loading has no inference dependencies, so model
code in a worker venv can use it too.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math
from pathlib import Path
from typing import Any, Protocol

from tts_audiobook_tool.tts_models.model_catalog import (
    CATALOG_PATH,
    ParameterChoice,
    _load_catalog,
    parse_local_declaration,
    parse_spec,
)
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind, TtsModelSpec


class SettingsReader(Protocol):
    """What resolving needs from a Project (kept structural for worker-side use)."""
    def get_model_setting(self, model_id: str, name: str) -> Any: ...


@dataclass(frozen=True)
class LocalParameter:
    model_id: str
    name: str
    type: str
    default: int | float
    min: int | float
    max: int | float
    default_sentinel: int | None = None
    input_prompt_suffix: str = ""

    def validate(self, value: object) -> int | float:
        typ = int if self.type == "int" else (int, float)
        if isinstance(value, bool) or not isinstance(value, typ) or not math.isfinite(value) or not self.min <= value <= self.max:
            raise ValueError(f"{self.model_id}.{self.name} must be a {self.type} between {self.min} and {self.max}")
        return value


@dataclass(frozen=True)
class LocalChoiceParameter:
    """A fixed-choice local setting (a bool is an implicit True/False choice).

    Menus key options by `str(value)`, so `default` and each choice's `value`
    are those keys; `values` keeps the typed values in choice order.
    """
    model_id: str
    name: str
    type: str
    typed_default: Any
    values: tuple[Any, ...]
    choices: tuple[ParameterChoice, ...]

    @property
    def default(self) -> str:
        return str(self.typed_default)

    def validate(self, value: object) -> Any:
        """The matching choice's typed value, if the stored value is one of the choices.

        Returns the catalog's value rather than the stored one: a float setting
        may store an int (`1`), which must key as its choice (`"1.0"`).
        """
        if self.type == "float":
            typed = not isinstance(value, bool) and isinstance(value, (int, float))
        else:
            typed = type(value) is {"bool": bool, "int": int, "str": str}[self.type]
        if typed and value in self.values:
            return self.values[self.values.index(value)]
        raise ValueError(f"{self.model_id}.{self.name} must be one of: {', '.join(map(str, self.values))}")

    def from_key(self, key: str) -> Any:
        for value, choice in zip(self.values, self.choices):
            if choice.value == key:
                return value
        raise ValueError(f"{self.model_id}.{self.name}: unknown choice {key!r}")


LocalParameterTypes = LocalParameter | LocalChoiceParameter


@dataclass(frozen=True)
class LocalMenuControl:
    kind: str
    parameter: str = ""
    label: str = ""
    subheading: str = ""
    # Local catalog menus render only in Model settings.
    target_menu: str | None = "model"


@dataclass(frozen=True)
class LocalModelDefinition:
    spec: TtsModelSpec
    parameters: dict[str, LocalParameterTypes]
    # Empty when the model keeps a hand-written Model settings menu.
    menu: tuple[LocalMenuControl, ...]
    seed_batch_warning: bool = False

    @property
    def has_menu(self) -> bool:
        return bool(self.menu)

    def number(self, name: str) -> LocalParameter:
        parameter = self.parameters[name]
        if not isinstance(parameter, LocalParameter):
            raise ValueError(f"{self.spec.id}.{name} is not a numeric parameter")
        return parameter

    def choice(self, name: str) -> LocalChoiceParameter:
        parameter = self.parameters[name]
        if not isinstance(parameter, LocalChoiceParameter):
            raise ValueError(f"{self.spec.id}.{name} is not a choice parameter")
        return parameter

    def resolve(self, project: SettingsReader, name: str) -> Any:
        """The effective value of a declared parameter, for inference.

        The -1 sentinel ("use default") and any stored value outside the
        catalog bounds/choices resolve to the catalog default. Lenient by
        design: load validation and the menus already reject invalid values.
        """
        parameter = self.parameters[name]
        stored = project.get_model_setting(self.spec.id, name)
        if isinstance(parameter, LocalChoiceParameter):
            try:
                return parameter.validate(stored)
            except ValueError:
                return parameter.typed_default
        if parameter.default_sentinel is not None and stored == parameter.default_sentinel:
            return parameter.default
        try:
            return parameter.validate(stored)
        except ValueError:
            return parameter.default

    def resolve_float(self, project: SettingsReader, name: str) -> float:
        return float(self.resolve(project, name))

    def resolve_int(self, project: SettingsReader, name: str) -> int:
        return int(self.resolve(project, name))


def load_local_definitions(path: Path = CATALOG_PATH) -> dict[str, LocalModelDefinition]:
    """Every local entry's declaration (possibly empty), keyed by model ID."""
    document, _, _, _ = _load_catalog(path)
    models: dict[str, LocalModelDefinition] = {}
    for entry in document["models"]:
        if entry.get("backend_kind") != TtsBackendKind.LOCAL.value:
            continue
        spec = parse_spec(entry)
        declaration = parse_local_declaration(entry, f"local model {spec.id}")
        parameters: dict[str, LocalParameterTypes] = {}
        for name, raw in declaration["parameters"].items():
            if raw["choices"] is not None:
                parameters[name] = LocalChoiceParameter(
                    spec.id, name, raw["type"], raw["default"],
                    tuple(item["value"] for item in raw["choices"]),
                    tuple(ParameterChoice(str(item["value"]), item["label"], item["description"])
                          for item in raw["choices"]))
            else:
                parameters[name] = LocalParameter(
                    spec.id, name, raw["type"], raw["default"], raw["min"], raw["max"],
                    raw["default_sentinel"], raw["input_prompt_suffix"])
        menu = tuple(
            LocalMenuControl("seed") if item["kind"] == "seed"
            else LocalMenuControl(item["kind"], item["parameter"], item["label"], item["subheading"])
            for item in declaration["menu"]
        )
        seed_batch_warning = any(item["kind"] == "seed" and item["batch_warning"] for item in declaration["menu"])
        models[spec.id] = LocalModelDefinition(spec, parameters, menu, seed_batch_warning)
    return models


@cache
def _installed() -> dict[str, LocalModelDefinition]:
    return load_local_definitions()


def get_local_definition(model_id: str) -> LocalModelDefinition | None:
    """The shipped catalog's declaration for a local model ID, or None."""
    return _installed().get(model_id)


def require_local_definition(model_id: str) -> LocalModelDefinition:
    definition = get_local_definition(model_id)
    if definition is None:
        raise ValueError(f"No local catalog definition for {model_id!r}")
    return definition
