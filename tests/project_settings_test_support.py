"""Test helpers for model-scoped project settings.

Version 3 stores model settings in `Project.model_settings`, keyed by the stable
model or shared-group ID and the declared setting name. Tests written against
the former flat project fields can keep naming those fields through these
helpers: the declaration registry maps each old field name to the (model, name)
pair that now owns its value.
"""
from __future__ import annotations

from typing import Any

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY


def set_setting(project: Project, flat_field: str, value: Any) -> None:
    """Store an override for the setting formerly written as `flat_field`."""
    binding = REGISTRY.legacy[flat_field]
    project.set_model_setting(binding.model_id, binding.name, value)


def get_setting(project: Project, flat_field: str) -> Any:
    """Read the effective value of the setting formerly written as `flat_field`."""
    binding = REGISTRY.legacy[flat_field]
    return project.get_model_setting(binding.model_id, binding.name)


def model_object(project: Project, model_id: str) -> dict[str, Any]:
    """The project's private storage object for one model, or an empty object."""
    return project.model_settings.models.get(model_id, {})


def shared_object(project: Project, group: str) -> dict[str, Any]:
    """The project's storage object for one shared group, or an empty object."""
    return project.model_settings.shared.get(group, {})
