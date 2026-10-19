"""The "voices may vary" run note covers every voice in play for the run, for any model."""
from __future__ import annotations

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.tts_models.tts_model_type import TtsBackendKind, TtsModelType
from tts_audiobook_tool.tts_models.voice_variation_warning import (
    GENERIC_VOICE_VARIATION_NOTE, format_voice_numbers, get_run_warnings, get_voice_variation_warnings,
)

NOTE = f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 2)"


class _NoModelWarnings:
    def get_warning_issues(self, project: Project) -> list[str]:
        return []


def make_project(model_id="chatterbox_local", mode=VoiceSelectMode.USER_DEFINED) -> Project:
    # Voice 1 has a sample, voice 2 does not; line N is assigned to voice N % 2
    project = Project(tts_model_type=model_id, voice_select_mode=mode.id,
                      voice_references=[Voice("a.flac", "text"), Voice()])
    project.phrase_groups = [PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=index % 2)
                             for index in range(4)]
    return project


@pytest.mark.parametrize("mode, lines, warned", [
    # Only voice 1 (which has a sample) is in play
    (VoiceSelectMode.DISABLED, [0, 1, 2], False),
    (VoiceSelectMode.USER_DEFINED, [0, 2], False),
    (VoiceSelectMode.USER_DEFINED, None, False),
    # The sample-less voice 2 is in play: previously missed, since only voice 1 was checked
    (VoiceSelectMode.USER_DEFINED, [0, 1], True),
    (VoiceSelectMode.USER_DEFINED, [3], True),
    (VoiceSelectMode.AUTO_ADVANCE, [0], True),
    (VoiceSelectMode.AUTO_ADVANCE, None, True),
    # An empty job uses no voices
    (VoiceSelectMode.AUTO_ADVANCE, [], False),
])
def test_note_follows_the_runs_voices(mode, lines, warned):
    project = make_project(mode=mode)
    assert get_run_warnings(project, _NoModelWarnings(), lines) == ([NOTE] if warned else [])


def test_note_comes_before_model_warnings():
    project = make_project(mode=VoiceSelectMode.AUTO_ADVANCE)

    class Model:
        def get_warning_issues(self, project: Project) -> list[str]:
            return ["model warning"]

    assert get_run_warnings(project, Model(), None) == [NOTE, "model warning"]


def _local_models_with_voice_storage() -> list[str]:
    return [model.id for model in TtsModelType.all()
            if model.value.backend_kind is TtsBackendKind.LOCAL and REGISTRY.voice_binding(model.id) is not None]


@pytest.mark.parametrize("model_id", _local_models_with_voice_storage())
def test_rule_is_the_same_for_every_model(model_id):
    project = make_project(model_id, VoiceSelectMode.AUTO_ADVANCE)
    model = TtsModelType.require_by_id(model_id).value
    # Required samples are the voice pre-flight's job; there is no "may vary" note then.
    assert get_voice_variation_warnings(project, [0, 1]) == ([] if model.requires_voice else [NOTE])
    assert get_voice_variation_warnings(project, [0]) == []


def test_model_without_voice_storage_never_warns():
    project = make_project("none", VoiceSelectMode.AUTO_ADVANCE)
    assert get_voice_variation_warnings(project, [0, 1]) == []


def test_instructions_without_a_model_specific_note_use_the_generic_note():
    # Local OmniVoice takes instructions but declares no variation note of its own
    project = Project(tts_model_type="omnivoice_local",
                      voice_references=[Voice(instructions={"omnivoice_local": "female, low pitch"})])
    assert get_voice_variation_warnings(project, [0]) == [f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 1)"]


def test_out_of_range_indices_are_ignored():
    project = make_project()
    assert get_voice_variation_warnings(project, [0, 7, -1]) == []


def test_note_lists_every_sample_less_voice_in_play():
    project = Project(tts_model_type="chatterbox_local",
                      voice_references=[Voice(), Voice("a.flac", "text"), Voice(), Voice()])
    assert get_voice_variation_warnings(project, [3, 0, 1, 2, 0]) == [
        f"{GENERIC_VOICE_VARIATION_NOTE} (Voices 1, 3, 4)"]


@pytest.mark.parametrize("indices, expected", [
    ([], ""), ([0], " (Voice 1)"), ([2, 0, 2], " (Voices 1, 3)"),
])
def test_format_voice_numbers(indices, expected):
    assert format_voice_numbers(indices) == expected


def test_vibevoice_lora_suppresses_the_note():
    project = Project(tts_model_type="vibevoice_local", voice_references=[Voice()])
    assert get_voice_variation_warnings(project, [0]) == [f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 1)"]
    project.set_model_setting("vibevoice_local", "lora_target", "some/lora")
    assert get_voice_variation_warnings(project, [0]) == []
