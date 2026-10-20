"""The "voices may vary" run note covers every voice in play for the run, for any model."""
from __future__ import annotations

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.tts_models.voice_variation_warning import (
    GENERIC_VOICE_VARIATION_NOTE, format_voice_numbers, get_run_warnings, get_voice_variation_warnings,
)

NOTE = f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 2)"
QWEN_MODEL_IDS = ["qwen3tts_local", "qwen3tts_audiocpp"]


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


# The check is model-agnostic; these differ in what it reads: sample-optional,
# instructions-capable sample-optional, and voice-required.
@pytest.mark.parametrize("model_id", ["chatterbox_local", "omnivoice_local", "glm_local"])
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


@pytest.mark.parametrize("model_id", QWEN_MODEL_IDS)
@pytest.mark.parametrize("model_type", ["base", "custom_voice", "voice_design"])
@pytest.mark.parametrize("instructions", ["", " \n\t", "A warm, low voice."])
@pytest.mark.parametrize("stale_values", [False, True])
def test_qwen_variation_follows_active_model_type(model_id, model_type, instructions, stale_values):
    # VoiceDesign ignores saved samples/transcripts/presets; Base and CustomVoice do not vary.
    voice = Voice(
        file_name="old.flac" if stale_values else "",
        transcript="Old transcript." if stale_values else "",
        presets={model_id: "Ryan"} if stale_values else {},
        instructions={model_id: instructions},
    )
    project = Project(tts_model_type=model_id, voice_references=[voice])
    project.set_model_setting(model_id, "model_type", model_type)
    support = Tts.get_model_support(project)
    assert project.get_tts_model_type().value.requires_voice
    assert support.can_voice_clone(project) is (model_type == "base")
    assert support.expects_voice_instructions(project) is (model_type == "voice_design")

    expected = []
    if model_type == "voice_design":
        expected = [f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 1)"]
    # Missing Base samples and CustomVoice presets belong to validation, not this note.
    assert get_voice_variation_warnings(project, [0, 0]) == expected
    assert get_run_warnings(project, _NoModelWarnings(), None) == expected


@pytest.mark.parametrize("model_id", QWEN_MODEL_IDS)
def test_qwen_voice_design_groups_only_used_voices_once(model_id):
    # Described and empty slots share the generic constant and one grouped note.
    project = Project(
        tts_model_type=model_id,
        voice_select_mode=VoiceSelectMode.USER_DEFINED.id,
        voice_references=[
            Voice("old.flac", "Old transcript.", presets={model_id: "Ryan"},
                  instructions={model_id: "A warm, low voice."}),
            Voice(),
            Voice(instructions={model_id: "A bright, high voice."}),
            Voice(instructions={model_id: " \n\t"}),
            Voice(instructions={model_id: "Unused description."}),
        ],
    )
    project.set_model_setting(model_id, "model_type", "voice_design")
    project.phrase_groups = [
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=index)
        for index in [2, 3, 0, 2, 1, 4]
    ]
    expected = [f"{GENERIC_VOICE_VARIATION_NOTE} (Voices 1, 3, 4)"]
    assert get_voice_variation_warnings(project, [3, 0, 2, 0, -1, 99]) == expected

    class Model:
        def get_warning_issues(self, project: Project) -> list[str]:
            return ["model warning"]

    assert get_run_warnings(project, Model(), [3, 1, 2, 0, 3]) == expected + ["model warning"]
    assert get_run_warnings(project, _NoModelWarnings(), [4]) == [
        f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 2)"]
    assert get_run_warnings(project, _NoModelWarnings(), []) == []


def test_vibevoice_lora_suppresses_the_note():
    project = Project(tts_model_type="vibevoice_local", voice_references=[Voice()])
    assert get_voice_variation_warnings(project, [0]) == [f"{GENERIC_VOICE_VARIATION_NOTE} (Voice 1)"]
    project.set_model_setting("vibevoice_local", "lora_target", "some/lora")
    assert get_voice_variation_warnings(project, [0]) == []
