from importlib import import_module

import pytest


@pytest.mark.parametrize("module_name,class_name", [
    ("textual.content_textual_app", "ContentTextualApp"),
    ("textual.worker_app", "WorkerTextualApp"),
    ("textual.text_input_app", "TextInputTextualApp"),
    ("textual.conversation_app", "ConversationTextualApp"),
    ("textual.voice_crop_app", "VoiceCropApp"),
    ("textual.text_editor", "TextEditor"),
    ("textual.generate_editor", "GenerateEditor"),
    ("textual.section_markers_editor", "SectionMarkersEditor"),
    ("textual.voice_line_editor", "VoiceLineEditorTextualApp"),
    ("textual.word_substitutions_app", "WordSubstitutionsApp"),
    ("textual.generation_app", "GenerationApp"),
    ("textual.real_time_playback_app", "RealTimePlaybackApp"),
    ("enhance.unmatched_lines_app", "UnmatchedLinesApp"),
])
def test_all_textual_apps_shadow_ctrl_q_with_a_noop(module_name, class_name):
    app_class = getattr(import_module(f"tts_audiobook_tool.{module_name}"), class_name)
    # Inspect Textual's merged bindings, including inherited base-app guards.
    bindings = app_class._merged_bindings.key_to_bindings["ctrl+q"]
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.priority
    assert not binding.show
    assert binding.action in {"ignore_ctrl_q", "ignore"}
    # The handler must not need app state or trigger any exit workflow.
    handler = getattr(app_class, f"action_{binding.action}")
    assert handler(object()) is None
