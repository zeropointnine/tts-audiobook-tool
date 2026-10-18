"""`can_instructions` / `can_voice_clone` support getters default to the catalog spec."""
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.sgl_omni_configured import ConfiguredModelSupport
from tts_audiobook_tool.tts_models.sgl_omni_definition import load_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

AUK_IDS = {"auk_sglomni", "auk_flash_sglomni"}

def test_every_model_support_reports_catalog_capabilities():
    sgl_omni = load_definitions().models
    audio_cpp = load_audio_cpp_definitions().models
    checked = []
    for model in TtsModelType.all():
        kind = model.value.backend_kind
        if kind == TtsBackendKind.SGL_OMNI:
            support = ConfiguredModelSupport(sgl_omni[model.id])
        elif kind == TtsBackendKind.AUDIO_CPP:
            support = AudioCppModelSupport(audio_cpp[model.id])
        else:
            support = Tts.get_class_for_type(model)
        project = Project(tts_model_type=model.id)
        # Runtime defaults: CosyVoice3 (either backend) defaults to Zero-shot, so
        # it reads False here; AuK always takes instructions (sample-less voices).
        if kind == TtsBackendKind.AUDIO_CPP:
            expected_instructions = model.id in {"omnivoice_audiocpp", "breeze_tts_2_audiocpp"}
        elif kind == TtsBackendKind.SGL_OMNI:
            expected_instructions = model.id in AUK_IDS
        else:
            expected_instructions = model.value.can_instructions
        assert support.can_instructions(project) is expected_instructions, model.id
        assert support.uses_instructions_with_sample(project) is (model.id not in AUK_IDS), model.id
        assert support.can_voice_clone(project) is model.value.can_voice_clone, model.id
        checked.append(model.id)
    assert "none" in checked and "omnivoice_local" in checked


def test_local_omnivoice_and_placeholder_values():
    project = Project()
    omnivoice = Tts.get_class_for_type(TtsModelType.require_by_id("omnivoice_local"))
    placeholder = Tts.get_class_for_type(TtsModelType.require_by_id("none"))
    assert omnivoice.can_instructions(project) is True and omnivoice.can_voice_clone(project) is True
    assert placeholder.can_instructions(project) is False and placeholder.can_voice_clone(project) is False
