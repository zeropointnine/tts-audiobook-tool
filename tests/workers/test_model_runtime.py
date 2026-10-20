import pytest

from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.model_runtime import (
    ModelRuntimeRole,
    current_role,
    mark_interactive_main,
    mark_model_worker,
    mark_standalone,
)
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts


@pytest.fixture(autouse=True)
def restore_standalone_role():
    mark_standalone()
    yield
    mark_standalone()


def test_interactive_main_rejects_all_heavy_model_boundaries() -> None:
    mark_interactive_main()

    guarded_calls = (
        Tts.get_instance,
        Tts.get_instance_if_exists,
        Tts.instance_exists,
        Tts.clear_tts_model,
        Stt.get_whisper,
        Stt.has_instance,
        Stt.clear_stt_model,
        ModelManager.get_yamnet_detector,
        ModelManager.get_lava_sr_upsampler,
        ModelManager.is_any_model_loaded,
        ModelManager.clear_all_models,
        ModelManager.clear_yamnet_detector,
        ModelManager.clear_lava_sr_upsampler,
    )
    for call in guarded_calls:
        with pytest.raises(RuntimeError, match="ModelWorker"):
            call()


def test_runtime_roles_are_explicit_and_process_local() -> None:
    assert current_role() is ModelRuntimeRole.STANDALONE
    mark_interactive_main()
    assert current_role() is ModelRuntimeRole.INTERACTIVE_MAIN
    mark_model_worker()
    assert current_role() is ModelRuntimeRole.MODEL_WORKER
    mark_standalone()
    assert current_role() is ModelRuntimeRole.STANDALONE
