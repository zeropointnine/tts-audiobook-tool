from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts


def test_warm_up_retains_desired_loaded_stt_and_clears_unwanted_yamnet() -> None:
    state = SimpleNamespace(project=SimpleNamespace())
    tts_instance = object()
    with patch.object(Stt, "should_skip", return_value=False), patch.object(
        Tts, "instance_exists", return_value=True
    ), patch.object(Stt, "has_instance", return_value=True), patch.object(
        Tts, "get_instance", return_value=tts_instance
    ), patch.object(
        Tts,
        "get_class",
        return_value=SimpleNamespace(can_hallucinate_music=lambda project, instance: False),
    ), patch.object(Stt, "eager_warm_up_for_inference") as warm_stt, patch.object(
        ModelManager, "clear_yamnet_detector"
    ) as clear_yamnet:
        result = ModelManager.warm_up_models(state)  # type: ignore[arg-type]

    assert not result.should_stop
    warm_stt.assert_not_called()
    clear_yamnet.assert_called_once_with()


def test_warm_up_clears_disabled_stt_without_skipping_tts_reconciliation() -> None:
    state = SimpleNamespace(project=SimpleNamespace())
    tts_instance = object()
    with patch.object(Stt, "should_skip", return_value=True), patch.object(
        Tts, "instance_exists", return_value=True
    ), patch.object(Stt, "has_instance", return_value=True), patch.object(
        Stt, "clear_stt_model"
    ) as clear_stt, patch.object(
        Tts, "get_instance", return_value=tts_instance
    ) as get_tts, patch.object(
        Tts,
        "get_class",
        return_value=SimpleNamespace(can_hallucinate_music=lambda project, instance: False),
    ), patch.object(ModelManager, "clear_yamnet_detector"):
        result = ModelManager.warm_up_models(state)  # type: ignore[arg-type]

    assert not result.should_stop
    clear_stt.assert_called_once_with()
    get_tts.assert_called_once_with()


def test_clear_all_models_is_best_effort() -> None:
    calls: list[str] = []

    def fail_stt() -> None:
        calls.append("stt")
        raise RuntimeError("stt failed")

    with patch.object(Stt, "clear_stt_model", side_effect=fail_stt), patch.object(
        Tts, "clear_tts_model", side_effect=lambda: calls.append("tts")
    ), patch.object(
        ModelManager, "clear_yamnet_detector", side_effect=lambda: calls.append("yamnet")
    ), patch.object(
        ModelManager, "clear_lava_sr_upsampler", side_effect=lambda: calls.append("lava")
    ), patch(
        "tts_audiobook_tool.model_manager.app_memory.gc_ram_vram",
        side_effect=lambda: calls.append("gc"),
    ):
        with pytest.raises(RuntimeError, match="stt failed"):
            ModelManager.clear_all_models()

    assert calls == ["stt", "tts", "yamnet", "lava", "gc"]
