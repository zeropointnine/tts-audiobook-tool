from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np

from tts_audiobook_tool.app_types import ExportType, NormalizationType, Sound
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import (
    ProjectSerializationUtil,
)
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil
from tts_audiobook_tool.sound.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State


def test_model_manager_lazily_loads_and_clears_lava_sr() -> None:
    model = MagicMock(spec=LavaSrUtil)
    ModelManager.lava_sr_upsampler = None

    with patch.object(LavaSrUtil, "has_lava_sr", return_value=True), patch(
        "tts_audiobook_tool.model_manager.LavaSrUtil", return_value=model
    ), patch("tts_audiobook_tool.model_manager.print_init") as print_init, patch(
        "tts_audiobook_tool.model_manager.app_memory.gc_ram_vram"
    ) as gc_ram_vram:
        assert ModelManager.get_lava_sr_upsampler() is model
        assert ModelManager.get_lava_sr_upsampler() is model
        ModelManager.clear_lava_sr_upsampler()

    print_init.assert_called_once_with("Initializing LavaSR v2 upsampler...")
    model.kill.assert_called_once_with()
    gc_ram_vram.assert_not_called()
    assert ModelManager.lava_sr_upsampler is None


def test_model_manager_returns_none_when_lava_sr_is_unavailable() -> None:
    ModelManager.lava_sr_upsampler = None

    with patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        assert ModelManager.get_lava_sr_upsampler() is None


def test_sound_pipeline_uses_lava_sr_without_denoising() -> None:
    sound = Sound(np.zeros(16, dtype=np.float32), 16_000)
    restored = Sound(np.zeros(48, dtype=np.float32), 48_000)
    upsampler = MagicMock()
    upsampler.process.return_value = restored

    with patch.object(
        ModelManager, "get_lava_sr_upsampler", return_value=upsampler
    ):
        assert SoundPipeline.apply_lava_sr_upsampling(sound) is restored

    upsampler.process.assert_called_once_with(sound, denoise=False)


def test_concat_unloads_worker_owned_models_before_lava_sr() -> None:
    with patch.object(ModelWorker, "is_alive", return_value=True), patch.object(
        ModelWorker, "clear_models_if_running_blocking", return_value=""
    ) as clear_worker_models, patch(
        "tts_audiobook_tool.concat_util.printt"
    ) as printt:
        error = ConcatUtil.unload_models_for_upsampling()

    assert error == ""
    clear_worker_models.assert_called_once_with()
    assert any(
        call.args and "free up VRAM for generative upsampling" in call.args[0]
        for call in printt.call_args_list
    )


def test_concat_reports_worker_model_unload_failure() -> None:
    with patch.object(ModelWorker, "is_alive", return_value=True), patch.object(
        ModelWorker, "clear_models_if_running_blocking", return_value="worker failed"
    ):
        error = ConcatUtil.unload_models_for_upsampling()

    assert "worker failed" in error


def _make_minimal_concat_state() -> State:
    project = SimpleNamespace(
        export_type=ExportType.FLAC,
        normalization_type=NormalizationType.DEFAULT,
        use_upsampler=True,
        use_break_sound_effect=False,
        high_shelf="disabled",
        reason_pauses=SimpleNamespace(),
        phrase_groups=[],
    )
    return cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(aac_bitrate="128k", save_debug_files=False),
        ),
    )


def test_concat_unloads_lava_sr_before_normalization() -> None:
    events: list[str] = []

    def concatenate(*args: object, **kwargs: object) -> list[float]:
        events.append("upsampling finished")
        return []

    def unload() -> str:
        events.append("LavaSR unloaded")
        return ""

    def normalize(**kwargs: object) -> str:
        events.append("normalization started")
        return "stop after order assertion"

    with patch.object(
        ProjectTextIOUtil, "load_raw_text", return_value="raw"
    ), patch.object(
        ConcatUtil, "make_phrases_and_paths", return_value=[]
    ), patch.object(
        ConcatUtil, "concatenate_sound_segments", side_effect=concatenate
    ), patch.object(
        ModelWorker, "clear_models_if_running_blocking", side_effect=unload
    ), patch.object(
        LoudnessNormalizationUtil, "normalize_file", side_effect=normalize
    ), patch(
        "tts_audiobook_tool.concat_util.app_support.log_unload_memory_snapshot"
    ), patch(
        "tts_audiobook_tool.concat_util.delete_silently"
    ):
        _, error = ConcatUtil.make_file(
            _make_minimal_concat_state(), 0, 0, [], "book", use_upsampler=True
        )

    assert error == "stop after order assertion"
    assert events == [
        "upsampling finished",
        "LavaSR unloaded",
        "normalization started",
    ]


def test_concat_unloads_lava_sr_when_upsampling_is_interrupted() -> None:
    with patch.object(
        ProjectTextIOUtil, "load_raw_text", return_value="raw"
    ), patch.object(
        ConcatUtil, "make_phrases_and_paths", return_value=[]
    ), patch.object(
        ConcatUtil,
        "concatenate_sound_segments",
        return_value="Interrupted by user",
    ), patch.object(
        ModelWorker, "clear_models_if_running_blocking", return_value=""
    ) as unload, patch.object(
        LoudnessNormalizationUtil, "normalize_file"
    ) as normalize, patch(
        "tts_audiobook_tool.concat_util.app_support.log_unload_memory_snapshot"
    ), patch(
        "tts_audiobook_tool.concat_util.delete_silently"
    ):
        _, error = ConcatUtil.make_file(
            _make_minimal_concat_state(), 0, 0, [], "book", use_upsampler=True
        )

    assert error == "Interrupted by user"
    unload.assert_called_once_with()
    normalize.assert_not_called()


def test_use_upsampler_serialization_field_remains_compatible() -> None:
    project = Project.model_validate({"use_upsampler": True})

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    loaded = Project.model_validate(payload)

    assert payload["use_upsampler"] is True
    assert loaded.use_upsampler is True


def test_concat_with_persisted_setting_skips_unavailable_lava_sr() -> None:
    project = SimpleNamespace(
        use_upsampler=True,
        concat_path="output",
        chapter_mode=SimpleNamespace(),
        markers=[],
        phrase_groups=[],
    )
    state = cast(
        State,
        SimpleNamespace(
            project=project,
            prefs=SimpleNamespace(),
        ),
    )

    with patch.object(ModelWorker, "probe_lava_sr_blocking", return_value=(False, "")), patch(
        "tts_audiobook_tool.concat_util.printt"
    ) as printt, patch(
        "tts_audiobook_tool.concat_util.app_support.log_unload_memory_snapshot",
    ), patch(
        "tts_audiobook_tool.concat_util.os.makedirs"
    ), patch(
        "tts_audiobook_tool.concat_util.make_stem", return_value="book"
    ), patch.object(
        ConcatUtil, "make_file", return_value=("book.abr.flac", "")
    ) as make_file, patch(
        "tts_audiobook_tool.concat_util.app_support.play_done_sound"
    ), patch(
        "tts_audiobook_tool.concat_util.app_hint_util.show_player_hint"
    ), patch(
        "tts_audiobook_tool.concat_util.ask.ask_hotkey", return_value=""
    ), patch.object(
        ModelManager, "clear_lava_sr_upsampler"
    ):
        ConcatUtil.make_files(state, [], [])

    assert make_file.call_args.kwargs["use_upsampler"] is False
    warning_messages = [
        call.args[0]
        for call in printt.call_args_list
        if call.args and "LavaSR v2 upsampling is enabled" in call.args[0]
    ]
    assert len(warning_messages) == 1
    assert project.use_upsampler
