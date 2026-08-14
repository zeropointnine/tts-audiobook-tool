from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import (
    ProjectSerializationUtil,
)
from tts_audiobook_tool.sound.lava_sr_util import LavaSrUtil
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


def test_concat_menu_always_shows_generative_upsampling() -> None:
    project = Project.model_validate({})
    prefs = SimpleNamespace(aac_bitrate="128k")
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))

    with patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.menu"
    ) as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ), patch.object(LavaSrUtil, "has_lava_sr", return_value=False):
        ConcatMenu.menu(state)
        make_items = menu.call_args.args[2]
        items = make_items(state)

    labels = [get_string_from(state, item.label) for item in items]
    assert any(label.startswith("Generative upsampling") for label in labels)


def test_concat_menu_prevents_enabling_unavailable_lava_sr() -> None:
    project = Project.model_validate({"use_upsampler": False})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(LavaSrUtil, "has_lava_sr", return_value=False), patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.options_menu"
    ) as options_menu, patch(
        "tts_audiobook_tool.menus.concat_menu.print_feedback"
    ) as print_feedback, patch.object(Project, "save") as save:
        ConcatMenu.upsample_menu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](True)

    assert "LavaSR v2 upsampler not installed" in kwargs["subheading"]
    assert not project.use_upsampler
    save.assert_not_called()
    print_feedback.assert_called_once_with(
        "LavaSR v2 is not installed; generative upsampling cannot be enabled",
        is_error=True,
    )


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

    with patch.object(LavaSrUtil, "has_lava_sr", return_value=False), patch(
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
