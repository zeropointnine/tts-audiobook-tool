import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tts_audiobook_tool import ask
from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM_ITALICS


@pytest.mark.parametrize(
    ("helper", "dialog_method", "kind"),
    [
        (ask.ask_file_path, "askopenfilename", "file"),
        (ask.ask_dir_path, "askdirectory", "directory"),
    ],
)
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("title", ["Choose a source", "Choose a source:", "Choose a source: "])
def test_gui_path_notice_uses_dialog_title(
    monkeypatch, tmp_path, helper, dialog_method, kind, cancelled, title
):
    # GUI selection and cancellation both show the GUI title, not typing instructions.
    selected_path = "" if cancelled else str(tmp_path)
    dialog = Mock(return_value=selected_path)
    monkeypatch.setitem(
        sys.modules, "tkinter",
        SimpleNamespace(filedialog=SimpleNamespace(**{dialog_method: dialog})),
    )
    output = Mock()
    monkeypatch.setattr(ask, "printt", output)
    fallback = Mock(side_effect=AssertionError("Unexpected console fallback"))
    monkeypatch.setattr(ask, "ask_path_input", fallback)

    assert helper("Enter a path:", title) == selected_path

    assert output.call_args_list[0].args == (
        f"{COL_DIM_ITALICS}GUI {kind} dialog: {COL_DEFAULT}Choose a source",
    )
    # Only the printed notice drops the trailing colon; the GUI title is unchanged.
    assert dialog.call_args.kwargs["title"] == title
    assert len(output.call_args_list) == (1 if cancelled else 3)
    fallback.assert_not_called()


@pytest.mark.parametrize(
    ("helper", "dialog_method", "fallback_kwargs"),
    [
        (ask.ask_file_path, "askopenfilename", {"prefill": ""}),
        (ask.ask_dir_path, "askdirectory", {}),
    ],
)
def test_unavailable_gui_preserves_console_prompt(
    monkeypatch, helper, dialog_method, fallback_kwargs
):
    # A failing GUI still routes the original console instructions to path input.
    dialog = Mock(side_effect=RuntimeError("GUI unavailable"))
    monkeypatch.setitem(
        sys.modules, "tkinter",
        SimpleNamespace(filedialog=SimpleNamespace(**{dialog_method: dialog})),
    )
    monkeypatch.setattr(ask, "printt", Mock())
    fallback = Mock(return_value="")
    monkeypatch.setattr(ask, "ask_path_input", fallback)

    assert helper("Enter a path:", "Choose a source") == ""

    fallback.assert_called_once_with("Enter a path:", **fallback_kwargs)
