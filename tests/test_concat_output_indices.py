from types import SimpleNamespace
from unittest.mock import patch

from tts_audiobook_tool.menus import concat_menu


def test_ask_output_indices_empty_input_cancels() -> None:
    infos = [
        SimpleNamespace(output_index=0, num_files_exist=1),
        SimpleNamespace(output_index=1, num_files_exist=0),
    ]

    with patch.object(concat_menu.ask, "ask", return_value=""), patch.object(
        concat_menu, "printt"
    ), patch.object(concat_menu, "print_feedback") as print_feedback:
        result = concat_menu.ask_output_indices(infos)  # type: ignore[arg-type]

    assert result is None
    print_feedback.assert_not_called()
