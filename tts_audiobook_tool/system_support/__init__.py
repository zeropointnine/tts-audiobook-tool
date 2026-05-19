from tts_audiobook_tool.system_support.ansi import Ansi


def get_chromium_info() -> tuple[str, str] | None:
    from tts_audiobook_tool.system_support.browser import get_chromium_info as impl

    return impl()


def get_chromium_user_data_dir() -> str:
    from tts_audiobook_tool.system_support.browser import get_chromium_user_data_dir as impl

    return impl()


def launch_player_with_chromium(
        chromium_path: str,
        audio_file_path: str,
        user_data_dir: str,
) -> None:
    from tts_audiobook_tool.system_support.browser import launch_player_with_chromium as impl

    impl(chromium_path, audio_file_path, user_data_dir)


def has_gui() -> bool:
    from tts_audiobook_tool.system_support.platforms import has_gui as impl

    return impl()


def is_wsl() -> bool:
    from tts_audiobook_tool.system_support.platforms import is_wsl as impl

    return impl()


def open_directory(path: str) -> None:
    from tts_audiobook_tool.system_support.platforms import open_directory as impl

    impl(path)


def clear_input_buffer() -> None:
    from tts_audiobook_tool.system_support.terminal import clear_input_buffer as impl

    impl()


def get_terminal_width() -> int:
    from tts_audiobook_tool.system_support.terminal import get_terminal_width as impl

    return impl()

__all__ = [
    "Ansi",
    "clear_input_buffer",
    "get_chromium_info",
    "get_chromium_user_data_dir",
    "get_terminal_width",
    "has_gui",
    "is_wsl",
    "launch_player_with_chromium",
    "open_directory",
]
