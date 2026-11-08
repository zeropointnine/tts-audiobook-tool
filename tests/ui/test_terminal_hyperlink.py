import pytest

from tts_audiobook_tool.text_util import make_terminal_hyperlink, strip_ansi_codes


def test_make_terminal_hyperlink_plain_url_and_file() -> None:
    assert make_terminal_hyperlink("https://x.org/a?b=1;c", "X") == (
        "\x1b]8;;https://x.org/a?b=1;c\x1b\\X\x1b]8;;\x1b\\"
    )
    assert make_terminal_hyperlink("/tmp/a b.txt", is_file=True) == (
        "\x1b]8;;file:///tmp/a b.txt\x1b\\/tmp/a b.txt\x1b]8;;\x1b\\"
    )


@pytest.mark.parametrize(
    ("target", "encoded"),
    [
        # BEL or ESC (including an ST, ESC \) would end the OSC early and let
        # the rest of the target run as new terminal controls.
        ("x\x07\x1b]0;pwn\x07", "x%07%1B]0;pwn%07"),
        ("x\x1b\\y", "x%1B\\y"),
        # 8-bit ST / CSI are encoded as their UTF-8 bytes; DEL too.
        ("x\x9cy\x9b\x7f", "x%C2%9Cy%C2%9B%7F"),
        ("a\nb\tc", "a%0Ab%09c"),
    ],
)
def test_make_terminal_hyperlink_encodes_controls_in_target(target: str, encoded: str) -> None:
    link = make_terminal_hyperlink(target, "label")

    assert link == f"\x1b]8;;{encoded}\x1b\\label\x1b]8;;\x1b\\"
    # Only the builder's own BEL-free open/close sequences remain.
    assert "\x07" not in link
    assert link.count("\x1b") == 4
    assert strip_ansi_codes(link) == "label"
