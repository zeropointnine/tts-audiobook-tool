from tts_audiobook_tool import ask_phrase_groups
from tts_audiobook_tool.app_types import SegmentationStrategy


def test_get_from_std_in_segments_text_from_textual_app(monkeypatch) -> None:
    phrase_groups = [object()]
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        ask_phrase_groups, "run_text_input_app", lambda: "First\nSecond"
    )

    def text_to_groups(raw_text: str, **kwargs):
        calls.append((raw_text, kwargs))
        return phrase_groups

    monkeypatch.setattr(
        ask_phrase_groups.PhraseGrouper, "text_to_groups", text_to_groups
    )

    result = ask_phrase_groups.get_from_std_in(
        42,
        SegmentationStrategy.SENTENCE,
        pysbd_language="en",
        dialog_segmentation=True,
    )

    assert result == (phrase_groups, "First\nSecond")
    assert calls == [
        (
            "First\nSecond",
            {
                "pysbd_lang": "en",
                "max_words": 42,
                "strategy": SegmentationStrategy.SENTENCE,
                "dialog_segmentation": True,
            },
        )
    ]


def test_get_from_std_in_returns_empty_without_segmenting(monkeypatch) -> None:
    segmentation_calls: list[object] = []
    monkeypatch.setattr(ask_phrase_groups, "run_text_input_app", lambda: "")
    monkeypatch.setattr(
        ask_phrase_groups.PhraseGrouper,
        "text_to_groups",
        lambda *args, **kwargs: segmentation_calls.append((args, kwargs)),
    )

    assert ask_phrase_groups.get_from_std_in(
        42,
        SegmentationStrategy.SENTENCE,
        pysbd_language="en",
    ) == ([], "")
    assert segmentation_calls == []
