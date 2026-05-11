import re

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.constants import COL_DEFAULT, COL_ERROR
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.whitelist import Whitelist


def highlight_failed_words_normalized(text: str) -> str:
    chunk_results: list[str] = []

    groups = PhraseGrouper.text_to_groups(
        text=text,
        max_words=40,
        strategy=SegmentationStrategy.NORMAL,
        pysbd_lang="es",
    )

    for group in groups:
        source = group.as_flattened_phrase().text
        normalized = TextNormalizer.normalize_source(source, language_code="es")

        parts: list[str] = []
        for token in TextUtil.get_words(normalized):
            leading, middle, trailing = TextUtil.split_raw_word(token)
            if not middle:
                parts.append(token)
                continue

            if not Whitelist().has(middle):
                parts.append(f"{leading}{COL_ERROR}[{middle}]{COL_DEFAULT}{trailing}")
                continue

            parts.append(token)

        chunk_results.append("".join(parts))

    return "\n\n".join(chunk_results)


def main() -> None:
    Whitelist().set_language_code("es")

    print("Enter Spanish text:")
    text = AskUtil.ask_multiline()
    if not text:
        return
    
    print("-" * 80)
    print(highlight_failed_words_normalized(text))
    print()


if __name__ == "__main__":
    main()