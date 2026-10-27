from __future__ import annotations

from pathlib import Path
import re


INPUT_PATH = Path("assets/words_sp.txt")
OUTPUT_PATH = Path("assets/words_sp_2.txt")
SPANISH_WORD_RE = re.compile(r"^[a-záéíóúüñ]+$", re.IGNORECASE)


def main() -> None:
    words = INPUT_PATH.read_text(encoding="utf-8").splitlines()

    good_words: list[str] = []
    bad_words: list[str] = []

    for word in words:
        stripped_word = word.strip()
        if not stripped_word:
            continue

        if SPANISH_WORD_RE.fullmatch(stripped_word):
            good_words.append(stripped_word)
        else:
            bad_words.append(stripped_word)

    for bad_word in bad_words:
        print(bad_word)

    output_text = "\n".join(good_words) + "\n"
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print()
    print(f"Rejected {len(bad_words)} entries.")
    print(f"Wrote {len(good_words)} cleaned entries to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()