"""Search-only equivalences shared by editor and worker-log Find."""

import unicodedata


_FIND_TRANSLATION = str.maketrans({
    **dict.fromkeys("‘’‚‛‹›", "'"),
    **dict.fromkeys("“”„‟«»", '"'),
    **dict.fromkeys("‐‑‒–—", "-"),
    "\u00a0": " ",  # Nonbreaking space.
    "\u202f": " ",  # Narrow nonbreaking space.
    "…": "...",
})


def normalize_find_text(text: str) -> str:
    """Fold case and typographic variants without changing source content.

    Keep accents, other punctuation, and whitespace significant. NFC handles
    canonically equivalent Unicode without NFKC's broader compatibility folds.
    The result is for substring matching, not indexing into the original text.
    """
    text = unicodedata.normalize("NFC", text).translate(_FIND_TRANSLATION)
    return unicodedata.normalize("NFC", text.casefold())
