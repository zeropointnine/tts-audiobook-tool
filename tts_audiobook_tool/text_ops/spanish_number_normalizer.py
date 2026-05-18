from text_to_num import text2num  # pyright: ignore[reportMissingImports]


class SpanishNumberNormalizer:
    """
    Deterministic Spanish integer cardinal normalizer (0–9999).

    Instead of calling find_numbers() over the whole text (which produces
    context-dependent, unpredictable matches), this scanner:

    1. Scans tokens left-to-right for maximal contiguous runs of number words
    2. Calls text2num() on each isolated span
    3. Applies policy guards:
       - Range: 0–9999 only
       - Skips ordinals (primero, segundo, etc.)
       - Skips decimals/fractions (coma, punto, medio, etc.)
       - Skips sign indicators (menos, negativo, positivo)
       - Skips standalone "un"/"una" (article/determiner forms)
       - Skips "uno de"/"una de" patterns
       - Skips standalone "uno" unless preceded by "número" or part of
         a multi-token number phrase (e.g. "treinta y uno", "ciento uno")
    4. Replaces only spans that pass all checks
    5. Preserves leading/trailing punctuation on the replaced tokens

    This ensures the same number phrase always normalizes the same way
    regardless of surrounding punctuation or non-number words.
    """

    # All Spanish number words that text2num recognizes as number components.
    # Used both for scanning and for adjacent-word partial-conversion detection.
    _SPANISH_NUMBER_WORDS = {
        # ones
        "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
        "diez", "once", "doce", "trece", "catorce", "quince",
        # teens
        "dieciséis", "dieciseis", "diecisiete", "dieciocho", "diecinueve",
        # twenties
        "veinte", "veintiuno", "veintiuna", "veintidós", "veintidos",
        "veintitrés", "veintitres", "veinticuatro", "veinticinco",
        "veintiséis", "veintiseis", "veintisiete", "veintiocho", "veintinueve",
        # tens
        "treinta", "cuarenta", "cincuenta", "sesenta",
        "setenta", "ochenta", "noventa",
        # hundreds
        "ciento", "cien", "doscientos", "doscientas", "trescientos", "trescientas",
        "cuatrocientos", "cuatrocientas", "quinientos", "quinientas",
        "seiscientos", "seiscientas", "setecientos", "setecientas",
        "ochocientos", "ochocientas", "novecientos", "novecientas",
        # multipliers
        "mil", "miles", "millón", "millones", "billón", "billones",
        "trillón", "trillones",
        # connectors
        "y",
        # decimals / fractions (should not be converted but are number words)
        "coma", "punto", "con", "medio", "media", "cuarto", "tercio",
        # ordinals (should not be converted but are number words)
        "primero", "primera", "segundo", "segunda", "tercero", "tercera", "tercer",
        "cuarto", "cuarta", "quinto", "quinta", "sexto", "séptima", "séptimo",
        "octavo", "octava", "noveno", "novena", "décimo", "décima",
        "último", "última",
        # sign indicators
        "menos", "negativo", "positivo",
    }

    # Spanish ordinal words. Used to block conversion of ordinal phrases.
    _SPANISH_ORDINAL_WORDS = {
        "primero", "primera", "segundo", "segunda", "tercero", "tercera", "tercer",
        "cuarto", "cuarta", "quinto", "quinta", "sexto", "séptima", "séptimo",
        "octavo", "octava", "noveno", "novena", "décimo", "décima",
        "último", "última",
    }

    # Decimal/fraction keywords that indicate the span is not a simple cardinal.
    _DECIMAL_KEYWORDS = {"coma", "punto", "con", "medio", "media", "cuarto", "tercio"}

    # Punctuation characters to strip from tokens before set membership checks.
    _PUNCTUATION = ".,;:!?\"'()[]{}—–-…"

    @classmethod
    def _strip_punct(cls, tok: str) -> str:
        """Strip leading/trailing punctuation and lowercase."""
        return tok.strip(cls._PUNCTUATION).lower()

    @classmethod
    def _find_number_spans(cls, tokens: list[str]) -> list[tuple[int, int]]:
        """
        Find maximal contiguous runs of number-word tokens.
        Returns list of (start_idx, end_idx) where end_idx is exclusive.
        """
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(tokens):
            word = cls._strip_punct(tokens[i])
            if word in cls._SPANISH_NUMBER_WORDS:
                start = i
                i += 1
                while i < len(tokens) and cls._strip_punct(tokens[i]) in cls._SPANISH_NUMBER_WORDS:
                    i += 1
                spans.append((start, i))
            else:
                i += 1
        return spans

    @classmethod
    def _is_safe_cardinal_span(cls, tokens: list[str], start: int, end: int) -> bool:
        """
        Check policy guards for a candidate number span.
        Returns True only if the span should be converted.
        """
        span_words = [cls._strip_punct(tokens[i]) for i in range(start, end)]
        span_text = " ".join(tokens[start:end]).lower()

        # Skip standalone un/una (article/determiner forms)
        if span_text in ("un", "una"):
            return False

        # Skip "uno de" / "una de" patterns
        if span_text.startswith(("uno de", "una de")):
            return False

        # Do not normalize standalone "uno" unless:
        # - it is part of a multi-token number phrase (e.g. "treinta y uno", "ciento uno")
        # - or preceded by an explicit numeric cue (e.g. "número uno")
        if span_text == "uno" and len(span_words) == 1:
            if start == 0 or tokens[start - 1].lower().strip(cls._PUNCTUATION) != "número":
                return False

        # Skip if any token is an ordinal word
        if any(w in cls._SPANISH_ORDINAL_WORDS for w in span_words):
            return False

        # Skip if any token is a decimal/fraction keyword
        if any(w in cls._DECIMAL_KEYWORDS for w in span_words):
            return False

        # Skip sign indicators
        if any(w in {"menos", "negativo", "positivo"} for w in span_words):
            return False

        return True

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Convert spelled-out Spanish integer cardinals (0-9999) to digits.
        """
        from tts_audiobook_tool.l import L

        tokens = text.split()
        if not tokens:
            return text

        spans = cls._find_number_spans(tokens)
        if not spans:
            return text

        any_replaced = False
        replaced_tokens: dict[int, str | None] = {}

        for start, end in spans:
            # Check policy guards before attempting conversion
            if not cls._is_safe_cardinal_span(tokens, start, end):
                continue

            # Build the clean span text (strip punctuation from each word)
            span_clean = " ".join(cls._strip_punct(tokens[i]) for i in range(start, end))

            # Try text2num on the isolated span
            try:
                value = text2num(span_clean, lang="es")
            except (ValueError, TypeError):
                continue

            # Range check: 0-9999 only
            if not isinstance(value, int) or value < 0 or value > 9999:
                continue

            # All checks passed — mark for replacement.
            # Preserve leading/trailing punctuation on the first and last tokens
            # of the span (e.g. "uno." -> "1.", not "1").
            digit_str = str(value)
            first_tok = tokens[start]
            last_tok = tokens[end - 1]

            # Preserve leading punctuation on the first token
            stripped_first = cls._strip_punct(first_tok)
            leading_len = len(first_tok) - len(stripped_first)
            if leading_len > 0:
                # Check if there's leading (not just trailing) punctuation
                # by seeing if the first character of the original is in PUNCTUATION
                if first_tok[0] in cls._PUNCTUATION:
                    digit_str = first_tok[:leading_len] + digit_str

            # Preserve trailing punctuation on the last token
            stripped_last = cls._strip_punct(last_tok)
            trailing_len = len(last_tok) - len(stripped_last)
            if trailing_len > 0:
                # Check if there's trailing (not just leading) punctuation
                if last_tok[-1] in cls._PUNCTUATION:
                    digit_str = digit_str + last_tok[-trailing_len:]

            replaced_tokens[start] = digit_str
            for i in range(start + 1, end):
                replaced_tokens[i] = None  # delete extra tokens

            any_replaced = True

        if not any_replaced:
            return text

        # Rebuild text
        new_tokens: list[str] = []
        for i, tok in enumerate(tokens):
            if i in replaced_tokens:
                replacement = replaced_tokens[i]
                if replacement is not None:
                    new_tokens.append(replacement)
            else:
                new_tokens.append(tok)

        result = " ".join(new_tokens)

        try:
            L.d(f"before: \"{text}\"")
            L.d(f"after:  \"{result}\"")
        except AttributeError:
            pass

        return result