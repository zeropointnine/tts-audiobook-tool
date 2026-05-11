class WhitelistUtilEs:
    """
    Static utility class for Spanish whitelist dynamic expansion.

    Provides heuristic checks for Spanish noun/adjective gender/number
    variants and verb inflected forms that are not explicitly in the
    whitelist but should be trusted based on stem family analysis.
    """

    # Class-level cache for stem family index
    _stem_family_index: dict[str, int] | None = None

    @staticmethod
    def reset_cache() -> None:
        """Clear the cached stem family index."""
        WhitelistUtilEs._stem_family_index = None

    @staticmethod
    def _ensure_index(words: set[str]) -> None:
        """Build the stem family index if not already cached."""
        if WhitelistUtilEs._stem_family_index is None:
            WhitelistUtilEs._stem_family_index = WhitelistUtilEs._build_stem_family_index(words)

    @staticmethod
    def has_dynamic(words: set[str], word: str) -> bool:
        """Check if word should be trusted via Spanish-specific rules."""
        if WhitelistUtilEs._is_noun(words, word):
            return True
        WhitelistUtilEs._ensure_index(words)
        if WhitelistUtilEs._trusted_by_stem_family(word):
            return True
        return False

    @staticmethod
    def _is_noun(words: set[str], token: str) -> bool:
        """Treat some likely Spanish noun/adjective variants as trusted if a close
        sibling form already exists in the whitelist.

        Covers:
        - gender/number families: o, a, os, as
        - invariant/plural families: e, es

        Only applied when the shared root is at least 4 characters long.
        """
        # Gender/number family
        endings = ["o", "a", "os", "as"]
        for ending in endings:
            if not token.endswith(ending):
                continue
            root = token[:-len(ending)]
            if len(root) < 4:
                continue
            for variant_ending in endings:
                if root + variant_ending in words:
                    return True

        # Invariant/plural family
        endings = ["e", "es"]
        for ending in endings:
            if not token.endswith(ending):
                continue
            root = token[:-len(ending)]
            if len(root) < 4:
                continue
            for variant_ending in endings:
                if root + variant_ending in words:
                    return True

        return False

    @staticmethod
    def _build_stem_family_index(words: set[str]) -> dict[str, int]:
        """Precompute a stem -> count mapping for Spanish verb endings.

        For each word with a known ending, extract the stem (>= 5 chars).
        Then count ALL words in the whitelist that start with that stem and
        are within length tolerance (<= stem_length + 12). Only store stems
        with count >= 2.

        Optimized: scans each word once and checks all its prefixes against
        candidate stems (O(words * avg_word_length)).
        """
        endings = (
            "abamos", "abais", "aban", "abas", "aba",
            "aramos", "arais", "aran", "aras", "ara",
            "eramos", "erais", "eran", "eras", "era",
            "ieramos", "ierais", "ieran", "ieras", "iera",
            "iriamos", "iriais", "irian", "irias", "iria",
            "ariamos", "ariais", "arian", "arias", "aria",
            "iendo", "ando",
            "ado", "ada", "ados", "adas",
            "ido", "ida", "idos", "idas",
        )

        # First pass: extract candidate stems from words with known endings
        candidate_stems: set[str] = set()
        for w in words:
            for ending in endings:
                if w.endswith(ending):
                    stem = w[:-len(ending)]
                    if len(stem) >= 5:
                        candidate_stems.add(stem)

        # Second pass: for each word, check all prefixes against candidate stems
        counts: dict[str, int] = {}
        for w in words:
            wlen = len(w)
            # Check prefixes of length 5 .. wlen (the prefix IS the stem)
            for stem_len in range(5, wlen + 1):
                prefix = w[:stem_len]
                if prefix in candidate_stems:
                    # Check length tolerance: word length <= stem_length + 12
                    if wlen <= stem_len + 12:
                        counts[prefix] = counts.get(prefix, 0) + 1

        # Filter to only stems with count >= 2
        return {stem: c for stem, c in counts.items() if c >= 2}

    @staticmethod
    def _trusted_by_stem_family(token: str) -> bool:
        """Allow some longer Spanish inflected forms when the whitelist already
        contains multiple nearby words from the same stem family.

        Heuristic for plausible missing verb/adjectival forms: if a token
        has a common Spanish ending, yields a reasonably long stem, and that
        stem is witnessed by at least two whitelist entries of similar length,
        we treat the token as trusted.

        Uses a precomputed stem index for O(1) lookup instead of O(n) scan.
        """
        if WhitelistUtilEs._stem_family_index is None:
            return False

        # Only apply to reasonably long tokens
        if len(token) < 7:
            return False

        # Common endings where missing forms are plausible
        endings = (
            "abamos", "abais", "aban", "abas", "aba",
            "aramos", "arais", "aran", "aras", "ara",
            "eramos", "erais", "eran", "eras", "era",
            "ieramos", "ierais", "ieran", "ieras", "iera",
            "iriamos", "iriais", "irian", "irias", "iria",
            "ariamos", "ariais", "arian", "arias", "aria",
            "iendo", "ando",
            "ado", "ada", "ados", "adas",
            "ido", "ida", "idos", "idas",
        )

        for ending in endings:
            if token.endswith(ending):
                stem = token[:-len(ending)]

                # Avoid short accidental stems
                if len(stem) < 5:
                    return False

                # O(1) lookup in precomputed index
                if WhitelistUtilEs._stem_family_index.get(stem, 0) >= 2:
                    return True

        return False