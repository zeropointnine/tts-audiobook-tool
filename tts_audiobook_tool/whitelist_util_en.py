class WhitelistUtilEn:
    """
    Static utility class for English whitelist dynamic expansion.

    Provides heuristic checks for possessive and plural noun forms
    that are not explicitly in the whitelist but should be trusted.
    """

    @staticmethod
    def has_dynamic(words: set[str], word: str) -> bool:
        """Check if word should be trusted via English-specific rules."""
        if WhitelistUtilEn._is_possessive_or_plural_noun(words, word):
            return True
        return False

    @staticmethod
    def _is_possessive_or_plural_noun(words: set[str], word: str) -> bool:
        """Check possessive ('s) or plural (s) noun forms.

        Returns True if removing the possessive/plural suffix yields
        a word that exists in the whitelist.
        """
        # Possessive noun
        if word.endswith("'s") and len(word) > 2:
            return word[:-2] in words
        # Plural noun
        if word.endswith("s") and len(word) > 1:
            return word[:-1] in words
        return False