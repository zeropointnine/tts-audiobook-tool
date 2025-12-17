import os


class DictionaryEn:
    """
    Holds a dict of common English-language words
    Must call init()

    TODO: singleton
    """

    _inited: bool = False
    _words: set[str] = set()

    @staticmethod
    def _init():
        """
        `words_en.txt` is COCA words list (including word variations)
        https://www.eapfoundation.com/vocab/general/bnccoca
        NB, it contains alternative spellings of the same word ("fraternizing" and "fraternising")
        Plus manually added set of English contraction words ("don't", etc)
        Plus added same contraction words without apost
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, "assets", "words_en.txt")

        with open(path, 'r', encoding='utf-8') as file:
            words_text = file.read() # Not catching exception here

        # Assumes one word per line, no leading or trailing whitespace
        lst = words_text.split("\n")
        DictionaryEn._words = set(lst)
        DictionaryEn._inited = True

    @staticmethod
    def has_strict(word: str) -> bool:
        if not DictionaryEn._inited:
            DictionaryEn._init()

        return word in DictionaryEn._words

    @staticmethod
    def has(word: str) -> bool:
        if not DictionaryEn._inited:
            DictionaryEn._init()

        if word in DictionaryEn._words:
            return True
        
        if word.endswith("'s") and len(word) > 2:
            # Special case - possessive 
            word = word[:-2]
            return word in DictionaryEn._words
        if word.endswith("s") and len(word) > 1:
            # Special case - possessive with apostrophe already removed by normalization
            # Not perfect but
            word = word[:-1]
            return word in DictionaryEn._words
        return False
