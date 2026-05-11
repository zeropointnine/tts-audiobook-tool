import os

from tts_audiobook_tool.whitelist_util_en import WhitelistUtilEn
from tts_audiobook_tool.whitelist_util_es import WhitelistUtilEs


class Whitelist:
    """
    Singleton holding a dict of common words for the currently selected language.
    Use Whitelist() to get the singleton.

    ---

    Notes on English word list:

    `words_en.txt` is COCA words list (including word variations)
    https://www.eapfoundation.com/vocab/general/bnccoca
    NB, it contains alternative spellings of the same word ("fraternizing" and "fraternising")
    Plus manually added set of English contraction words ("don't", etc)
    Plus added same contraction words without apost.
    Does not contain plural or possessive noun forms.

    See docs w/r/t Spanish word list
    """

    _instance: "Whitelist | None" = None

    LANGUAGES = {
        "en": "whitelist_english.txt",
        "es": "whitelist_spanish.txt",
    }

    words: set[str]
    current_language_code: str = ""

    def __new__(cls) -> "Whitelist":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_language_code = ""
            cls._instance.words = set()
        return cls._instance

    def __init__(self):
        """Initialize singleton state only; language data is loaded on demand."""

    def _load_words(self, language_code: str) -> set[str]:
        file_name = self.LANGUAGES.get(language_code, "")
        if not file_name:
            return set()

        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, "assets", file_name)

        with open(path, 'r', encoding='utf-8') as file:
            # Note, extant file is a requirement. No catching of exception.
            return set(file.read().splitlines())

    @staticmethod
    def normalize_language_code(language_code: str) -> str:
        code = language_code.strip().lower()
        if not code:
            return ""
        return code.split("-", 1)[0]

    @staticmethod
    def supports_language(language_code: str) -> bool:
        code = Whitelist.normalize_language_code(language_code)
        return code in Whitelist.LANGUAGES

    def set_language_code(self, language_code: str) -> None:
        code = Whitelist.normalize_language_code(language_code)
        if not Whitelist.supports_language(code):
            code = ""

        if self.current_language_code == code:
            return

        self.current_language_code = code
        self.words = self._load_words(code)
        WhitelistUtilEs.reset_cache()

    def has(self, word: str) -> bool:
        if word in self.words:
            return True

        if self.current_language_code not in Whitelist.LANGUAGES.keys():
            return False

        # Special language-specific word variants that do not exist in whitelist
        # but should be treated as if they are.
        match self.current_language_code:
            case "en":
                if WhitelistUtilEn.has_dynamic(self.words, word):
                    return True
            case "es":
                if WhitelistUtilEs.has_dynamic(self.words, word):
                    return True

        return False

    def has_strict(self, word: str) -> bool:
        """Is word in whitelist, no 'dynamic expansion'."""
        return word in self.words
