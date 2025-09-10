import os


class Dictionary:
    """
    Literally an English dictionary of words
    Must call init()
    """

    words: set[str]

    @staticmethod
    def init():
        """
        `words.txt` is COCA words list (including word variations)
        https://www.eapfoundation.com/vocab/general/bnccoca
        Plus manually added set of English contraction words ("don't", etc)
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, "assets", "words.txt")

        # Don't catch exception
        with open(path, 'r', encoding='utf-8') as file:
            words_text = file.read()

        lst = words_text.split("\n")
        Dictionary.words = set(lst)


    @staticmethod
    def dict_words_only(words: list[str]) -> list[str]:
        result = []
        for word in words:
            if word in Dictionary.words:
                result.append(word)
            else:
                result.append("#") # Experiment, WIP
        return result

