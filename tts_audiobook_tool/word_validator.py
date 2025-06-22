import os


class WordsValidator:
    """
    WORK IN PROGRESS
    """

    words: set[str]

    @staticmethod
    def init():
        """
        COCA words list plus contractions added
        https://www.eapfoundation.com/vocab/general/bnccoca
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, "assets", "words.txt")

        # Don't catch exception
        with open(path, 'r', encoding='utf-8') as file:
            words_text = file.read()

        lst = words_text.split("\n")
        WordsValidator.words = set(lst)


    @staticmethod
    def filter_out(words: list[str]) -> list[str]:
        result = []
        for word in words:
            if word in WordsValidator.words:
                result.append(word)
            else:
                result.append("#")
        return result

    @staticmethod
    def are_equal(source_words: list[str], trans_words: list[str]):
        return source_words == trans_words
