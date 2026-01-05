from __future__ import annotations
import math
import string
import pysbd

from tts_audiobook_tool.phrase import Phrase, Reason
from tts_audiobook_tool.text_util import TextUtil


class PhraseSegmenter:
    """ 
    Creates Phrases from strings
    """

    @staticmethod
    def text_to_phrases(text: str, max_words: int, pysbd_lang: str, ) -> list[Phrase]:
        """
        Returns list of Phrases with 'reasons', ready to be grouped as needed
        """
        print()

        phrases: list[Phrase] = []

        # Text to sentence strings
        sentence_strings = PhraseSegmenter.string_to_sentence_strings(text, pysbd_lang)

        for sentence_string in sentence_strings:

            # Sentence strings to phrase strings
            phrase_strings = PhraseSegmenter.sentence_string_to_phrase_strings(sentence_string)

            # Make Phrases proper, disambiguating btw sentence/paragraph/section
            for i, phrase in enumerate(phrase_strings):
                is_last_phrase_in_sentence = (i == len(phrase_strings) - 1)
                if is_last_phrase_in_sentence:
                    num_lf = TextUtil.num_trailing_line_breaks(phrase)
                    match num_lf:
                        case 0:
                            reason = Reason.SENTENCE
                        case 1:
                            reason = Reason.PARAGRAPH
                        case 2:
                            reason = Reason.PARAGRAPH
                        case _: # >= 3
                            reason = Reason.SECTION
                else:
                    reason = Reason.PHRASE
                phrases.append( Phrase(phrase, reason) )

        # Split long phrases if necessary
        new_result: list[Phrase] = []
        for phrase in phrases:
            phrases = PhraseSegmenter.long_phrase_to_phrases(phrase, max_words)
            new_result.extend(phrases)
        phrases = new_result

        return phrases

    @staticmethod
    def string_to_sentence_strings(source: str, pysbd_lang: str) -> list[str]:
        """ 
        Segments source text into sentences using pysbd lib, preserving all characters.
        """

        from pysbd.languages import Language
        try:
            _ = Language.get_language_code(pysbd_lang)
        except:
            pysbd_lang = "en" # fail silently

        # Segment text into sentences using pysbd
        # Important: "clean=False" preserves leading and trailing whitespace
        segmenter = pysbd.Segmenter(language=pysbd_lang, clean=False, char_span=False)
        sentences = segmenter.segment(source)

        def merge_danging_punc_word(sentences: list[str]) -> list[str]:
            # TODO: Still not fully resolved, even aside from linefeed bug
            # pysbd can create danging punc-only sentences
            # eg:: "And you can . . . Yes?" -> "And you can . ", ". . ", "Yes?"
                        
            result: list[str] = []
            for sentence in sentences:
                if result and TextUtil.is_ws_punc(sentence):                    
                    # TODO even more, fml
                    #   pysbd can replace linefeed with space. 
                    #   eg: "And you can . . .\nYes?" 
                    result[-1] += sentence
                else:
                    result.append(sentence)
            return result

        sentences = merge_danging_punc_word(sentences) # type: ignore

        # pysbd treats everything enclosed in quotes as a single sentence, so split those up
        new_sentences = []
        for string in sentences:
            if is_sentence_quotation(string):
                inner_sentences = segment_quote_sentence(string, segmenter)
                inner_sentences = merge_danging_punc_word(inner_sentences) # type: ignore
                new_sentences.extend(inner_sentences)
            else:
                new_sentences.append(string)
        sentences = new_sentences

        return sentences

    @staticmethod
    def sentence_string_to_phrase_strings(sentence: str) -> list[str]:
        """ Returns list of phrase strings from a sentence string """

        # Segment at phrase delimiter
        items = []
        string = ""
        for char in sentence:
            string += char
            if char in PHRASE_DELIMITERS:
                items.append(string)
                string = ""
        if string:
            items.append(string)

        if len(items) <= 1:
            return items

        # Move starting whitespace to end of previous item
        for i in range(1, len(items)):
            leading_ws, remainder = split_leading_ws_punc(items[i])
            items[i - 1] += leading_ws
            items[i] = remainder

        # Combine any ws-punc items to predecessor (edge case)
        new_items = [ items[0] ]
        for i in range(1, len(items)):
            if TextUtil.is_ws_punc(items[i]):
                items[i-1] += items[i]
            else:
                new_items.append(items[i])
        items = new_items

        return items

    @staticmethod
    def long_phrase_to_phrases(phrase: Phrase, max_words:int) -> list[Phrase]:
        """
        Splits a phrase at arbitrary word if it exceeds max_words
        """
        num_phrases = math.ceil(phrase.num_words / max_words)
        stride = math.ceil(phrase.num_words / num_phrases)

        result = []
        for i in range(0, num_phrases):

            # TODO: if not at last phrase, if end of phrase is from a set of certain common words, move split point back 1 (eg, common particles, prepositions, pronouns, conjunction words...)

            start = (i + 0) * stride
            end   = (i + 1) * stride
            words = phrase.words[start:end]
            reason = phrase.reason if (i == num_phrases - 1) else Reason.WORD
            new_phrase = Phrase("".join(words), reason)
            result.append(new_phrase)
        return result

# ---

def is_sentence_quotation(pysbd_segmented_string: str) -> bool:
    """
    Given a pysbd-segmented string, does it appear to be a "quotation sentence"
    (ie, a candidate for further segmentation).
    Tested on english only for now.
    """

    def has_start_quote_char(chunk: str) -> bool:
        for quote_char in "\"“": # normal-double-quote and fancy-starting-double-quote
            if quote_char in chunk:
                return True
        return False

    def has_end_quote_char(chunk: str) -> bool:
        for quote_char in "\"”": # normal-double-quote and fancy-ending-double-quote
            if quote_char in chunk:
                return True
        return False

    start, _, end = split_string_parts(pysbd_segmented_string)
    return has_start_quote_char(start) and has_end_quote_char(end)

def segment_quote_sentence(sentence: str, segmenter) -> list[str]:
    """
    Given a quote which may consist of multiple sentences and may have whitespace before and/or after the quote,
    segment the inside of the quote by sentence, preserving whitespace.
    """
    before, content, after = split_string_parts(sentence)
    inner_sentences = segmenter.segment(content)
    if not inner_sentences:
        return [sentence]

    inner_sentences[0] = before + inner_sentences[0]
    inner_sentences[-1] = inner_sentences[-1] + after
    return inner_sentences

def split_string_parts(text: str) -> tuple[str, str, str]:
    """
    Splits a string into three parts:
    - before: Leading whitespace + first non-whitespace character
    - content: Everything between before and after
    - after: Last non-whitespace character + trailing whitespace

    If there is only one non-whitespace character,
    it should be assigned to "before", and "content" should be empty.

    If there are only two non-whitespace characters,
    the first character should be assigned to "before",
    the second character should be assigned to "after",
    and "content" should be empty.

    Returns:
        Tuple of (before, content, after)
    """
    if not text:
        return ('', '', '')

    stripped_text = text.strip()

    # Handle strings that are empty or contain only whitespace.
    # In this case, 'before' contains the whole string.
    if not stripped_text:
        return (text, '', '')

    # Find the indices of the first and last non-whitespace characters.
    # This is a more direct and robust way to find the split points.
    first_char_index = text.find(stripped_text[0])
    last_char_index = text.rfind(stripped_text[-1])

    # If the first and last non-whitespace character is the same,
    # it fully belongs to 'before' as per the docstring.
    if first_char_index == last_char_index:
        before = text[:first_char_index + 1]
        content = ''
        after = text[first_char_index + 1:]
        return (before, content, after)

    # For all other cases (2 or more non-whitespace characters),
    # slice the string based on the found indices.
    before = text[:first_char_index + 1]
    content = text[first_char_index + 1:last_char_index]
    after = text[last_char_index:]

    return (before, content, after)

def split_leading_whitespace(s: str) -> tuple[str, str]:
    lstripped_string = s.lstrip()
    leading_whitespace_len = len(s) - len(lstripped_string)
    return s[:leading_whitespace_len], s[leading_whitespace_len:]


def split_leading_ws_punc(input_string: str) -> tuple[str, str]:
  for i, char in enumerate(input_string):
    if char not in WHITESPACE_PUNCTUATION:
      return (input_string[:i], input_string[i:])
  return (input_string, "")


# Characters that are either whitespace or punctuation
WHITESPACE_PUNCTUATION = set(string.whitespace + string.punctuation) # TODO this is incomplete; reconcile with other related usages

# comma, semicolon, colon, en-dash, em-dash, open paren, close paren
# not including normal dash, apostrophe/single-quote or double-quote
PHRASE_DELIMITERS = r'[(),;:\–\—]'
