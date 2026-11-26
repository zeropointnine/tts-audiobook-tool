import re
import unicodedata

from num2words import num2words

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.text_ops import language_util


_DIGIT_RUN_RE = re.compile(r"\d+")
_ENGLISH_NUMBER_RUN_RE = re.compile(r"\d+(?:[,.]\d+)*")
_ENGLISH_GROUPED_INTEGER_RE = re.compile(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+")
_ENGLISH_SPACED_NUMBER_AFFIX_RE = re.compile(
    r"(?<=\$) (?=\d+,\d)|(\d+,\d+(?:[,.]\d+)*) (?=[%$])"
)


def _expand_prompt_numbers(text: str, language_code: str) -> str:
    def expand_digits(digits: str) -> str:
        try:
            value = int(digits)
        except ValueError:
            return digits  # The digit run can exceed Python's integer parsing limit.
        return digits if value > 999 else num2words(value, lang=language_code)

    def expand_small(match: re.Match[str]) -> str:
        return expand_digits(match.group())

    def expand_english(match: re.Match[str]) -> str:
        number = match.group()
        if "," not in number:
            return _DIGIT_RUN_RE.sub(expand_small, number)
        # Consume the whole run first: never expand a substring of a comma-bearing decimal
        # or an ambiguous list/malformed grouping such as 1,234,56.
        if not _ENGLISH_GROUPED_INTEGER_RE.fullmatch(number):
            return number

        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end:end + 1]
        if (before and (before.isalnum() or before in "_.#/+=-−")) or (
            after and (after.isalnum() or after in "_/+=-−")
        ):
            return number

        # Leave grouped currency and percentages to a rule that can speak the whole amount.
        prefix = text[:start].rstrip()[-1:]
        suffix = text[end:].lstrip()[:1]
        if suffix in ("%", "‰", "‱", "٪") or any(
            ch and unicodedata.category(ch) == "Sc" for ch in (prefix, suffix)
        ):
            return number

        # Grouping is formatting; apply the same expansion cutoff as for plain digits.
        return expand_digits(number.replace(",", ""))

    try:
        if language_code == "en":
            return _ENGLISH_NUMBER_RUN_RE.sub(expand_english, text)
        return _DIGIT_RUN_RE.sub(expand_small, text)
    except NotImplementedError:
        return text


class PromptNormalizer:
    """
    Normalizes source text for the TTS prompt
    """

    @staticmethod
    def normalize_prompt(
        text: str, 
        language_code: str,
        un_all_caps: bool=False
    ) -> str:
        """
        Similar but not quite the same as source/transcript normalization.
        We want to preserve more "information" in this case (eg, letter case).

        Model-specific transformations should be done using the model subclasses.

        TODO: Strip out stray punctuation-only "words" which do not seem "caesura-related". Or smth.
        """

        text = app_text.normalize_text_general(text)

        # Collapse multiple triple-dots or multiple ellipses character
        pattern = r'\.{3,}'
        text = re.sub(pattern, '...', text)
        pattern = r'…+'
        text = re.sub(pattern, '…', text)

        # Replace " . . . " with triple-dot
        text = text.replace(" . . . ", " ... ")

        # Use the base code so aliases reach num2words (e.g. "spa" -> "es", not "sp").
        num2words_lang = language_util.normalize_language_code(language_code)
        if num2words_lang == "en":
            # Attach spaced $/% to comma-bearing numbers before punctuation-only tokens are dropped.
            text = _ENGLISH_SPACED_NUMBER_AFFIX_RE.sub(r"\1", text)

        # Keep pause marks and standalone ampersands when filtering punctuation-only words.
        retained_tokens = {"...", "…", "-", "&", *app_text.NON_ASCII_DASHES}
        text = " ".join(
            word for word in text.split(" ")
            if word in retained_tokens or not app_text.is_ws_punc(word)
        )

        if un_all_caps:
            text = PromptNormalizer.un_all_caps_prompt(text)

        # Expand after the all-caps check so an inserted lowercase word can't interrupt it.
        text = app_text.expand_standalone_ampersands(text, language_code)

        if num2words_lang:
            text = _expand_prompt_numbers(text, num2words_lang)

        text = app_text.massage_post_normalize(text)

        return text
    
    @staticmethod
    def apply_prompt_word_substitutions(
        prompt: str, substitutions: dict[str, str], language_code: str
    ) -> str:

        # Alias/region codes (eg "en-US", "eng") reach the English noun-suffix
        # rule, matching how the whitelist and word equivalence subsystems
        # resolve language codes
        is_english = language_util.is_english(language_code)

        prompt_raw_words_original = app_text.get_words(prompt)
        prompt_raw_words = []
        
        for prompt_raw_word in prompt_raw_words_original:

            before, prompt_word, after = app_text.split_raw_word(prompt_raw_word)
            prompt_word_lc = prompt_word.lower()

            for reference_word, substitute_word in substitutions.items():

                reference_word_lc = reference_word.lower()

                if prompt_word_lc == reference_word_lc:
                    # Direct, case-insensitive match
                    should_make_upper = substitute_word[0].islower() and prompt_word[0].isupper()
                    if should_make_upper: 
                        substitute_word = substitute_word[0].upper() + substitute_word[1:]
                    prompt_word = substitute_word
                    break
                
                if is_english:
                    
                    TREAT_AS_NOUN = True # bc why not
                    if TREAT_AS_NOUN: 

                        SUFFIXES = ["s", "es", "'s", "s'"]
                        for suffix in SUFFIXES:
                            if prompt_word_lc.endswith(suffix) and len(prompt_word_lc) > len(suffix):
                                root_prompt_word_lc = prompt_word_lc[:-len(suffix)]
                                if root_prompt_word_lc == reference_word_lc:
                                    # A match
                                    should_make_upper = substitute_word[0].islower() and prompt_word[0].isupper()
                                    if should_make_upper: 
                                        substitute_word = substitute_word[0].upper() + substitute_word[1:]
                                    prompt_word = substitute_word + suffix
                                    break

            prompt_raw_words.append(before + prompt_word + after)
        new_prompt = "".join(prompt_raw_words)
        return new_prompt
    
    @staticmethod
    def un_all_caps_prompt(prompt: str) -> str:
        """
        If the first n words with alpha chars are all-caps, 
        force lowercase until a word with lowercase is reached.

        Some books stylize in all caps the first several words or full sentence of a new section. 
        Many TTS models fall down when words are formatted like that.
        """
        THRESHOLD = 3

        words = app_text.get_words(prompt)
        should = False
        count = 0

        for word in words:
            has_alpha = any(char.isalpha() for char in word) 
            if not has_alpha:
                continue
            has_lower = any(char.islower() for char in word)
            if has_lower:
                return prompt
            count += 1
            if count >= THRESHOLD:
                should = True
                break

        if not should:
            return prompt
        
        # Transform

        new_words = []
        force_lower = True
        for word in words:
            if force_lower:
                has_lower = any(char.islower() for char in word)
                if has_lower:
                    force_lower = False
            if force_lower:
                word = word.lower()
            new_words.append(word)

        return "".join(new_words)
