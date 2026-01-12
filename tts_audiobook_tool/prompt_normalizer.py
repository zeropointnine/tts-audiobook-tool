import re
from num2words import num2words

from tts_audiobook_tool.text_util import TextUtil

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

        TODO: Strip out 'non-verbal' characters, or at least emojis. Use re backslash-w, etc.
        TODO: Strip out stray punctuation-only "words" which do not seem "caesura-related"
        """

        text = text.strip()

        # Replace fancy double-quotes (important for various models)
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")

        # Collapse multiple triple-dots or multiple ellipses character
        pattern = r'\.{3,}'
        text = re.sub(pattern, '...', text)
        pattern = r'…+'
        text = re.sub(pattern, '…', text)

        # Replace " . . . " with triple-dot
        text = text.replace(" . . . ", " ... ")

        # Strip "punctuation only" words unless ellipsis, dash variants
        words = text.split(" ")
        new_words = []
        for word in words:
            if not TextUtil.is_ws_punc(word):
                new_words.append(word)
            else:
                whitelist = ["...", "…", "\u2014", "\u2013", "-"]
                for item in whitelist:
                    if word == item:
                        new_words.append(word)
                        break
        text = " ".join(new_words)
 
        if un_all_caps:
            text = PromptNormalizer.un_all_caps_prompt(text) 

        if language_code:
            # Expand relatively small "int words" to prevent TTS model from "naively" verbalizing a string of digits
            MAX_VALUE = 999
            try:
                text = re.sub(
                    r'\d+', 
                    lambda x: x.group() if int(x.group()) > MAX_VALUE else num2words(int(x.group()), lang=language_code), 
                    text
                )        
            except NotImplementedError:
                pass # Fail silently

        return text
    
    @staticmethod
    def apply_prompt_word_substitutions(
        prompt: str, substitutions: dict[str, str], language_code: str
    ) -> str:

        prompt_raw_words_original = TextUtil.get_words(prompt)
        prompt_raw_words = []
        
        for prompt_raw_word in prompt_raw_words_original:

            before, prompt_word, after = TextUtil.split_raw_word(prompt_raw_word)
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
                
                if language_code == "en":
                    
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

        words = TextUtil.get_words(prompt)
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
