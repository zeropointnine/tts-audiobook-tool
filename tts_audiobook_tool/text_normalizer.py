import difflib
import re
import string
import unicodedata
from num2words import num2words
from whisper_normalizer.english import EnglishNumberNormalizer

from tts_audiobook_tool.text_util import TextUtil

class TextNormalizer:
    """
    Normalizes source and transcription text so they can be more reliably compared
    """

    _whisper_number_normalizer_en = EnglishNumberNormalizer()

    @staticmethod
    def normalize_common(text: str, language_code: str="") -> str: 
        """
        Normalizes text to be comparable with another piece of text.
        Function should be run on 'both sides'.
        Handles international characters via Unicode-aware regex (backslash-w).
        """
        # Normalize Unicode (fixes 'é' vs 'e'+'´' mismatches)
        text = unicodedata.normalize('NFKC', text)

        # Use casefold for aggressive lowercase (better for Intl text)
        text = text.casefold().strip()

        # Replace fancy apost with normal apost
        text = text.replace("’", "'")

        # Strip apostrophe which is adjacent to word character
        text = re.sub(r"\b'|'\b", "", text)

        # Text-to-numbers
        if language_code == "en":
            # Some forms of punctuation need to remain intact here
            # The exception of dash, which needs be replaced by space
            # (Eg, for some reason, "ninety-nine" does not get transformed to "99", but "ninety nine" does)
            text = re.sub(r'-', ' ', text)
            text = TextNormalizer._whisper_number_normalizer_en(text)

        # Replace punctuation (including em-dash and en-dash) with space
        punc = string.punctuation + "\u2014" + "\u2013" + "…"
        text = re.sub(f"[{re.escape(punc)}]", " ", text) # re.escape is necessary because punc contains special regex chars like [ ] - ^

        # Strip non-word characters except space (rem, non-word characters are not just limited to ascii)
        text = re.sub(r'[^\w\s]', '', text) # TODO: '' or ' ' ?

        # Collapse consecutive whitespace characters into one space
        text = re.sub(r"\s+", " ", text)
        
        # Strip white space from ends
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def normalize_source(source: str, language_code="") -> str:
        return TextNormalizer.normalize_common(source, language_code=language_code)

    @staticmethod
    def normalize_transcript(transcript: str, normalized_source: str, language_code="") -> str:
        """ Note how already-normalized source text is required in this case """
        normalized_transcript = TextNormalizer.normalize_common(transcript, language_code)
        normalized_transcript = TextNormalizer.normalize_spacing(normalized_source, normalized_transcript)
        return normalized_transcript

    @staticmethod
    def normalize_source_and_transcript(source: str, transcript: str, language_code="") -> tuple[str, str]:
        normalized_source = TextNormalizer.normalize_source(source, language_code=language_code)
        normalized_transcript = TextNormalizer.normalize_transcript(
            transcript=transcript, normalized_source=normalized_source, language_code=language_code
        )
        return normalized_source, normalized_transcript

    @staticmethod
    def normalize_spacing(source: str, transcript: str) -> str:
        """
        Normalizes the spacing of the `transcript` to match the `source` text
        wherever the characters align, fixing split/merged compound words.
        """
        # Create a SequenceMatcher to align the two strings
        # autojunk=False is important to prevent spaces from being treated as 'noise'
        matcher = difflib.SequenceMatcher(None, source, transcript, autojunk=False)
        
        normalized_parts = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            
            # 1. MATCH: The text is the same. Keep the transcript text.
            if tag == 'equal':
                normalized_parts.append(transcript[j1:j2])
                
            # 2. DELETE: Text exists in Source but is missing in Transcript.
            # If the missing part was ONLY whitespace, we restore it.
            # (Example: Source "high school" -> Trans "highschool". We restore the space.)
            elif tag == 'delete':
                missing_source_part = source[i1:i2]
                if missing_source_part.isspace():
                    normalized_parts.append(missing_source_part)
                    
            # 3. INSERT: Text exists in Transcript but is missing in Source.
            # If the extra part is ONLY whitespace, we remove it.
            # (Example: Source "firefly" -> Trans "fire fly". We skip the inserted space.)
            elif tag == 'insert':
                inserted_trans_part = transcript[j1:j2]
                if not inserted_trans_part.isspace():
                    normalized_parts.append(inserted_trans_part)
                    
            # 4. REPLACE: The text differs significantly. 
            # Usually implies a typo or a completely different word. 
            # We keep the Transcript version as-is.
            elif tag == 'replace':
                normalized_parts.append(transcript[j1:j2])

        return "".join(normalized_parts)

    @staticmethod
    def sounds_the_same_en(source_word: str, transcript_word: str) -> bool:
        from metaphone import doublemetaphone # is en only 

        # doublemetaphone returns a tuple: (Primary Code, Secondary Code)
        # Example: doublemetaphone("Schmidt") -> ('XMT', 'SMT')
        source_codes = doublemetaphone(source_word)
        transcript_codes = doublemetaphone(transcript_word)

        # Only checking the primary codes; considering secondary code is too permissive for our use case
        return transcript_codes[0] == source_codes[0]

    # ---

    @staticmethod
    def normalize_prompt_common(text: str, language_code: str) -> str:
        """
        Massages source text for inference.
        Similar but not quite the same as source/transcript normalization.
        We want to preserve more "information" in this case (eg, letter case).

        These are operations that should be common to (ie, compatible with) any tts model.
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
    def apply_prompt_word_substitutions(prompt: str, substitutions: dict[str, str], language_code: str) -> str:
        
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
