import pysbd
from .sentence_segmenter import SentenceSegmenter

class TextSegmenter:
    """
    Identifies complete sentences from a text stream as it arrives,
    using the 'pysbd' library for robust sentence boundary detection.

    Handles text arriving in arbitrary chunks by buffering input and
    processing the buffer when potential sentence endings are encountered.
    """

    def __init__(self, max_words: int, language="en"):
        """
        Initializes the StreamingSentenceDetector.

        Args:
            language (str): The language code for pysbd (e.g., "en" for English).
        """
        self.max_words = max_words
        self.buffer = ""
        # clean=False prevents pysbd from altering the text (like removing newlines)
        # char_span=False prevents it from returning character spans, we just want text.
        self.segmenter = pysbd.Segmenter(language=language, clean=False, char_span=False)
        # Store terminators for a quick check if processing is potentially needed
        self.terminators = {'.', '?', '!'}

    def add_text(self, text_chunk):
        """
        Adds a chunk of text and identifies any complete sentences formed.

        Args:
            text_chunk (str): The incoming piece of text.

        Returns:
            list[str]: A list of complete sentences identified using the accumulated text.
                       Returns an empty list if no complete sentence was finalized.
        """
        if not isinstance(text_chunk, str):
            # Handle non-string input gracefully
            return []

        self.buffer += text_chunk

        # Optimization: Avoid processing if the buffer is obviously empty or hasn't changed meaningfully
        if not self.buffer.strip():
             # If buffer only contains whitespace after adding chunk, clear it and return
             # Note: This might discard leading whitespace if a sentence starts later.
             # If preserving all whitespace exactly is critical, remove this strip check.
             # self.buffer = "" # Optional: clear whitespace buffer
             return []

        # Process the entire current buffer with pysbd
        potential_sentences = self.segmenter.segment(self.buffer)

        # If pysbd returns nothing or only an empty string (can happen with whitespace), return empty list
        if not potential_sentences or (len(potential_sentences) == 1 and not potential_sentences[0].strip()):
             # It's possible the buffer only contains whitespace or pysbd couldn't segment.
             # We might retain the buffer if it wasn't just whitespace.
            return []

        # --- Logic to decide which sentences are complete ---
        # The core idea: pysbd processes the whole buffer. If the *last* segment
        # it identified looks like a complete sentence (ends with a terminator),
        # we assume *all* segments it returned are complete. If the last segment
        # *doesn't* end properly, we assume it's an incomplete sentence fragment,
        # and only the segments *before* it are complete.

        last_segment = potential_sentences[-1]
        num_potential = len(potential_sentences)
        sentences = []

        # Check if the last segment genuinely ends with a known terminator (ignoring trailing whitespace)
        if last_segment.strip() and last_segment.strip()[-1] in self.terminators:
            # Assume all identified segments are complete sentences
            sentences = potential_sentences
            # The entire buffer was consumed to make these sentences
            self.buffer = ""
        else:
            # The last segment is incomplete. Only return the ones before it (if any).
            if num_potential > 1:
                sentences = potential_sentences[:-1]
                # The remaining buffer *is* the last (incomplete) segment
                self.buffer = last_segment
            else:
                # Only one segment was found, and it's incomplete. Return nothing yet.
                # The buffer (which equals last_segment) remains unchanged.
                sentences = []
                # self.buffer = last_segment # Already true

        # Ensure we don't return empty strings resulting from splitting odd whitespace
        sentences = [s for s in sentences if s.strip()]

        result = []
        for sentence in sentences:
            items = SentenceSegmenter.segment_sentence(sentence, max_words=self.max_words)
            result.extend(items)

        return result

    def get_remaining_text(self):
        """
        Returns the portion of the accumulated text that hasn't been returned
        as a complete sentence yet (i.e., the current buffer content).
        """
        return self.buffer

    @staticmethod
    def segment_full_message(full_message: str, max_words: int) -> list[str]:
        """ 
        Segments a "full message" for synchronous use case.
        Preserves white space before/after each segmented item.
        """
        text_segmenter = TextSegmenter(max_words=max_words)
        result = text_segmenter.add_text(full_message)
        remainder = text_segmenter.get_remaining_text()
        if remainder:
            result.append(remainder)
        return result
