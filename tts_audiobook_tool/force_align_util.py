import difflib

from tts_audiobook_tool.app_types import Word
from tts_audiobook_tool.phrase import Phrase
from tts_audiobook_tool.stt_util import normalize_text
from tts_audiobook_tool.timed_phrase import TimedPhrase

class ForceAlignUtil:
    """
    Logic for doing forced alignment between a list of source text phrases and the TTS audio's transcription
    """

    @staticmethod
    def make_timed_phrases(
            phrases: list[Phrase], 
            words: list[Word],
            sound_duration: float
    ) -> list[TimedPhrase]:
        """
        Creates list of `TimedPhrase`s by force-aligning source text `phrases` and transcribed text + timestamps in `words`.
        """
        if not phrases or not words:
            raise ValueError("lists must not be empty")

        if len(phrases) == 1:
            return [ TimedPhrase(phrases[0].text, 0.0, time_end=sound_duration) ]
        
        # The important part
        phrase_strings = [phrase.text for phrase in phrases]
        word_strings = [word.word for word in words]
        boundary_indices = ForceAlignUtil.force_align(phrase_strings, word_strings) 
        # Rem, boundary_indices length is 1 less than that of phrases        
        if not (len(boundary_indices) == len(phrases) - 1):
            raise ValueError("Length of indices must be 1 less than that of phrases")

        timed_phrases: list[TimedPhrase] = []

        for i, phrase in enumerate(phrases):

            time_start = 0.0 if i == 0 else timed_phrases[-1].time_end
            
            if i < len(boundary_indices):
                next_word_index = boundary_indices[i] + 1
                time_end = (words[next_word_index].start if next_word_index < len(words) else words[-1].end)
            else:
                time_end = sound_duration
            
            timed_phrases.append(TimedPhrase(phrase.text, time_start, time_end))

        return timed_phrases

    @staticmethod
    def force_align(source_phrases: list[str], transcribed_words: list[str]) -> list[int]:
        """
        `source_phrases`:
            The list of phrases that makes up the source text.
        
        `transcribed_words`: 
            The flat list of transcribed words.
            Typically quite close in content to the text of `source_phrases`. 

        Returns: 
            A list of "boundary indices", where a "boundary index" is the index from transcribed_words 
            that corresponds to the end of the corresponding source phrase. 
            The length of returned list should of course be equal to len(source_phrases) - 1.

        Example:
            Simple example, where the content of transcribed_words happens to completely match that of source_phrases:
            source_phrases = ["hello", "i'm john", "how are you"]
            transcribed_words = ["hello", "i'm", "john", "how", "are", "you"]
            result: [0, 2]
        """

        # Normalize input text
        source_phrases = [normalize_text(item) for item in source_phrases]
        transcribed_words = [normalize_text(item) for item in transcribed_words]

        boundaries = []

        n_phrases = len(source_phrases)
        n_words = len(transcribed_words)

        if n_phrases == 0:
            return []
        
        if n_phrases == 1:
            # No internal boundaries for a single phrase
            return []
            
        if n_words == 0:
             # Fallback if no words to align against
            return [0] * (n_phrases - 1)

        # dp[k][j] = best score for aligning first k+1 phrases (indices 0..k),
        # where the k-th phrase ends at word index j (inclusive).
        # initialized to -1.0 (since ratio is 0.0 to 1.0)
        dp = [[-1.0] * n_words for _ in range(n_phrases)]
        
        # backtrack[k][j] = the index 'prev_j' (end of phrase k-1) that maximized dp[k][j]
        backtrack = [[-1] * n_words for _ in range(n_phrases)]
        
        # 1. Base Case: First phrase (k=0)
        for j in range(n_words):
             # phrase 0 spans words[0 : j+1]
             score = compare_likeness(source_phrases[0], transcribed_words, 0, j + 1)
             dp[0][j] = score
             
        # 2. DP Step: Phrases k=1 to k=N-1
        for k in range(1, n_phrases):
            
            # For the last phrase, it MUST end at the last word (index n_words-1).
            # For others, they can end anywhere.
            possible_end_indices = range(n_words)
            if k == n_phrases - 1:
                possible_end_indices = [n_words - 1]
            
            for j in possible_end_indices:
                best_score = -1.0
                best_prev = -1
                
                # prev_j is end index of phrase k-1.
                # phrase k corresponds to words[prev_j + 1 : j + 1]
                # prev_j can range from 0 to j.
                for prev_j in range(j + 1):
                    if dp[k-1][prev_j] < 0:
                        continue
                    
                    # Calculate score for current phrase k
                    current_phrase_score = compare_likeness(source_phrases[k], transcribed_words, prev_j + 1, j + 1)
                    total_score = dp[k-1][prev_j] + current_phrase_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_prev = prev_j
                
                dp[k][j] = best_score
                backtrack[k][j] = best_prev
                
        # 3. Backtrack to find boundaries
        boundaries = []
        curr_j = n_words - 1 # Last phrase ends at last word
        
        # Trace back from last phrase down to phrase 1
        for k in range(n_phrases - 1, 0, -1):
            prev_j = backtrack[k][curr_j]
            boundaries.append(prev_j)
            curr_j = prev_j
            
        # The list is [end of N-2, end of N-3, ... end of 0]. Reverse it.
        boundaries.reverse()
        
        return boundaries

# ---

def compare_likeness(
        source_phrase: str, transcribed_words: list[str], transcribed_index_start: int, transcribed_index_end: int
    ) -> float:    
    transcribed_phrase = " ".join(transcribed_words[transcribed_index_start:transcribed_index_end])
    matcher = difflib.SequenceMatcher(None, source_phrase, transcribed_phrase)
    score = matcher.ratio()
    return score
