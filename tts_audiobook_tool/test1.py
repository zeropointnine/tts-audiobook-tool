import pickle

from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.stt_util import SttUtil, TranscribedWord
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.text_segmenter import TextSegmenter
from tts_audiobook_tool.util import *


# with open("temp_words.pickle", "rb") as file:
#    words: list[TranscribedWord] = pickle.load(file)

# for i, word in enumerate(words):
#    if i % 100 == 0:
#       print(f"{COL_DIM}{i}{COL_DEFAULT}", end=" ")
#    print(word.word.strip(), end=" ", )
#    if i > 5000:
#       break
# print()

# with open(r"C:\workspace\stt\iron dragons mother\iron dragons mother.txt", "r", encoding="utf-8") as file:
#       raw_text = file.read()
# text_segments = TextSegmenter.segment_text(raw_text, max_words=DEFAULT_MAX_WORDS_PER_SEGMENT)

# timed_text_segments = SttUtil.make_timed_text_segments(text_segments, words)

# discon_ranges = TimedTextSegment.get_discontinuities(timed_text_segments)
